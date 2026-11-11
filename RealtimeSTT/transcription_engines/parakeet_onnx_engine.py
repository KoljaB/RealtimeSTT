"""CPU-only dynamic INT8 Parakeet with the exported NeMo frontend."""
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
import threading
import time
import numpy as np
from .base import BaseTranscriptionEngine, TranscriptionEngineError, TranscriptionInfo, TranscriptionResult


class _RequestSession:
    def __init__(self, session, owner):
        self.session, self.owner = session, owner

    def __getattr__(self, name):
        return getattr(self.session, name)

    def run(self, outputs, inputs, **kwargs):
        self.owner._check_cancelled()
        kwargs['run_options'] = self.owner._run_options
        result = self.session.run(outputs, inputs, **kwargs)
        self.owner._check_cancelled()
        return result


class ParakeetOnnxEngine(BaseTranscriptionEngine):
    """Resident greedy TDT recognizer, serialized and request-cancellable."""
    engine_name = 'parakeet_onnx'

    def __init__(self, config, *, ort_module=None, model_cls=None):
        super().__init__(config)
        if str(config.device).lower() != 'cpu':
            raise TranscriptionEngineError('parakeet_onnx requires device=cpu')
        options = dict(config.engine_options or {})
        unknown = set(options) - {'num_threads', 'decoder_threads', 'preprocessor_threads', 'allow_spinning'}
        if unknown:
            raise TranscriptionEngineError(f'Unknown parakeet_onnx options: {sorted(unknown)}')
        threads = {}
        for key, default in [('num_threads', 7), ('decoder_threads', 1), ('preprocessor_threads', 1)]:
            value = options.get(key, default)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise TranscriptionEngineError(f'{key} must be a positive integer')
            threads[key] = value
        spin = options.get('allow_spinning', False)
        if not isinstance(spin, bool):
            raise TranscriptionEngineError('allow_spinning must be a boolean')
        root = Path(config.model).expanduser().resolve()
        files = {key: root / name for key, name in {
            'config': 'config.json', 'vocab': 'vocab.txt',
            'encoder': 'encoder-model.int8.onnx', 'decoder_joint': 'decoder_joint-model.int8.onnx',
        }.items()}
        frontend = root / 'nemo128.onnx'
        for path in [*files.values(), frontend]:
            if not path.is_file() or not path.stat().st_size:
                raise TranscriptionEngineError(f'Missing Parakeet ONNX file: {path}')
        try:
            self.ort = ort_module or import_module('onnxruntime')
            model_cls = model_cls or import_module('onnx_asr.models.nemo').NemoConformerTdt
        except ImportError as exc:
            raise TranscriptionEngineError('Install realtimestt[parakeet-onnx] for CPU Parakeet') from exc
        self._lock, self._active_lock = threading.Lock(), threading.Lock()
        self._cancel_event = self._run_options = None
        self._closed = False

        def opts(count):
            o = self.ort.SessionOptions()
            o.intra_op_num_threads, o.inter_op_num_threads = count, 1
            o.execution_mode = self.ort.ExecutionMode.ORT_SEQUENTIAL
            o.graph_optimization_level = self.ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            o.add_session_config_entry('session.intra_op.allow_spinning', str(int(spin)))
            o.add_session_config_entry('session.inter_op.allow_spinning', '0')
            return o

        providers = ['CPUExecutionProvider']
        prep = self.ort.InferenceSession(str(frontend), sess_options=opts(threads['preprocessor_threads']), providers=providers)
        self._preprocessor = _RequestSession(prep, self)
        inputs = prep.get_inputs()

        def preprocess(w, lengths):
            return tuple(self._preprocessor.run(None, {inputs[0].name: w, inputs[1].name: lengths}))

        self.model = model_cls(files, lambda _: preprocess, {
            'sess_options': opts(threads['decoder_threads']), 'providers': providers,
        })
        # Replace the small initial encoder pool; keep the decoder single-threaded.
        self.model._encoder = _RequestSession(self.ort.InferenceSession(
            str(files['encoder']), sess_options=opts(threads['num_threads']), providers=providers,
        ), self)
        self.model._decoder_joint = _RequestSession(self.model._decoder_joint, self)
        self.runtime_metadata = {
            'engine': self.engine_name, 'provider': 'cpu', 'model_path': str(root),
            'quantization': 'dynamic_int8', 'preprocessor': 'nemo128.onnx', 'decoding': 'greedy',
            **threads, 'allow_spinning': spin,
        }
        self.backend = SimpleNamespace(runtime_metadata=self.runtime_metadata)

    def _check_cancelled(self):
        if self._cancel_event is not None and self._cancel_event.is_set():
            raise TranscriptionEngineError('Parakeet ONNX transcription cancelled')

    def cancel_active(self):
        with self._active_lock:
            if self._run_options is None:
                return False
            self._cancel_event.set()
            self._run_options.terminate = True
            return True

    def transcribe_cancellable(self, audio, language=None, use_prompt=True, *, cancel_event, **kwargs):
        return self.transcribe(audio, language, use_prompt, _cancel_event=cancel_event, **kwargs)

    def transcribe(self, audio, language=None, use_prompt=True, **kwargs):
        if kwargs.get('word_timestamps'):
            raise TranscriptionEngineError('parakeet_onnx does not expose word timestamps')
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
            raise TranscriptionEngineError('Expected nonempty finite mono float32 PCM at 16000 Hz')
        audio = np.ascontiguousarray(self._normalize_audio(audio))
        with self._lock:
            if self._closed:
                raise TranscriptionEngineError('parakeet_onnx is closed')
            with self._active_lock:
                self._cancel_event = kwargs.get('_cancel_event') or threading.Event()
                self._run_options = self.ort.RunOptions()
            started = time.perf_counter()
            try:
                self._check_cancelled()
                result = list(self.model.recognize_batch(audio[None, :], np.array([audio.size], dtype=np.int64)))
                self._check_cancelled()
                text = result[0].text.strip()
            except Exception as exc:
                if isinstance(exc, TranscriptionEngineError):
                    raise
                raise TranscriptionEngineError(f'Parakeet ONNX inference failed: {exc}') from exc
            finally:
                with self._active_lock:
                    self._run_options = self._cancel_event = None
            return TranscriptionResult(text=text, info=TranscriptionInfo(), metadata={
                **self.runtime_metadata, 'inference_ms': (time.perf_counter() - started) * 1000,
                'timestamp_kind': 'none',
            })

    def close(self):
        self.cancel_active()
        with self._lock:
            self._closed = True
            self.model = self._preprocessor = None

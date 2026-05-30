"""
Silero VAD backend adapters.

The recorder processes one 32 ms / 512-sample stream at a time. For that
workload, PyTorch CUDA launch overhead is usually larger than the tiny recurrent
Silero model itself, so the automatic backend prefers CPU ONNX Runtime and only
uses CUDA when explicitly requested.
"""

import logging
import os
from importlib import import_module
from importlib.util import find_spec

import numpy as np


SILERO_BACKEND_AUTO = "auto"
SILERO_BACKEND_LEGACY = "legacy"
SILERO_BACKEND_PYTORCH_CPU = "pytorch_cpu"
SILERO_BACKEND_PYTORCH_CUDA = "pytorch_cuda"
SILERO_BACKEND_OFFICIAL_ONNX = "official_onnx"
SILERO_BACKEND_RAW_ONNX = "raw_onnx"
SILERO_BACKEND_RAW_ONNX_IFLESS = "raw_onnx_ifless"

SILERO_OP18_IFLESS_MODEL = "silero_vad_op18_ifless.onnx"
SILERO_ONNX_MODEL = "silero_vad.onnx"


_BACKEND_ALIASES = {
    "": SILERO_BACKEND_AUTO,
    "auto": SILERO_BACKEND_AUTO,
    "fast": SILERO_BACKEND_AUTO,
    "legacy": SILERO_BACKEND_LEGACY,
    "default": SILERO_BACKEND_LEGACY,
    "torch_hub": SILERO_BACKEND_LEGACY,
    "torchhub": SILERO_BACKEND_LEGACY,
    "pytorch": SILERO_BACKEND_PYTORCH_CPU,
    "torch": SILERO_BACKEND_PYTORCH_CPU,
    "pytorch_cpu": SILERO_BACKEND_PYTORCH_CPU,
    "torch_cpu": SILERO_BACKEND_PYTORCH_CPU,
    "cpu": SILERO_BACKEND_PYTORCH_CPU,
    "pytorch_cuda": SILERO_BACKEND_PYTORCH_CUDA,
    "torch_cuda": SILERO_BACKEND_PYTORCH_CUDA,
    "cuda": SILERO_BACKEND_PYTORCH_CUDA,
    "onnx": SILERO_BACKEND_OFFICIAL_ONNX,
    "onnx_official": SILERO_BACKEND_OFFICIAL_ONNX,
    "official_onnx": SILERO_BACKEND_OFFICIAL_ONNX,
    "silero_onnx": SILERO_BACKEND_OFFICIAL_ONNX,
    "onnx_raw": SILERO_BACKEND_RAW_ONNX,
    "raw_onnx": SILERO_BACKEND_RAW_ONNX,
    "onnxruntime": SILERO_BACKEND_RAW_ONNX,
    "onnx_runtime": SILERO_BACKEND_RAW_ONNX,
    "ort": SILERO_BACKEND_RAW_ONNX,
    "onnx_raw_ifless": SILERO_BACKEND_RAW_ONNX_IFLESS,
    "raw_onnx_ifless": SILERO_BACKEND_RAW_ONNX_IFLESS,
    "op18_ifless": SILERO_BACKEND_RAW_ONNX_IFLESS,
    "ifless": SILERO_BACKEND_RAW_ONNX_IFLESS,
}


class SileroVadError(RuntimeError):
    """Raised when a requested Silero VAD backend cannot be loaded."""


def normalize_silero_backend(backend):
    key = str(backend or "").strip().lower().replace("-", "_")
    try:
        return _BACKEND_ALIASES[key]
    except KeyError:
        choices = sorted(set(_BACKEND_ALIASES.values()))
        raise ValueError(
            "Unknown Silero VAD backend %r. Expected one of: %s"
            % (backend, ", ".join(choices))
        )


def resolve_silero_backend(backend="auto", silero_use_onnx=None):
    """Resolve the new backend option with the legacy silero_use_onnx flag."""
    resolved = normalize_silero_backend(backend)
    if resolved != SILERO_BACKEND_AUTO:
        return resolved
    if silero_use_onnx is True:
        return SILERO_BACKEND_LEGACY
    if silero_use_onnx is False:
        return SILERO_BACKEND_LEGACY
    return SILERO_BACKEND_AUTO


def create_silero_vad_model(
    backend="auto",
    silero_use_onnx=None,
    onnx_model_path=None,
    onnx_threads=2,
    sample_rate=16000,
    chunk_samples=512,
    logger=None,
):
    """Create a Silero VAD adapter.

    Automatic selection order is:
    raw ONNX Runtime op18-ifless, raw ONNX Runtime regular model, PyTorch CPU.
    CUDA is deliberately excluded from auto because it is slower for the
    recorder's single-stream 512-sample chunks on common GPUs.
    """
    log = logger or logging.getLogger("realtimestt")
    requested = resolve_silero_backend(backend, silero_use_onnx)

    if requested == SILERO_BACKEND_AUTO:
        return _create_auto_vad(
            onnx_model_path=onnx_model_path,
            onnx_threads=onnx_threads,
            sample_rate=sample_rate,
            chunk_samples=chunk_samples,
            logger=log,
        )
    if requested == SILERO_BACKEND_LEGACY:
        return _create_legacy_vad(bool(silero_use_onnx))
    if requested == SILERO_BACKEND_PYTORCH_CPU:
        return _create_pytorch_vad("cpu")
    if requested == SILERO_BACKEND_PYTORCH_CUDA:
        return _create_pytorch_vad("cuda")
    if requested == SILERO_BACKEND_OFFICIAL_ONNX:
        return _create_official_onnx_vad(onnx_model_path)
    if requested == SILERO_BACKEND_RAW_ONNX:
        return _create_raw_onnx_vad(
            SILERO_ONNX_MODEL,
            SILERO_BACKEND_RAW_ONNX,
            onnx_model_path,
            onnx_threads,
            sample_rate,
            chunk_samples,
        )
    if requested == SILERO_BACKEND_RAW_ONNX_IFLESS:
        return _create_raw_onnx_vad(
            SILERO_OP18_IFLESS_MODEL,
            SILERO_BACKEND_RAW_ONNX_IFLESS,
            onnx_model_path,
            onnx_threads,
            sample_rate,
            chunk_samples,
        )
    raise AssertionError("Unhandled Silero backend %r" % requested)


def _create_auto_vad(
    onnx_model_path,
    onnx_threads,
    sample_rate,
    chunk_samples,
    logger,
):
    errors = []
    candidates = (
        (
            SILERO_OP18_IFLESS_MODEL,
            SILERO_BACKEND_RAW_ONNX_IFLESS,
            onnx_model_path,
        ),
        (
            SILERO_ONNX_MODEL,
            SILERO_BACKEND_RAW_ONNX,
            onnx_model_path,
        ),
    )
    for model_name, backend_name, explicit_path in candidates:
        try:
            return _create_raw_onnx_vad(
                model_name,
                backend_name,
                explicit_path,
                onnx_threads,
                sample_rate,
                chunk_samples,
            )
        except Exception as exc:
            errors.append("%s: %s" % (backend_name, exc))
            logger.debug(
                "Silero VAD auto backend %s unavailable",
                backend_name,
                exc_info=True,
            )
            if explicit_path:
                break

    try:
        return _create_pytorch_vad("cpu")
    except Exception as exc:
        errors.append("%s: %s" % (SILERO_BACKEND_PYTORCH_CPU, exc))
        raise SileroVadError(
            "Could not initialize any automatic Silero VAD backend. "
            "Tried: %s" % "; ".join(errors)
        )


def _create_raw_onnx_vad(
    model_name,
    backend_name,
    onnx_model_path,
    onnx_threads,
    sample_rate,
    chunk_samples,
):
    if int(sample_rate) != 16000 or int(chunk_samples) != 512:
        raise SileroVadError(
            "%s requires 16 kHz audio and 512-sample chunks; got %s Hz and %s "
            "samples. Use silero_backend='pytorch_cpu' or keep buffer_size=512."
            % (backend_name, sample_rate, chunk_samples)
        )

    model_path = onnx_model_path or find_silero_model_file(model_name)
    if not model_path:
        raise SileroVadError(
            "Could not find %s in the installed silero_vad package or the "
            "local torch hub cache. Install 'silero-vad' or pass "
            "silero_onnx_model_path." % model_name
        )
    return RawSileroOnnxVad(
        model_path,
        backend=backend_name,
        intra_op_num_threads=onnx_threads,
    )


def _create_official_onnx_vad(onnx_model_path=None):
    if onnx_model_path:
        try:
            utils_vad = import_module("silero_vad.utils_vad")
            wrapper = getattr(utils_vad, "OnnxWrapper")
        except Exception as exc:
            raise SileroVadError(
                "silero_backend='official_onnx' with silero_onnx_model_path "
                "requires the silero-vad package."
            ) from exc
        try:
            model = wrapper(onnx_model_path, force_onnx_cpu=True)
        except TypeError:
            model = wrapper(onnx_model_path)
        return SileroCallableVad(model, SILERO_BACKEND_OFFICIAL_ONNX)

    try:
        loaded = _load_silero_package_model(onnx=True)
    except Exception:
        loaded = _load_torch_hub_model(onnx=True)
    return SileroCallableVad(loaded, SILERO_BACKEND_OFFICIAL_ONNX)


def _create_legacy_vad(onnx=False):
    return SileroCallableVad(
        _load_torch_hub_model(onnx=onnx),
        SILERO_BACKEND_LEGACY,
    )


def _create_pytorch_vad(device):
    if device == "cuda":
        torch = import_module("torch")
        if not torch.cuda.is_available():
            raise SileroVadError(
                "silero_backend='pytorch_cuda' requested but CUDA is unavailable"
            )
    try:
        loaded = _load_silero_package_model(onnx=False)
    except Exception:
        loaded = _load_torch_hub_model(onnx=False)
    backend = (
        SILERO_BACKEND_PYTORCH_CPU
        if device == "cpu"
        else SILERO_BACKEND_PYTORCH_CUDA
    )
    return SileroCallableVad(loaded, backend, device=device)


def _load_silero_package_model(onnx=False):
    module = import_module("silero_vad")
    loader = getattr(module, "load_silero_vad", None)
    if loader is None:
        raise SileroVadError(
            "The installed silero_vad package does not expose load_silero_vad"
        )
    loaded = loader(onnx=onnx)
    return _first_model(loaded)


def _load_torch_hub_model(onnx=False):
    torch = import_module("torch")
    loaded = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        verbose=False,
        onnx=onnx,
    )
    return _first_model(loaded)


def _first_model(loaded):
    if isinstance(loaded, tuple):
        return loaded[0]
    return loaded


def find_silero_model_file(filename):
    """Find a Silero model asset in silero_vad or an existing torch hub cache."""
    for root in _silero_package_roots():
        found = _find_file_under(root, filename)
        if found:
            return found

    try:
        torch = import_module("torch")
        hub_root = torch.hub.get_dir()
    except Exception:
        hub_root = None
    if hub_root:
        found = _find_file_under(hub_root, filename)
        if found:
            return found

    return None


def _silero_package_roots():
    try:
        spec = find_spec("silero_vad")
    except Exception:
        return []
    if spec is None:
        return []
    roots = []
    if spec.submodule_search_locations:
        roots.extend(spec.submodule_search_locations)
    elif spec.origin:
        roots.append(os.path.dirname(spec.origin))
    return [root for root in roots if root and os.path.isdir(root)]


def _find_file_under(root, filename):
    if not root or not os.path.isdir(root):
        return None
    for dirpath, _dirnames, filenames in os.walk(root):
        if filename in filenames:
            return os.path.join(dirpath, filename)
    return None


class SileroCallableVad:
    """Adapter for Silero's PyTorch or official ONNX callable wrappers."""

    def __init__(self, model, backend, device=None):
        self._torch = import_module("torch")
        self.model = model
        self.backend = backend
        self.device = device
        if device and hasattr(self.model, "to"):
            self.model.to(device)
        if hasattr(self.model, "eval"):
            self.model.eval()

    def reset_states(self):
        reset = getattr(self.model, "reset_states", None)
        if reset:
            reset()

    def __call__(self, audio, sample_rate):
        if hasattr(audio, "detach"):
            tensor = audio.float()
        else:
            tensor = self._torch.from_numpy(_as_float32_audio(audio))
        if self.device:
            tensor = tensor.to(self.device)

        with self._torch.no_grad():
            result = self.model(tensor, sample_rate)
        return _result_to_float(result)


class RawSileroOnnxVad:
    """Fast direct ONNX Runtime Silero VAD for 16 kHz, 512-sample chunks."""

    def __init__(self, model_path, backend, intra_op_num_threads=2):
        ort = import_module("onnxruntime")
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = int(intra_op_num_threads or 2)
        session_options.inter_op_num_threads = 1
        session_options.log_severity_level = 3
        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        self.session = ort.InferenceSession(
            model_path,
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        self.backend = backend
        self.model_path = model_path
        self.sample_rate = 16000
        self.chunk_samples = 512
        self.context_samples = 64
        self.input_buffer = np.zeros(
            (1, self.context_samples + self.chunk_samples),
            dtype=np.float32,
        )

        self.inputs = self.session.get_inputs()
        self.outputs = self.session.get_outputs()
        self.audio_input_name = _select_audio_input_name(self.inputs)
        self.sr_input_name = _select_sample_rate_input_name(self.inputs)
        self.state_input_names = [
            item.name
            for item in self.inputs
            if item.name not in (self.audio_input_name, self.sr_input_name)
        ]
        self.state = {
            item.name: np.zeros(_state_shape(item), dtype=np.float32)
            for item in self.inputs
            if item.name in self.state_input_names
        }

    def reset_states(self):
        self.input_buffer.fill(0.0)
        for value in self.state.values():
            value.fill(0.0)

    def __call__(self, audio, sample_rate):
        if int(sample_rate) != self.sample_rate:
            raise ValueError("Raw Silero ONNX backend expects 16000 Hz audio")
        audio = _as_float32_audio(audio)
        if audio.shape[0] != self.chunk_samples:
            raise ValueError(
                "Raw Silero ONNX backend expects %s samples, got %s"
                % (self.chunk_samples, audio.shape[0])
            )

        self.input_buffer[:, self.context_samples:] = audio.reshape(1, -1)
        feeds = {self.audio_input_name: self.input_buffer}
        if self.sr_input_name:
            feeds[self.sr_input_name] = np.array(self.sample_rate, dtype=np.int64)
        for name in self.state_input_names:
            feeds[name] = self.state[name]

        results = self.session.run(None, feeds)
        probability = _result_to_float(results[0])
        for name, value in zip(self.state_input_names, results[1:]):
            if name in self.state and self.state[name].shape == value.shape:
                self.state[name][...] = value
            else:
                self.state[name] = np.asarray(value, dtype=np.float32)

        self.input_buffer[:, :self.context_samples] = audio[
            -self.context_samples:
        ].reshape(1, -1)
        return probability


def _select_audio_input_name(inputs):
    names = [item.name for item in inputs]
    if "input" in names:
        return "input"
    if "x" in names:
        return "x"
    return names[0]


def _select_sample_rate_input_name(inputs):
    for item in inputs:
        if item.name in ("sr", "sample_rate"):
            return item.name
    return None


def _state_shape(input_info):
    shape = []
    fallback = (2, 1, 128)
    lower_name = input_info.name.lower()
    if lower_name in ("h", "c", "hn", "cn"):
        fallback = (2, 1, 64)
    for index, dim in enumerate(input_info.shape):
        if isinstance(dim, int) and dim > 0:
            shape.append(dim)
        elif index < len(fallback):
            shape.append(fallback[index])
        else:
            shape.append(1)
    if not shape:
        return fallback
    return tuple(shape)


def _as_float32_audio(audio):
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    array = np.asarray(audio, dtype=np.float32).reshape(-1)
    return array


def _result_to_float(result):
    if isinstance(result, (float, int)):
        return float(result)
    if hasattr(result, "detach"):
        result = result.detach().cpu()
    if hasattr(result, "item"):
        return float(result.item())
    return float(np.asarray(result).reshape(-1)[0])

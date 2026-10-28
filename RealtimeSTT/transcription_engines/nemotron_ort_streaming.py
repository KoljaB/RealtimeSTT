"""Cache-aware NeMo streaming transducer on ONNX Runtime with snapshot flush.

sherpa-onnx's ``OnlineStream`` cannot be copied, so a pause cannot be flushed
without poisoning the live stream.  This engine runs the same exported
encoder/decoder/joiner and reproduces sherpa-onnx's NeMo streaming transducer
(``OnlineRecognizerTransducerNeMoImpl`` + greedy NeMo search) with the stream
state held as plain arrays.  ``flush_snapshot`` therefore decodes the open
chunk on a copy of the state and leaves the live stream untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import time
from typing import List, Optional, Sequence

import numpy as np

from .base import (
    BaseTranscriptionEngine,
    StreamingTranscriptionSession,
    TranscriptionEngineError,
    TranscriptionInfo,
    TranscriptionResult,
)

SAMPLE_RATE = 16000
FRAME_SHIFT_SAMPLES = 160
MAX_SYMBOLS_PER_FRAME = 10
# Temporary feature extractors for a flush start this many frames before the
# first frame they must reproduce, so edge padding never reaches used frames.
_FLUSH_FEATURE_MARGIN_FRAMES = 4


def _load_onnxruntime():
    import onnxruntime

    return onnxruntime


def _load_fbank():
    import kaldi_native_fbank

    return kaldi_native_fbank


def _is_language_tag(symbol: str) -> bool:
    """Mirror sherpa-onnx's language-tag test: <xx>, <xxx> or <xx-XX>."""

    if len(symbol) < 4 or symbol[0] != "<" or symbol[-1] != ">":
        return False
    body = symbol[1:-1]
    head, _, tail = body.partition("-")
    if not (2 <= len(head) <= 3 and head.isalpha() and head.islower()):
        return False
    if not tail:
        return "-" not in body
    return len(tail) == 2 and tail.isalpha() and tail.isupper()


@dataclass
class NemotronStreamState:
    """All mutable decoding state of one stream; copying it forks the stream."""

    cache_last_channel: np.ndarray
    cache_last_time: np.ndarray
    cache_last_channel_len: np.ndarray
    decoder_states: List[np.ndarray]
    decoder_out: Optional[np.ndarray] = None
    tokens: List[int] = field(default_factory=list)
    frame_offset: int = 0
    processed_frames: int = 0

    def fork(self) -> "NemotronStreamState":
        # ORT returns fresh arrays for every step and nothing mutates them in
        # place, so sharing the arrays is a valid copy; only lists are copied.
        return NemotronStreamState(
            cache_last_channel=self.cache_last_channel,
            cache_last_time=self.cache_last_time,
            cache_last_channel_len=self.cache_last_channel_len,
            decoder_states=list(self.decoder_states),
            decoder_out=self.decoder_out,
            tokens=list(self.tokens),
            frame_offset=self.frame_offset,
            processed_frames=self.processed_frames,
        )


class NemotronOrtStream:
    """One live stream: feature extractor, retained audio tail and state."""

    def __init__(self, model: "NemotronOrtStreamingModel", prompt_id: int):
        self.model = model
        self.prompt_id = int(prompt_id)
        self.fbank = model._new_fbank()
        self.state = model._initial_state()
        self.samples_seen = 0
        self._tail = np.zeros(0, dtype=np.float32)
        self._tail_start = 0

    def accept_waveform(self, samples) -> int:
        """Feed 16 kHz float samples and decode every complete chunk."""

        audio = np.ascontiguousarray(samples, dtype=np.float32).reshape(-1)
        if audio.size:
            self.fbank.accept_waveform(SAMPLE_RATE, audio.tolist())
            self._tail = np.concatenate((self._tail, audio))
            self.samples_seen += int(audio.size)
        steps = 0
        while self.fbank.num_frames_ready > self.state.processed_frames + self.model.window_size:
            frames = self.model._frames(self.fbank, self.state.processed_frames)
            self.state = self.model._step(self.state, frames, self.prompt_id)
            steps += 1
        self._trim_tail()
        return steps

    def _trim_tail(self) -> None:
        keep_from = max(0, (self.state.processed_frames - _FLUSH_FEATURE_MARGIN_FRAMES) * FRAME_SHIFT_SAMPLES)
        if keep_from > self._tail_start:
            self._tail = self._tail[keep_from - self._tail_start:]
            self._tail_start = keep_from

    def text(self) -> str:
        return self.model.detokenize(self.state.tokens)

    def flush_snapshot(
        self,
        pad_seconds: Optional[float] = None,
        *,
        refine_pad_seconds: Optional[float] = None,
        on_first=None,
    ) -> dict:
        """Decode the open chunk on a fork padded with silence; the stream is unchanged.

        With ``refine_pad_seconds`` the same fork continues over a longer
        silence pad after the first result has been handed to ``on_first``.
        The first stage equals a plain flush with ``pad_seconds``; the refined
        text equals a plain flush with ``refine_pad_seconds``.  A word that the
        model only emits after more right context (``awe`` -> ``awesome``)
        usually completes in the refined text.
        """

        started = time.perf_counter()
        pad = self.model.chunk_seconds if pad_seconds is None else float(pad_seconds)
        refine = None if refine_pad_seconds is None else max(pad, float(refine_pad_seconds))
        pad_samples = int(round(pad * SAMPLE_RATE))
        total_pad = int(round((refine if refine is not None else pad) * SAMPLE_RATE))
        state = self.state.fork()
        fbank = self.model._new_fbank()
        audio = np.concatenate((self._tail, np.zeros(total_pad, dtype=np.float32)))
        fbank.accept_waveform(SAMPLE_RATE, audio.tolist())
        fbank.input_finished()
        base = self._tail_start // FRAME_SHIFT_SAMPLES
        # Frames a finished extractor holds for the shorter pad (rounded like
        # kaldi-native-fbank with snip_edges=False).
        first_ready = (self._tail.size + pad_samples + FRAME_SHIFT_SAMPLES // 2) // FRAME_SHIFT_SAMPLES
        steps = 0
        while base + min(first_ready, fbank.num_frames_ready) > state.processed_frames + self.model.window_size:
            frames = self.model._frames(fbank, state.processed_frames - base)
            state = self.model._step(state, frames, self.prompt_id)
            steps += 1
        first = {
            "text": self.model.detokenize(state.tokens),
            "live_text": self.text(),
            "steps": steps,
            "flush_ms": (time.perf_counter() - started) * 1000.0,
            "audio_end_sample": self.samples_seen,
            "pad_seconds": pad,
        }
        if refine is None:
            return first
        if on_first is not None:
            on_first(first)
        refine_steps = 0
        while base + fbank.num_frames_ready > state.processed_frames + self.model.window_size:
            frames = self.model._frames(fbank, state.processed_frames - base)
            state = self.model._step(state, frames, self.prompt_id)
            refine_steps += 1
        return {
            **first,
            "refined_text": self.model.detokenize(state.tokens),
            "refine_steps": refine_steps,
            "refined_ms": (time.perf_counter() - started) * 1000.0,
            "refine_pad_seconds": refine,
        }

    def finish(self, pad_seconds: Optional[float] = None) -> str:
        """Final flush that does consume the stream (end of the turn)."""

        result = self.flush_snapshot(pad_seconds)
        return result["text"]


class NemotronOrtStreamingModel:
    """Loads one exported NeMo cache-aware streaming transducer."""

    def __init__(
        self,
        model_dir: str,
        *,
        num_threads: int = 4,
        precision: str = "int8",
        allow_spinning: bool = False,
    ):
        ort = _load_onnxruntime()
        self.knf = _load_fbank()
        suffix = ".int8.onnx" if precision == "int8" else ".onnx"
        options = ort.SessionOptions()
        options.intra_op_num_threads = int(num_threads)
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        spin = "1" if allow_spinning else "0"
        options.add_session_config_entry("session.intra_op.allow_spinning", spin)
        options.add_session_config_entry("session.inter_op.allow_spinning", spin)
        providers = ["CPUExecutionProvider"]
        self.encoder = ort.InferenceSession(f"{model_dir}/encoder{suffix}", options, providers=providers)
        self.decoder = ort.InferenceSession(f"{model_dir}/decoder{suffix}", options, providers=providers)
        self.joiner = ort.InferenceSession(f"{model_dir}/joiner{suffix}", options, providers=providers)

        meta = self.encoder.get_modelmeta().custom_metadata_map
        self.window_size = int(meta["window_size"])
        self.chunk_shift = int(meta["chunk_shift"])
        self.chunk_seconds = int(meta.get("chunk_size_ms", self.chunk_shift * 10)) / 1000.0
        self.subsampling = int(meta["subsampling_factor"])
        self.vocab_size = int(meta["vocab_size"]) + 1
        self.blank_id = self.vocab_size - 1
        self.feature_dim = int(meta["feat_dim"])
        self._cache_channel_shape = (1, int(meta["cache_last_channel_dim1"]), int(meta["cache_last_channel_dim2"]), int(meta["cache_last_channel_dim3"]))
        self._cache_time_shape = (1, int(meta["cache_last_time_dim1"]), int(meta["cache_last_time_dim2"]), int(meta["cache_last_time_dim3"]))
        self._pred_shape = (int(meta["pred_rnn_layers"]), 1, int(meta["pred_hidden"]))
        if meta.get("normalize_type", "") not in ("", "NA"):
            raise ValueError("Only exports without feature normalization are supported")
        self.multilingual = "prompt_index" in [i.name for i in self.encoder.get_inputs()]
        self.prompt_ids = json.loads(meta.get("prompt_dictionary", "{}")) if self.multilingual else {}
        self.default_prompt_id = int(meta.get("auto_prompt_id", 0)) if self.multilingual else 0

        self.symbols: List[str] = [""] * self.vocab_size
        with open(f"{model_dir}/tokens.txt", encoding="utf-8") as handle:
            for line in handle:
                parts = line.rstrip("\n").rsplit(" ", 1)
                if len(parts) == 2 and parts[1].lstrip("-").isdigit():
                    index = int(parts[1])
                    if 0 <= index < self.vocab_size:
                        self.symbols[index] = parts[0]
        if self.symbols[self.blank_id] != "<blk>":
            raise ValueError("tokens.txt must end with <blk>")
        self._language_tags = {i for i, s in enumerate(self.symbols) if self.multilingual and _is_language_tag(s)}

        self._enc_names = [i.name for i in self.encoder.get_inputs()]
        self._dec_names = [i.name for i in self.decoder.get_inputs()]
        self._join_names = [i.name for i in self.joiner.get_inputs()]

    # -- construction -------------------------------------------------
    def _new_fbank(self):
        knf = self.knf
        opts = knf.FbankOptions()
        opts.frame_opts.dither = 0.0
        opts.frame_opts.snip_edges = False
        opts.frame_opts.samp_freq = float(SAMPLE_RATE)
        opts.frame_opts.frame_shift_ms = 10.0
        opts.frame_opts.frame_length_ms = 25.0
        opts.frame_opts.remove_dc_offset = False
        opts.frame_opts.window_type = "hann"
        opts.mel_opts.num_bins = self.feature_dim
        opts.mel_opts.low_freq = 0.0
        opts.mel_opts.high_freq = 8000.0
        opts.mel_opts.is_librosa = True
        return knf.OnlineFbank(opts)

    def _initial_state(self) -> NemotronStreamState:
        return NemotronStreamState(
            cache_last_channel=np.zeros(self._cache_channel_shape, dtype=np.float32),
            cache_last_time=np.zeros(self._cache_time_shape, dtype=np.float32),
            cache_last_channel_len=np.zeros((1,), dtype=np.int64),
            decoder_states=[np.zeros(self._pred_shape, dtype=np.float32), np.zeros(self._pred_shape, dtype=np.float32)],
        )

    def prompt_id(self, language: Optional[str]) -> int:
        if not self.multilingual or not language:
            return self.default_prompt_id
        language = str(language)
        if language in self.prompt_ids:
            return int(self.prompt_ids[language])
        return int(self.prompt_ids.get(language.split("-")[0].lower(), self.default_prompt_id))

    def create_stream(self, language: Optional[str] = None) -> NemotronOrtStream:
        return NemotronOrtStream(self, self.prompt_id(language))

    # -- decoding -----------------------------------------------------
    def _frames(self, fbank, start: int) -> np.ndarray:
        rows = [fbank.get_frame(start + i) for i in range(self.window_size)]
        return np.asarray(rows, dtype=np.float32)

    def _run_decoder(self, token: int, states: Sequence[np.ndarray]):
        out = self.decoder.run(None, {
            self._dec_names[0]: np.array([[token]], dtype=np.int32),
            self._dec_names[1]: np.array([1], dtype=np.int32),
            self._dec_names[2]: states[0],
            self._dec_names[3]: states[1],
        })
        return out[0], [out[2], out[3]]

    def _step(self, state: NemotronStreamState, frames: np.ndarray, prompt_id: int) -> NemotronStreamState:
        feeds = {
            self._enc_names[0]: np.ascontiguousarray(frames.T[None, :, :]),
            self._enc_names[1]: np.array([self.window_size], dtype=np.int64),
            self._enc_names[2]: state.cache_last_channel,
            self._enc_names[3]: state.cache_last_time,
            self._enc_names[4]: state.cache_last_channel_len,
        }
        if self.multilingual:
            feeds[self._enc_names[5]] = np.array([prompt_id], dtype=np.int64)
        enc_out, _length, cache_ch, cache_t, cache_len = self.encoder.run(None, feeds)
        new = NemotronStreamState(
            cache_last_channel=cache_ch,
            cache_last_time=cache_t,
            cache_last_channel_len=cache_len,
            decoder_states=list(state.decoder_states),
            decoder_out=state.decoder_out,
            tokens=list(state.tokens),
            frame_offset=state.frame_offset,
            processed_frames=state.processed_frames + self.chunk_shift,
        )
        self._greedy(new, enc_out[0].T)
        return new

    def _greedy(self, state: NemotronStreamState, encoder_rows: np.ndarray) -> None:
        if state.decoder_out is None:
            last = state.tokens[-1] if state.tokens else self.blank_id
            state.decoder_out, state.decoder_states = self._run_decoder(last, state.decoder_states)
        dec_out, dec_states = state.decoder_out, state.decoder_states
        for row in encoder_rows:
            cur = np.ascontiguousarray(row.reshape(1, -1, 1), dtype=np.float32)
            for _ in range(MAX_SYMBOLS_PER_FRAME):
                logit = self.joiner.run(None, {self._join_names[0]: cur, self._join_names[1]: dec_out})[0]
                y = int(np.argmax(logit.reshape(-1)))
                if y == self.blank_id:
                    break
                state.tokens.append(y)
                dec_out, dec_states = self._run_decoder(y, dec_states)
        state.decoder_out, state.decoder_states = dec_out, dec_states
        state.frame_offset += int(encoder_rows.shape[0])

    def detokenize(self, tokens: Sequence[int]) -> str:
        pieces: List[str] = []
        pending = bytearray()

        def flush_bytes():
            if pending:
                pieces.append(pending.decode("utf-8", errors="ignore"))
                pending.clear()

        for token in tokens:
            if token in self._language_tags:
                continue
            symbol = self.symbols[token]
            if len(symbol) == 6 and symbol.startswith("<0x") and symbol.endswith(">"):
                pending.append(int(symbol[3:5], 16))
                continue
            flush_bytes()
            pieces.append(symbol)
        flush_bytes()
        return " ".join("".join(pieces).replace("▁", " ").split())


def _fixed_language(language) -> Optional[str]:
    value = str(language or "").strip()
    return None if not value or value.lower() == "auto" else value


class NemotronOrtEngine(BaseTranscriptionEngine):
    """Streaming engine for cache-aware NeMo exports with snapshot flush.

    ``engine_options``: ``num_threads`` (default 4), ``precision`` (int8/fp32),
    ``model_dir`` (default: the configured model path), ``flush_pad_seconds``
    (default: one chunk).
    """

    engine_name = "nemotron_ort"
    supports_streaming = True

    def __init__(self, config):
        super().__init__(config)
        options = config.engine_options or {}
        model_dir = options.get("model_dir") or config.model
        if not model_dir:
            raise TranscriptionEngineError("nemotron_ort needs a model directory")
        self.model = NemotronOrtStreamingModel(
            str(model_dir),
            num_threads=int(options.get("num_threads", 4)),
            precision=str(options.get("precision", "int8")),
            allow_spinning=bool(options.get("allow_spinning", False)),
        )
        pad = options.get("flush_pad_seconds")
        self.flush_pad_seconds = None if pad is None else float(pad)

    def create_streaming_session(self, language=None, use_prompt=True):
        options = self.config.engine_options or {}
        requested = options.get("language") if "language" in options else language
        return NemotronOrtStreamingSession(self, language=requested)

    def transcribe(self, audio, language=None, use_prompt=True, **kwargs):
        session = self.create_streaming_session(language=language, use_prompt=use_prompt)
        try:
            session.accept_audio(audio, sample_rate=kwargs.pop("sample_rate", None))
            return session.finish()
        finally:
            session.close()


class NemotronOrtStreamingSession(StreamingTranscriptionSession):
    """One live stream; ``flush_snapshot`` never changes it."""

    def __init__(self, engine: NemotronOrtEngine, language=None):
        self.engine = engine
        self.language = language
        self.closed = False
        self.finished = False
        self.stream = None
        self._final_text = None
        self.reset()

    def _result(self, text: str) -> TranscriptionResult:
        return TranscriptionResult(
            text=text,
            info=TranscriptionInfo(language=_fixed_language(self.language), language_probability=0.0),
        )

    def reset(self):
        if self.closed:
            raise TranscriptionEngineError("Cannot reset a closed nemotron_ort session.")
        self.stream = self.engine.model.create_stream(_fixed_language(self.language) or "auto")
        self.finished = False
        self._final_text = None

    def accept_audio(self, audio, sample_rate=None):
        if self.closed or self.finished:
            raise TranscriptionEngineError("Cannot feed a closed or finished nemotron_ort session.")
        if sample_rate is not None and int(sample_rate) != SAMPLE_RATE:
            raise TranscriptionEngineError("nemotron_ort accepts 16 kHz mono audio only.")
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size:
            self.stream.accept_waveform(audio)

    def decode(self):
        # accept_audio already decodes every complete chunk.
        return None

    def get_result(self):
        if self._final_text is not None:
            return self._result(self._final_text)
        return self._result(self.stream.text() if self.stream is not None else "")

    def flush_snapshot(self, pad_seconds=None, *, refine_chunks=None, on_first=None) -> dict:
        """Snapshot flush; ``refine_chunks`` (e.g. 2) adds a refined stage."""

        if self.closed or self.stream is None:
            raise TranscriptionEngineError("Cannot flush a closed nemotron_ort session.")
        if self._final_text is not None:
            return {"text": self._final_text, "live_text": self._final_text, "steps": 0,
                    "flush_ms": 0.0, "audio_end_sample": self.stream.samples_seen, "pad_seconds": 0.0}
        pad = self.engine.flush_pad_seconds if pad_seconds is None else pad_seconds
        refine_pad = None
        if refine_chunks:
            refine_pad = float(refine_chunks) * self.engine.model.chunk_seconds
        return self.stream.flush_snapshot(pad, refine_pad_seconds=refine_pad, on_first=on_first)

    def finish(self):
        if self.closed:
            return self.get_result()
        if not self.finished:
            self._final_text = self.flush_snapshot()["text"]
            self.finished = True
        return self.get_result()

    def input_finished(self):
        return self.finish()

    def close(self):
        self.closed = True
        self.stream = None


__all__ = [
    "NemotronOrtEngine",
    "NemotronOrtStream",
    "NemotronOrtStreamingModel",
    "NemotronOrtStreamingSession",
    "NemotronStreamState",
]

# Configuration

`AudioToTextRecorder` is the main library entry point. This page is a parameter
reference for its constructor. Examples and recommended starting patterns live
in [quick-start.md](quick-start.md).

This page covers every public constructor parameter, grouped by purpose so the
reference stays searchable without making the README too large.

```python
from RealtimeSTT import AudioToTextRecorder

recorder = AudioToTextRecorder(
    model="small.en",
    language="en",
    enable_realtime_transcription=True,
)
```

## Model And Engine Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `model` | `"tiny"` | Main transcription model name or model path. Interpretation depends on `transcription_engine`. |
| `transcription_engine` | `"faster_whisper"` | Main transcription backend. See [transcription-engines.md](transcription-engines.md). |
| `transcription_engine_options` | `None` | Engine-specific dictionary passed only to the main backend. |
| `download_root` | `None` | Directory for model downloads or lookup. Behavior is engine-specific. |
| `language` | `""` | Language code. Empty string lets engines auto-detect when they support it. Some engines require a language. |
| `compute_type` | `"default"` | Numeric precision/quantization hint. For faster-whisper, see CTranslate2 quantization. Other engines map this where possible. |
| `gpu_device_index` | `0` | GPU id, or a list of GPU ids for compatible engines. |
| `device` | `"cuda"` | Device hint, usually `"cuda"` or `"cpu"`. CPU-only engines ignore GPU settings. |
| `batch_size` | `16` | Main transcription batch size. Set `0` to disable batched faster-whisper inference. |
| `beam_size` | `5` | Main transcription beam size where supported. |
| `initial_prompt` | `None` | String or token iterable passed to the main engine as prompt/context where supported. |
| `suppress_tokens` | `[-1]` | Token ids suppressed by Whisper-family engines where supported. |
| `faster_whisper_vad_filter` | `True` | Enables faster-whisper's own VAD filter during transcription in addition to recorder VAD. |
| `normalize_audio` | `False` | Normalizes audio peak before transcription in engine adapters that use the shared normalization helper. |

## Audio Input Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `input_device_index` | `None` | PyAudio input device index. `None` lets PyAudio choose the default device. |
| `use_microphone` | `True` | When `False`, audio must be supplied through `feed_audio()`. |
| `buffer_size` | `512` | Recorder audio buffer size. Changing this can affect VAD behavior. |
| `sample_rate` | `16000` | Recorder sample rate. WebRTC VAD is sensitive to sample rate changes. |
| `handle_buffer_overflow` | platform-dependent | Logs and drops overflowed microphone input. Defaults to `True` except on macOS. |
| `allowed_latency_limit` | `100` | Maximum unprocessed input chunks before old chunks may be discarded. |
| `on_recorded_chunk` | `None` | Callback receiving each recorded audio chunk. |

## Text Formatting And Lifecycle

| Parameter | Default | Description |
| --- | --- | --- |
| `ensure_sentence_starting_uppercase` | `True` | Capitalizes detected sentence starts. |
| `ensure_sentence_ends_with_period` | `True` | Adds a final period when final text does not end in punctuation. |
| `spinner` | `True` | Shows the console state spinner. |
| `level` | `logging.WARNING` | Logger level used by the recorder. |
| `debug_mode` | `False` | Prints additional debug information. |
| `print_transcription_time` | `False` | Logs main transcription processing time. |
| `no_log_file` | `False` | Skips the debug log file. |
| `use_extended_logging` | `False` | Enables more detailed recording worker logs. |
| `start_callback_in_new_thread` | `False` | Runs callbacks in new threads instead of the recorder thread. |

## Recording And VAD Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `silero_sensitivity` | `0.4` | Silero VAD sensitivity, from `0` to `1`. |
| `silero_use_onnx` | `False` | Uses Silero's ONNX path instead of the PyTorch path. |
| `silero_deactivity_detection` | `False` | Uses Silero for end-of-speech detection instead of the default WebRTC end detection path. |
| `deactivity_silence_confirmation_duration` | `0.16` | Required continuous VAD silence before end-of-speech silence is confirmed. |
| `webrtc_sensitivity` | `3` | WebRTC VAD aggressiveness from `0` to `3`; higher is more aggressive and less sensitive. |
| `warmup_vad` | `True` | Runs a small VAD warmup during initialization to avoid first-chunk lazy setup cost. |
| `post_speech_silence_duration` | `0.6` | Required silence after speech before a recording is considered complete. |
| `min_length_of_recording` | `0.5` | Minimum recording duration in seconds. |
| `min_gap_between_recordings` | `0` | Minimum gap in seconds between recordings. |
| `pre_recording_buffer_duration` | `1.0` | Amount of pre-roll audio to keep before detected speech. |
| `early_transcription_on_silence` | `0` | Starts an early final transcription after this many milliseconds of silence; the result is discarded if speech resumes. |

## Realtime Transcription Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `enable_realtime_transcription` | `False` | Enables interim transcription while recording is still active. |
| `use_main_model_for_realtime` | `False` | Reuses the main model for realtime updates instead of loading a separate realtime model. |
| `realtime_transcription_engine` | `None` | Realtime backend. `None` uses `transcription_engine`. |
| `realtime_transcription_engine_options` | `None` | Engine-specific options for realtime. `None` reuses `transcription_engine_options`. |
| `realtime_model_type` | `"tiny"` | Realtime model name or path. |
| `realtime_processing_pause` | `0.2` | Seconds between realtime transcription attempts. Lower values increase load. |
| `init_realtime_after_seconds` | `0.2` | Initial delay after recording starts before the first realtime update. |
| `realtime_batch_size` | `16` | Realtime transcription batch size. |
| `beam_size_realtime` | `3` | Realtime beam size where supported. |
| `initial_prompt_realtime` | `None` | Prompt/context for the realtime model where supported. |
| `realtime_transcription_use_syllable_boundaries` | `False` | Schedules realtime updates from a lightweight acoustic boundary detector instead of only a fixed timer. |
| `realtime_boundary_detector_sensitivity` | `0.6` | Boundary detector sensitivity, from conservative `0` to eager `1`. |
| `realtime_boundary_followup_delays` | `(0.05, 0.2)` | Extra realtime update delays after a detected boundary. `None` or empty disables follow-ups. |

## Wake Word Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `wakeword_backend` | `""` | Wake word backend. Use `"pvporcupine"`/`"pvp"` or `"oww"`/`"openwakeword"`. |
| `wake_words` | `""` | Comma-separated Porcupine keywords. Also enables wake word mode. |
| `wake_words_sensitivity` | `0.6` | Wake word sensitivity from `0` to `1`. |
| `wake_word_activation_delay` | `0.0` | Delay before switching from normal voice activation to wake word activation. |
| `wake_word_timeout` | `5.0` | Seconds after wake word detection to wait for speech before returning to wake word mode. |
| `wake_word_buffer_duration` | `0.1` | Audio removed/buffered around wake word detection so the wake word is not included in the transcription. |
| `openwakeword_model_paths` | `None` | Comma-separated OpenWakeWord `.onnx` or `.tflite` model paths. |
| `openwakeword_inference_framework` | `"onnx"` | OpenWakeWord inference framework: `"onnx"` or `"tflite"`. |

## Callback Parameters

All callbacks are optional. By default they run in the recorder flow; set
`start_callback_in_new_thread=True` if callbacks may block.

| Parameter | Called when |
| --- | --- |
| `on_recording_start` | A recording starts. |
| `on_recording_stop` | A recording stops. |
| `on_transcription_start` | Final transcription starts. |
| `on_realtime_transcription_update` | New interim realtime text is available. Receives text. |
| `on_realtime_transcription_stabilized` | Higher-quality stabilized realtime text is available. Receives text. |
| `on_realtime_text_stabilization_update` | Structured realtime stabilization event is available. |
| `on_vad_start` | Voice activity is detected. |
| `on_vad_stop` | Voice activity ends. |
| `on_vad_detect_start` | Recorder starts listening for voice activity. |
| `on_vad_detect_stop` | Recorder stops listening for voice activity. |
| `on_turn_detection_start` | Turn detection starts. |
| `on_turn_detection_stop` | Turn detection stops. |
| `on_wakeword_detected` | Wake word is detected. |
| `on_wakeword_timeout` | Wake word was detected but no speech arrived before timeout. |
| `on_wakeword_detection_start` | Wake word listening starts. |
| `on_wakeword_detection_end` | Wake word listening stops. |
| `on_recorded_chunk` | A raw recorded chunk is available. Receives bytes. |

## Executor Injection

| Parameter | Default | Description |
| --- | --- | --- |
| `transcription_executor` | `None` | Optional callable used instead of the default main transcription execution path. Primarily used by tests and server integration. |
| `realtime_transcription_executor` | `None` | Optional callable used instead of the default realtime transcription execution path. Primarily used by tests and shared-model server integration. |

## External Audio API

When `use_microphone=False`, call:

```python
recorder.feed_audio(chunk, original_sample_rate=16000)
```

`chunk` should be 16-bit mono PCM bytes. If `original_sample_rate` is not
16000, the recorder resamples before placing audio into its queue. More detail
lives in [external-audio.md](external-audio.md).

## Shutdown

Prefer the context manager:

```python
with AudioToTextRecorder() as recorder:
    print(recorder.text())
```

If you do not use `with`, call:

```python
recorder.shutdown()
```

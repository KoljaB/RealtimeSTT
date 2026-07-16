# Test Scripts

Fast automated tests live under `tests/unit/` and are documented in
[testing.md](testing.md). This page explains the Python scripts directly under
`tests/`. Most of them are demos, manual checks, regression harnesses, or
legacy experiments rather than maintained unit tests.

Run scripts from the repository root so relative imports and model paths work.

## Maintained Regression And Benchmark Harnesses

| Script | Category | Purpose |
| --- | --- | --- |
| `tests/final_transcription_gap_regression.py` | Regression harness | Streams a WAV file while `AudioToTextRecorder.text()` runs in parallel to reproduce slow final-transcription gaps. Can generate expected JSON and compare CPU output. |
| `tests/realtime_transcription_count_comparison.py` | Manual integration benchmark | Compares timer-based realtime transcription with syllable-boundary scheduling on deterministic WAV input. Reports realtime model-call counts and validates final text. |
| `tests/realtime_boundary_detector_live_test.py` | Manual detector check | Lightweight live check for the realtime boundary detector. |
| `tests/realtime_boundary_detector_microphone.py` | Manual detector visualizer | Microphone visualizer for syllable/speech boundary detection. Useful when tuning boundary sensitivity. |

These are useful during recorder and realtime scheduling work, but they are not
the default fast unit test suite.

## Core RealtimeSTT Demos

| Script | Category | Purpose |
| --- | --- | --- |
| `tests/simple_test.py` | Minimal demo | Smallest microphone transcription smoke script. |
| `tests/realtimestt_test.py` | Full interactive demo | Rich console demo with realtime transcription, final text, and optional keyboard typing. |
| `tests/realtimestt_test_whispercpp.py` | Engine demo | whisper.cpp version of the interactive demo with CPU profiles. |
| `tests/realtimestt_omnilingual_test.py` | Engine smoke/demo | Linux/WSL2 Omnilingual ASR script with deterministic file smoke, init-only check, and interactive microphone mode. |
| `tests/feed_audio.py` | External-audio demo | Opens a PyAudio stream manually and feeds chunks through `feed_audio()` with `use_microphone=False`. |
| `tests/openwakeword_test.py` | Wake word demo | OpenWakeWord demo using local sample wake word models in the `tests/` folder. |
| `tests/realtime_loop_test.py` | Realtime demo | Exercises realtime transcription in a loop. |
| `tests/realtimestt_chinese.py` | Language demo | Demonstrates Chinese transcription settings. |
| `tests/vad_test.py` | VAD demo | Manual VAD behavior check. |

Use these when developing locally with a microphone and real models. They may
load models, use optional packages, write to the keyboard, or require audio
device permissions.

## Application Experiments

| Script | Category | Purpose |
| --- | --- | --- |
| `tests/advanced_talk.py` | Assistant demo | Combines RealtimeSTT with RealtimeTTS and LLM calls. Requires API keys and TTS dependencies. |
| `tests/minimalistic_talkbot.py` | Assistant demo | Small talkbot example using speech input and generated responses. |
| `tests/openai_voice_interface.py` | Assistant demo | Voice interface experiment using OpenAI-compatible client setup. |
| `tests/translator.py` | Translation demo | Speech translation workflow experiment. |
| `tests/type_into_textbox.py` | Utility demo | Types recognized text into the focused text box. |
| `tests/recorder_client.py` | Client demo | Uses the packaged recorder client/server path. |

Treat these as examples, not correctness tests. Check credentials, local model
servers, and package imports before running them.

## Speech Endpoint And Hotkey Experiments

| Script | Category | Purpose |
| --- | --- | --- |
| `tests/realtimestt_speechendpoint.py` | Legacy/manual experiment | Experiments with speech endpoint detection and display behavior. |
| `tests/realtimestt_speechendpoint_binary_classified.py` | Legacy/manual experiment | Variant of the speech endpoint experiment with binary classification logic. |
| `tests/realtimestt_test_hotkeys_v2.py` | Manual UI/hotkey demo | Rich console and hotkey workflow for dictation-style usage. |
| `tests/realtimestt_test_stereomix.py` | Manual audio-device demo | Tests stereo mix/system audio input behavior. |

These scripts may be useful as reference material, but behavior is less stable
than the unit tests and maintained regression harnesses.

## Helpers And Data

| File | Category | Purpose |
| --- | --- | --- |
| `tests/install_packages.py` | Helper | Prompts or installs demo dependencies used by older scripts. Avoid it in automated test runs. |
| `tests/samanta.tflite` | Data/model | OpenWakeWord model artifact used by demos. |
| `tests/suh_mahn_thuh.onnx` | Data/model | OpenWakeWord model artifact used by `openwakeword_test.py`. |
| `tests/suh_man_tuh.onnx` | Data/model | OpenWakeWord model artifact used by `openwakeword_test.py`. |
| `tests/README.md` | Legacy note | Short OpenWakeWord test note. |
| `tests/__init__.py` | Package marker | Lets unit tests import from `tests`. |

## Running The Maintained Scripts

Final transcription gap regression:

```bash
python tests/final_transcription_gap_regression.py --mode both
```

Realtime scheduling comparison:

```bash
python tests/realtime_transcription_count_comparison.py --mode both
```

whisper.cpp interactive demo:

```bash
python -m pip install "RealtimeSTT[whisper-cpp]" rich pyautogui colorama
python tests/realtimestt_test_whispercpp.py --profile balanced
```

Boundary detector microphone visualizer:

```bash
python tests/realtime_boundary_detector_microphone.py --sensitivity 0.6
```

## Safety Notes

- Scripts with `pyautogui`, `keyboard`, or hotkey support can type into the
  active application.
- Scripts using real engines may download large models or require CUDA.
- Microphone scripts require OS audio permissions.
- Assistant demos can call local or hosted LLM/TTS services depending on their
  configuration.

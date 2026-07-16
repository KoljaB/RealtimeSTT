# Wake Words

RealtimeSTT can wait for a wake word before recording the following speech.
Two backends are supported:

- Porcupine through `pvporcupine`
- OpenWakeWord through `openwakeword`

Wake word mode is enabled when `wake_words` is set or when
`wakeword_backend` selects OpenWakeWord.

## Porcupine

Install the Porcupine extra before selecting this backend:

```bash
python -m pip install "RealtimeSTT[porcupine]"
```

```python
from RealtimeSTT import AudioToTextRecorder

if __name__ == "__main__":
    recorder = AudioToTextRecorder(
        wakeword_backend="pvporcupine",
        wake_words="jarvis",
    )

    print('Say "Jarvis" and then speak.')
    print(recorder.text())
    recorder.shutdown()
```

Aliases:

- `pvporcupine`
- `pvp`

Built-in keyword names include:

- `alexa`
- `americano`
- `blueberry`
- `bumblebee`
- `computer`
- `grapefruits`
- `grasshopper`
- `hey google`
- `hey siri`
- `jarvis`
- `ok google`
- `picovoice`
- `porcupine`
- `terminator`

Multiple Porcupine keywords can be comma-separated:

```python
recorder = AudioToTextRecorder(wake_words="jarvis,computer")
```

## OpenWakeWord

Install the OpenWakeWord extra before selecting this backend:

```bash
python -m pip install "RealtimeSTT[openwakeword]"
```

Select it with `wakeword_backend="oww"` or
`wakeword_backend="openwakeword"`.

```python
from RealtimeSTT import AudioToTextRecorder

if __name__ == "__main__":
    recorder = AudioToTextRecorder(
        wakeword_backend="oww",
        openwakeword_model_paths="models/hey_assistant.onnx",
        wake_words_sensitivity=0.35,
        wake_word_buffer_duration=1.0,
    )

    print("Say the trained wake word and then speak.")
    print(recorder.text())
    recorder.shutdown()
```

OpenWakeWord model names are inferred from model files, so `wake_words` is not
required for custom model paths.

## Model Files

Porcupine built-in keywords are provided by the Porcupine package. For custom
Porcupine keywords, follow Picovoice's model/key workflow and pass paths through
the Porcupine package options when the wake word abstraction grows to expose
them.

OpenWakeWord accepts comma-separated paths:

```python
openwakeword_model_paths="word1.onnx,word2.onnx"
```

Supported inference frameworks:

- `onnx`
- `tflite`

Set the framework with:

```python
openwakeword_inference_framework="onnx"
```

If you have a TensorFlow Lite model and need ONNX:

```bash
python -m pip install -U tf2onnx
python -m tf2onnx.convert --tflite my_model.tflite --output my_model.onnx
```

OpenWakeWord project resources include training notebooks and conversion
guidance. Train and convert models outside RealtimeSTT, then pass the resulting
model files into `openwakeword_model_paths`.

## Sensitivity And Timing

| Parameter | Default | Meaning |
| --- | --- | --- |
| `wake_words_sensitivity` | `0.6` | Detection threshold from `0` to `1`. Lower can reduce false negatives but may increase false positives. |
| `wake_word_activation_delay` | `0.0` | Delay before switching into wake word activation when no initial speech is detected. |
| `wake_word_timeout` | `5.0` | Seconds to wait for speech after the wake word before returning to wake word mode. |
| `wake_word_buffer_duration` | `0.1` | Audio buffered/removed around wake word detection so the wake word itself is less likely to appear in final text. |

For OpenWakeWord custom models, a starting sensitivity around `0.35` can be a
useful first test, then tune against real room audio.

## Callbacks

```python
def detected():
    print("wake word detected")


def timeout():
    print("wake word timeout")


recorder = AudioToTextRecorder(
    wake_words="jarvis",
    on_wakeword_detected=detected,
    on_wakeword_timeout=timeout,
)
```

Available wake word callbacks:

- `on_wakeword_detection_start`
- `on_wakeword_detection_end`
- `on_wakeword_detected`
- `on_wakeword_timeout`

## Troubleshooting

- If wake words never trigger, confirm the selected backend, microphone device,
  sample rate, and model paths.
- If OpenWakeWord model files are missing, pass absolute paths first to remove
  ambiguity.
- If too many false detections occur, raise `wake_words_sensitivity`, reduce
  room noise, or retrain the model with better negative examples.
- If the wake word appears in final text, increase `wake_word_buffer_duration`.

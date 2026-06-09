# FunASR

`FunASR` is an industrial-grade speech recognition toolkit: 170x realtime, 50+ languages, speaker diarization, emotion detection, streaming, and OpenAI-compatible API.

## Install

Install the `FunASR` extra:

```bash
pip install "funasr"
```


## Basic Use

```python
from RealtimeSTT import AudioToTextRecorder

recorder = AudioToTextRecorder(
    transcription_engine="funasr",
    model="iic/SenseVoiceSmall",
    device="cuda",
)
```

For CPU:

```python
recorder = AudioToTextRecorder(
    transcription_engine="funasr",
    model="iic/SenseVoiceSmall",
    device="cpu",
)
```

## Model Behavior

Known model names such as `SenseVoiceSmall`, `Fun-ASR-Nano`and `Paraformer-zh` are downloaded automatically through modelscope.


## Common Options

| RealtimeSTT parameter | FunASR mapping |
| --- | --- |
| `model` | `model` |
| `device` | `device` |
| `beam_size` | `beam_size` |
| `batch_size` | `batch_size` |
| `transcription_engine_options: {vad_filter: bool, vad_model: str}` | `vad_model`|


## Troubleshooting

- The feature is still under development, so please reach out through issues if you encounter an issue.
- For more details about FunASR please see their [official Github Repo](https://github.com/modelscope/FunASR)

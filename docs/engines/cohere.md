# Cohere Transcribe

`cohere_transcribe` and `cohere` use the Cohere Transcribe model through
Transformers. This is an experimental adapter and requires an explicit language.

## Install

```bash
python -m pip install transformers torch
```

Install any model-specific dependencies required by the current Transformers
release. The model may require accepting gated Hugging Face access before
weights can be downloaded.

## API And Credentials

This adapter loads `CohereLabs/cohere-transcribe-03-2026` locally through
Transformers; it is not the `openai_api` placeholder and it does not call a
hosted RealtimeSTT API path.

Authentication depends on Hugging Face model access. If the model is gated,
authenticate with a Hugging Face token in the environment used by Transformers,
for example through `huggingface-cli login` or the standard `HF_TOKEN`
environment variable.

## Basic Use

```python
from RealtimeSTT import AudioToTextRecorder

recorder = AudioToTextRecorder(
    transcription_engine="cohere_transcribe",
    model="CohereLabs/cohere-transcribe-03-2026",
    language="en",
    device="cuda",
)
```

You can also provide the language in engine options:

```python
recorder = AudioToTextRecorder(
    transcription_engine="cohere",
    model="CohereLabs/cohere-transcribe-03-2026",
    transcription_engine_options={"language": "en"},
)
```

If neither `language` nor `engine_options["language"]` is set, the engine raises
an error because this adapter does not auto-detect language.

## Supported Paths

The adapter accepts in-memory audio arrays from RealtimeSTT and passes them to
the processor at `sample_rate`, defaulting to 16000.

Set `download_root` to map model and processor cache files to a project-local
directory:

```python
recorder = AudioToTextRecorder(
    transcription_engine="cohere_transcribe",
    model="CohereLabs/cohere-transcribe-03-2026",
    language="en",
    download_root="models/hf",
)
```

## Options

| Option | Meaning |
| --- | --- |
| `engine_options["processor"]` | Passed to `AutoProcessor.from_pretrained`. |
| `engine_options["model"]` | Passed to `CohereAsrForConditionalGeneration.from_pretrained`. |
| `engine_options["processor_call"]` | Merged into the processor call. |
| `engine_options["generate"]` | Merged into `model.generate(...)`. |
| `engine_options["decode"]` | Merged into processor decode. |
| `engine_options["language"]` | Language when not passed through `language`. |
| `engine_options["punctuation"]` | Passed to the processor when set. |
| `engine_options["sample_rate"]` | Input sample rate, default 16000. |

## Latency And Cost Notes

Because this adapter runs a local model, runtime cost is local compute rather
than per-request API billing. The practical cost is model download size,
startup time, GPU/CPU memory, and inference latency. Treat it as a heavier
experimental backend until you validate it on your target hardware.

## Troubleshooting

- Missing language errors mean you need `language="en"` or another supported
  code.
- Gated model errors require accepting access terms and authenticating with
  Hugging Face.
- Missing class errors usually mean the installed Transformers version is too
  old for the Cohere ASR classes.

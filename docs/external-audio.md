# External Audio

Use external audio mode when audio does not come from the local microphone:

- files
- websocket clients
- browser streams
- telephony/media servers
- other processes
- test fixtures

Create the recorder with `use_microphone=False` and call `feed_audio()`.

## Required Format

`feed_audio()` accepts raw audio chunks and places normalized recorder audio
into the input queue.

Best input format:

- 16-bit signed PCM
- mono
- 16000 Hz
- little-endian bytes

```python
recorder.feed_audio(chunk, original_sample_rate=16000)
```

If `original_sample_rate` is not 16000, RealtimeSTT resamples the chunk before
processing.

## Minimal File Example

```python
from RealtimeSTT import AudioToTextRecorder

if __name__ == "__main__":
    recorder = AudioToTextRecorder(use_microphone=False)

    with open("audio_chunk.pcm", "rb") as audio_file:
        recorder.feed_audio(audio_file.read(), original_sample_rate=16000)

    print(recorder.text())
    recorder.shutdown()
```

## Streaming A File In Chunks

```python
from RealtimeSTT import AudioToTextRecorder

CHUNK_BYTES = 3200

if __name__ == "__main__":
    recorder = AudioToTextRecorder(use_microphone=False)

    with open("audio_stream.pcm", "rb") as audio_file:
        while True:
            chunk = audio_file.read(CHUNK_BYTES)
            if not chunk:
                break
            recorder.feed_audio(chunk, original_sample_rate=16000)

    print(recorder.text())
    recorder.shutdown()
```

In a realtime process, feed chunks as they arrive and call `text()` from the
application flow that should wait for final utterances.

## From A Websocket Or Browser

Convert incoming audio to 16-bit mono PCM before feeding:

```python
def handle_pcm_packet(recorder, pcm_bytes, sample_rate):
    recorder.feed_audio(pcm_bytes, original_sample_rate=sample_rate)
```

The FastAPI browser server includes a production-shaped example of this pattern.
It receives binary websocket packets, decodes metadata, resamples browser audio,
and feeds per-session recorder queues. See [fastapi-server.md](fastapi-server.md).

## From Another Process

Any process that can emit PCM bytes can be connected to RealtimeSTT. Keep the
boundary explicit:

```python
from RealtimeSTT import AudioToTextRecorder


def pcm_chunks_from_process():
    while True:
        chunk = read_next_chunk_somehow()
        if not chunk:
            break
        yield chunk


if __name__ == "__main__":
    recorder = AudioToTextRecorder(use_microphone=False)

    for chunk in pcm_chunks_from_process():
        recorder.feed_audio(chunk, original_sample_rate=16000)

    print(recorder.text())
    recorder.shutdown()
```

Replace `read_next_chunk_somehow()` with your pipe, socket, queue, or media
framework integration.

## Realtime Updates With External Audio

```python
from RealtimeSTT import AudioToTextRecorder


def live(text):
    print("live:", text)


if __name__ == "__main__":
    recorder = AudioToTextRecorder(
        use_microphone=False,
        enable_realtime_transcription=True,
        on_realtime_transcription_update=live,
    )

    for chunk in audio_chunks():
        recorder.feed_audio(chunk, original_sample_rate=48000)

    print("final:", recorder.text())
    recorder.shutdown()
```

## Practical Notes

- Feed reasonably small chunks so VAD and realtime updates can respond quickly.
- Preserve ordering; `feed_audio()` does not reorder chunks.
- Use one recorder per independent stream/session unless you are building a
  shared-engine server that injects its own executors.
- Call `shutdown()` when the stream is done if you are not using a context
  manager.
- For finite test files, make sure enough trailing silence is fed for VAD to
  finalize the utterance.

## Troubleshooting

- Garbled text often means the audio encoding, channel count, or sample rate is
  not what the recorder expects.
- If `text()` never returns, feed trailing silence or adjust
  `post_speech_silence_duration`.
- If audio arrives faster than it is processed, increase capacity, lower model
  cost, or watch `allowed_latency_limit`.

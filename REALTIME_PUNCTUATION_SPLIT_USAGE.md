# Realtime Punctuation Splitting

Realtime punctuation splitting is an opt-in feature for splitting long active
recordings when stable realtime text and word timestamps agree on a punctuation
boundary.

The default is disabled:

```python
AudioToTextRecorder(
    enable_realtime_transcription=True,
    realtime_punctuation_split_marks="off",
)
```

For the 1.0.3 release, the production-supported mode is sentence punctuation:

```python
AudioToTextRecorder(
    enable_realtime_transcription=True,
    realtime_punctuation_split_marks="sentence",
)
```

`"sentence"` enables `.`, `?`, and `!` split candidates. The feature remains
guarded by repeated realtime observations, word timestamp validation, and
short-segment checks so normal final transcription behavior is unchanged unless
the option is explicitly enabled.

The parser also accepts `"comma"`, `"dash"`, `"ellipsis"`, and `"all"` presets,
or an iterable of explicit marks. Those modes are available for experiments but
are not promoted as production-supported in this release.

Splitting needs word-level timestamps from the main transcription engine.
RealtimeSTT currently wires that path for `faster_whisper`; other built-in
engines skip punctuation splitting rather than failing the recorder.

Useful examples:

```python
AudioToTextRecorder(realtime_punctuation_split_marks="off")
AudioToTextRecorder(realtime_punctuation_split_marks="sentence")
AudioToTextRecorder(realtime_punctuation_split_marks=(".", "?", "!"))
AudioToTextRecorder(realtime_punctuation_split_marks="sentence,comma")
```

When a split happens, the left side is queued for final transcription and the
right side remains active as the next recording segment. If the split punctuation
is a comma, the next final result is lowercased at the start so joined text stays
readable.

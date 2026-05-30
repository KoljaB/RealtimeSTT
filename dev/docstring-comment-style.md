# Docstring And Comment Style

Use docstrings to explain what public objects do, not how the current
implementation happens to do it.

## Module Docstrings

- Keep module docstrings to one or two concrete sentences.
- Describe the module's user-facing responsibility.
- Avoid architectural filler such as "preserves the stable API" or "delegates
  implementation details" unless that is the module's actual purpose.

Example style from `RealtimeSTT/audio_recorder.py`:

```python
"""Expose the public RealtimeSTT audio recorder API.

The recorder captures speech, coordinates voice activity and wake-word handling,
and returns final or realtime transcription results.
"""
```

## Method Docstrings

- Start with a short, precise sentence in third-person present tense:
  `Stops recording audio.`
- Use block-style triple-quoted docstrings even for short summaries:
  write the summary on its own line between opening and closing quotes.
- Do not use one-line triple-quoted docstrings such as
  `"""Stops recording audio."""`.
- Add `Args:` only when the method has meaningful arguments.
- Keep argument descriptions short and practical.
- Do not add return sections for obvious fluent wrappers or simple delegations.
- Keep detailed constructor parameter documentation when the constructor is a
  large public compatibility surface.

Preferred shape:

```python
def stop(self, backdate_stop_seconds=0.0, backdate_resume_seconds=0.0):
    """
    Stops recording audio.

    Args:
    - backdate_stop_seconds: Seconds to backdate the stop time.
    - backdate_resume_seconds: Seconds to backdate resumed listening.
    """
```

## Comments

- Prefer clear names and small functions over comments.
- Keep comments when they explain compatibility, timing, platform behavior, or a
  non-obvious business/runtime constraint.
- Do not keep comments that merely narrate the next line.
- Do not remove existing behavior-sensitive comments during a refactor unless
  the behavior they describe is moved with them.

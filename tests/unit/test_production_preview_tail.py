import asyncio
import threading
import unittest
from types import SimpleNamespace

from RealtimeSTT_server import production_server as production
from RealtimeSTT_server.preview_tail import (
    PreviewTailAnchor,
    compose_tail_preview,
    plan_preview_tail,
)

SR = production.SERVER_SAMPLE_RATE


class PreviewTailCompositionTests(unittest.TestCase):
    def test_exact_anchor_takes_anchor_words_from_the_tail(self):
        result = compose_tail_preview(
            "And then we will see if.",
            "will see if the thing works",
        )

        self.assertTrue(result.matched)
        self.assertEqual(result.reason, "exact")
        # The pause-closed "if." is replaced by the tail's continuation.
        self.assertEqual(result.text, "And then we will see if the thing works")

    def test_four_word_anchor_keeps_the_untouched_prefix(self):
        result = compose_tail_preview(
            "Hey there this is a little test",
            "is a little test where I pause",
        )

        self.assertTrue(result.matched)
        self.assertEqual(result.anchor_length, 4)
        self.assertEqual(result.text, "Hey there this is a little test where I pause")

    def test_anchor_words_keep_their_case_when_the_tail_starts_on_them(self):
        result = compose_tail_preview(
            "It's more like that I make some",
            "That I make some pauses in between.",
        )

        self.assertTrue(result.matched)
        self.assertEqual(
            result.text,
            "It's more like that I make some pauses in between.",
        )

    def test_repeated_anchor_phrase_is_ambiguous(self):
        result = compose_tail_preview(
            "we go on and on",
            "go on and on and go on and on",
        )

        self.assertFalse(result.matched)
        self.assertEqual(result.reason, "ambiguous_anchor")

    def test_unrelated_tail_does_not_match(self):
        result = compose_tail_preview(
            "the quick brown fox jumps",
            "completely different words here",
        )

        self.assertFalse(result.matched)
        self.assertEqual(result.reason, "no_alignment")

    def test_short_anchor_and_empty_tail_do_not_match(self):
        self.assertEqual(compose_tail_preview("Able to", "to do this").reason, "short_anchor")
        self.assertEqual(compose_tail_preview("one two three", "").reason, "empty_tail")

    def test_live_anchor_drops_its_unstable_last_word(self):
        result = compose_tail_preview(
            "please turn on the lig",
            "turn on the light in the kitchen",
            drop_last_anchor_word=True,
        )

        self.assertTrue(result.matched)
        self.assertEqual(result.text, "please turn on the light in the kitchen")


class PreviewTailPlanTests(unittest.TestCase):
    def test_previous_preview_is_preferred_over_live(self):
        earlier = PreviewTailAnchor("one two", 4 * SR, "preview")
        preview = PreviewTailAnchor("one two three four five six", 5 * SR, "preview")
        live = PreviewTailAnchor("one two three four five six seven", 6 * SR, "live")

        plan, reason = plan_preview_tail(
            preview, live, 7 * SR, sample_rate=SR, history=[earlier]
        )

        self.assertEqual(reason, "tail")
        self.assertEqual(plan.anchor.source, "preview")
        self.assertEqual(plan.start_sample, int(3.5 * SR))

    def test_overlap_extends_back_until_it_holds_four_recognized_words(self):
        # Hesitant speech: the last 1.5 s before the anchor end held one word.
        history = [
            PreviewTailAnchor("so", 3 * SR, "preview"),
            PreviewTailAnchor("so I think that", 4 * SR, "preview"),
        ]
        preview = PreviewTailAnchor("so I think that we", 6 * SR, "preview")

        plan, _reason = plan_preview_tail(
            preview, None, 8 * SR, sample_rate=SR, history=history
        )

        self.assertEqual(plan.start_sample, int(2.7 * SR))

    def test_without_four_earlier_words_the_full_turn_is_decoded(self):
        preview = PreviewTailAnchor("one two three four five six", 5 * SR, "preview")

        plan, reason = plan_preview_tail(preview, None, 7 * SR, sample_rate=SR)

        self.assertIsNone(plan)
        self.assertEqual(reason, "turn_too_short")

    def test_live_anchor_gets_extra_overlap(self):
        live = PreviewTailAnchor("one two three four", 6 * SR, "live")

        plan, _reason = plan_preview_tail(None, live, 7 * SR, sample_rate=SR)

        self.assertEqual(plan.start_sample, 4 * SR)

    def test_short_turns_and_missing_anchors_use_the_full_turn(self):
        early = PreviewTailAnchor("one two three", 2 * SR, "preview")
        future = PreviewTailAnchor("one two three", 9 * SR, "preview")

        self.assertEqual(plan_preview_tail(None, None, 7 * SR, sample_rate=SR)[1], "no_anchor")
        self.assertEqual(plan_preview_tail(early, None, 3 * SR, sample_rate=SR)[1], "turn_too_short")
        # An anchor beyond the requested audio cannot describe its prefix.
        self.assertEqual(plan_preview_tail(future, None, 7 * SR, sample_rate=SR)[1], "no_anchor")


class _Manager:
    def __init__(self):
        self._loop = object()
        self.events = []
        self.published = threading.Event()

    def set_audio_sequence(self, session_id, sequence):
        return None

    def publish_session(self, session_id, event, authoritative=False):
        self.events.append(dict(event))
        self.published.set()
        delivered = production.concurrent.futures.Future()
        delivered.set_result(True)
        return delivered


class _Service:
    def __init__(self, texts):
        self.texts = list(texts)
        self.calls = []

    def transcribe_turn(self, audio, language, use_prompt):
        self.calls.append(audio.size)
        return SimpleNamespace(text=self.texts[min(len(self.calls), len(self.texts)) - 1])


class ProductionPreviewTailModeTests(unittest.TestCase):
    @staticmethod
    def _run(texts, *, anchor=None, history=(), seconds=7):
        manager = _Manager()
        service = _Service(texts)
        protocol = production.ProductionSessionProtocol(
            service,
            manager,
            "session",
            production.ProductionServerSettings(preview_input_mode="tail"),
        )
        protocol.turn = production.TurnState(
            turn_id="turn",
            language="en",
            generation=1,
            pcm_buffer=bytearray(b"\x01\x00" * (SR * seconds)),
            preview_anchor=anchor,
            preview_anchor_history=list(history),
        )

        async def request():
            await protocol.preview(
                {"type": "preview", "turnId": "turn", "previewRequestId": "r"}
            )
            if not await asyncio.to_thread(manager.published.wait, 1.0):
                raise AssertionError("Preview result was not published")

        asyncio.run(request())
        return protocol, service, manager.events[-1]

    def test_tail_decode_splices_onto_the_previous_preview(self):
        earlier = PreviewTailAnchor("Hey there", 3 * SR, "preview")
        anchor = PreviewTailAnchor("Hey there this is a little test", 5 * SR, "preview")

        protocol, service, event = self._run(
            ["is a little test where I pause"],
            anchor=anchor,
            history=[earlier],
        )

        self.assertEqual(service.calls, [int(4.3 * SR)])
        self.assertEqual(event["status"], "exact")
        self.assertEqual(event["text"], "Hey there this is a little test where I pause")
        # Text coverage, not decode coverage, stays the full turn.
        self.assertEqual(event["inputSampleRange"], {"start": 0, "end": 7 * SR})
        self.assertEqual(event["previewInputCoverage"], "full_turn")
        tail = event["previewTiming"]["previewTail"]
        self.assertEqual(tail["decodeScope"], "tail")
        self.assertEqual(tail["decodeStartSample"], int(2.7 * SR))
        self.assertEqual(protocol.turn.preview_anchor.end_sample, 7 * SR)
        self.assertEqual(protocol.turn.preview_anchor_history, [earlier, anchor])
        self.assertEqual(protocol.turn.preview_anchor.text, event["text"])

    def test_unaligned_tail_falls_back_to_one_full_turn_decode(self):
        earlier = PreviewTailAnchor("the", 3 * SR, "preview")
        anchor = PreviewTailAnchor("the quick brown fox jumps", 5 * SR, "preview")

        _protocol, service, event = self._run(
            ["completely different words", "the full turn text"],
            anchor=anchor,
            history=[earlier],
        )

        self.assertEqual(service.calls, [int(4.3 * SR), 7 * SR])
        self.assertEqual(event["status"], "full_buffer")
        self.assertEqual(event["text"], "the full turn text")
        self.assertEqual(event["previewTiming"]["previewTail"]["reason"], "fallback_no_alignment")
        self.assertEqual(
            [attempt["decodeScope"] for attempt in event["previewTiming"]["asrAttempts"]],
            ["tail", "full_turn"],
        )

    def test_first_preview_of_a_turn_decodes_the_full_turn(self):
        protocol, service, event = self._run(["first full text here"])

        self.assertEqual(service.calls, [7 * SR])
        self.assertEqual(event["status"], "full_buffer")
        self.assertEqual(event["previewTiming"]["previewTail"]["reason"], "no_anchor")
        self.assertEqual(protocol.turn.preview_anchor.text, "first full text here")


if __name__ == "__main__":
    unittest.main()

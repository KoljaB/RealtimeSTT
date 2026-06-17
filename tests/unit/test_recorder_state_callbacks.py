import sys
import types
import unittest
from unittest import mock

try:
    __import__("halo")
except ModuleNotFoundError:
    sys.modules["halo"] = types.SimpleNamespace(Halo=None)

from RealtimeSTT.core import state as state_helpers


UNSET = object()


class FakeHalo:
    def __init__(self, text=None):
        self.text = text
        self.started = False
        self.stopped = False
        self._interval = None

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True


class RecorderLike:
    def __init__(
        self,
        state="inactive",
        halo=None,
        spinner_listening_text=UNSET,
        spinner_wakeword_text=UNSET,
        spinner_transcribing_text=UNSET,
        spinner_recording_text=UNSET,
    ):
        self.state = state
        self.start_callback_in_new_thread = False
        self.spinner = True
        self.halo = halo
        self.wake_words = "jarvis"
        self.events = []
        self.on_vad_detect_start = None
        self.on_vad_detect_stop = None
        self.on_wakeword_detection_start = None
        self.on_wakeword_detection_end = None
        if spinner_listening_text is not UNSET:
            self.spinner_listening_text = spinner_listening_text
        if spinner_wakeword_text is not UNSET:
            self.spinner_wakeword_text = spinner_wakeword_text
        if spinner_transcribing_text is not UNSET:
            self.spinner_transcribing_text = spinner_transcribing_text
        if spinner_recording_text is not UNSET:
            self.spinner_recording_text = spinner_recording_text


class RecorderStateCallbackTests(unittest.TestCase):
    def patch_spinner_events(self):
        original_set_spinner = state_helpers.set_spinner

        def recording_set_spinner(recorder, text):
            recorder.events.append(f"spinner:{text}")
            return original_set_spinner(recorder, text)

        return mock.patch.object(
            state_helpers,
            "set_spinner",
            side_effect=recording_set_spinner,
        )

    def test_inactive_to_listening_runs_start_callback_before_spinner(self):
        recorder = RecorderLike()
        recorder.on_vad_detect_start = lambda: recorder.events.append("vad_start")

        with self.patch_spinner_events():
            with mock.patch.object(state_helpers.halo, "Halo", FakeHalo):
                with self.assertLogs("realtimestt", level="INFO") as logs:
                    state_helpers.set_recorder_state(recorder, "listening")

        self.assertEqual(recorder.state, "listening")
        self.assertEqual(recorder.events, ["vad_start", "spinner:speak now"])
        self.assertEqual(recorder.halo.text, "speak now")
        self.assertTrue(recorder.halo.started)
        self.assertEqual(recorder.halo._interval, 250)
        self.assertEqual(
            [record.getMessage() for record in logs.records],
            ["State changed from 'inactive' to 'listening'"],
        )

    def test_listening_to_recording_runs_stop_callback_before_spinner(self):
        recorder = RecorderLike(state="listening", halo=FakeHalo("speak now"))
        recorder.on_vad_detect_stop = lambda: recorder.events.append("vad_stop")

        with self.patch_spinner_events():
            state_helpers.set_recorder_state(recorder, "recording")

        self.assertEqual(recorder.state, "recording")
        self.assertEqual(recorder.events, ["vad_stop", "spinner:recording"])
        self.assertEqual(recorder.halo.text, "recording")
        self.assertEqual(recorder.halo._interval, 100)

    def test_recording_to_transcribing_sets_spinner_text_and_interval(self):
        recorder = RecorderLike(state="recording", halo=FakeHalo("recording"))

        with self.patch_spinner_events():
            state_helpers.set_recorder_state(recorder, "transcribing")

        self.assertEqual(recorder.state, "transcribing")
        self.assertEqual(recorder.events, ["spinner:transcribing"])
        self.assertEqual(recorder.halo.text, "transcribing")
        self.assertEqual(recorder.halo._interval, 50)

    def test_inactive_to_wakeword_uses_wake_words_text_and_interval(self):
        recorder = RecorderLike()
        recorder.on_wakeword_detection_start = (
            lambda: recorder.events.append("wakeword_start")
        )

        with self.patch_spinner_events():
            with mock.patch.object(state_helpers.halo, "Halo", FakeHalo):
                state_helpers.set_recorder_state(recorder, "wakeword")

        self.assertEqual(recorder.state, "wakeword")
        self.assertEqual(recorder.events, ["wakeword_start", "spinner:say jarvis"])
        self.assertEqual(recorder.halo.text, "say jarvis")
        self.assertEqual(recorder.halo._interval, 500)

    def test_wakeword_to_inactive_stops_spinner_after_end_callback(self):
        existing_halo = FakeHalo("say jarvis")
        recorder = RecorderLike(state="wakeword", halo=existing_halo)
        recorder.on_wakeword_detection_end = (
            lambda: recorder.events.append("wakeword_end")
        )

        state_helpers.set_recorder_state(recorder, "inactive")

        self.assertEqual(recorder.state, "inactive")
        self.assertEqual(recorder.events, ["wakeword_end"])
        self.assertTrue(existing_halo.stopped)
        self.assertIsNone(recorder.halo)

    def test_same_state_is_no_op(self):
        existing_halo = FakeHalo("speak now")
        recorder = RecorderLike(state="listening", halo=existing_halo)
        recorder.on_vad_detect_start = lambda: recorder.events.append("vad_start")
        recorder.on_vad_detect_stop = lambda: recorder.events.append("vad_stop")

        state_helpers.set_recorder_state(recorder, "listening")

        self.assertEqual(recorder.state, "listening")
        self.assertEqual(recorder.events, [])
        self.assertIs(recorder.halo, existing_halo)
        self.assertEqual(existing_halo.text, "speak now")

    def test_missing_spinner_text_attributes_use_legacy_texts(self):
        recorder = RecorderLike()

        with self.patch_spinner_events():
            with mock.patch.object(state_helpers.halo, "Halo", FakeHalo):
                state_helpers.set_recorder_state(recorder, "listening")
                state_helpers.set_recorder_state(recorder, "wakeword")
                state_helpers.set_recorder_state(recorder, "recording")
                state_helpers.set_recorder_state(recorder, "transcribing")

        self.assertEqual(
            recorder.events,
            [
                "spinner:speak now",
                "spinner:say jarvis",
                "spinner:recording",
                "spinner:transcribing",
            ],
        )
        self.assertEqual(recorder.halo.text, "transcribing")

    def test_custom_listening_spinner_text_is_used(self):
        recorder = RecorderLike(spinner_listening_text="ready when you are")

        with self.patch_spinner_events():
            with mock.patch.object(state_helpers.halo, "Halo", FakeHalo):
                state_helpers.set_recorder_state(recorder, "listening")

        self.assertEqual(recorder.events, ["spinner:ready when you are"])
        self.assertEqual(recorder.halo.text, "ready when you are")

    def test_empty_spinner_text_is_allowed(self):
        recorder = RecorderLike(spinner_listening_text="")

        with self.patch_spinner_events():
            with mock.patch.object(state_helpers.halo, "Halo", FakeHalo):
                state_helpers.set_recorder_state(recorder, "listening")

        self.assertEqual(recorder.events, ["spinner:"])
        self.assertEqual(recorder.halo.text, "")

    def test_wakeword_spinner_text_formats_wake_words(self):
        recorder = RecorderLike(
            spinner_wakeword_text="waiting for {wake_words}",
        )

        with self.patch_spinner_events():
            with mock.patch.object(state_helpers.halo, "Halo", FakeHalo):
                state_helpers.set_recorder_state(recorder, "wakeword")

        self.assertEqual(recorder.events, ["spinner:waiting for jarvis"])
        self.assertEqual(recorder.halo.text, "waiting for jarvis")

    def test_run_callback_inline_preserves_call_arguments(self):
        recorder = RecorderLike()
        calls = []

        state_helpers.run_callback(
            recorder,
            lambda *args, **kwargs: calls.append((args, kwargs)),
            "text",
            preview=True,
        )

        self.assertEqual(calls, [(("text",), {"preview": True})])

    def test_run_callback_threaded_starts_daemon_thread(self):
        recorder = RecorderLike()
        recorder.start_callback_in_new_thread = True
        calls = []
        created_threads = []

        class FakeThread:
            def __init__(self, target, args, kwargs, daemon):
                self.target = target
                self.args = args
                self.kwargs = kwargs
                self.daemon = daemon
                self.started = False
                created_threads.append(self)

            def start(self):
                self.started = True

        with mock.patch.object(state_helpers.threading, "Thread", FakeThread):
            state_helpers.run_callback(
                recorder,
                lambda *args, **kwargs: calls.append((args, kwargs)),
                "text",
                preview=True,
            )

        self.assertEqual(calls, [])
        self.assertEqual(len(created_threads), 1)
        thread = created_threads[0]
        self.assertEqual(thread.args, ("text",))
        self.assertEqual(thread.kwargs, {"preview": True})
        self.assertTrue(thread.daemon)
        self.assertTrue(thread.started)


if __name__ == "__main__":
    unittest.main()

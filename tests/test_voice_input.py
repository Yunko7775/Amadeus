import sys
import types
import unittest
from unittest.mock import Mock, patch
from threading import Event

import numpy as np

from desktop.voice_input import MIN_RECORDING_SAMPLES, PushToTalk


class FakeInputStream:
    def __init__(self, **kwargs):
        self.callback = kwargs["callback"]
        self.active = False

    def start(self):
        self.active = True

    def stop(self):
        self.active = False

    def close(self):
        self.active = False

    def feed(self, data):
        try:
            self.callback(data, len(data), None, None)
        except Exception as error:
            if type(error).__name__ != "CallbackStop":
                raise
            self.active = False


class PushToTalkTests(unittest.TestCase):
    def setUp(self):
        self.recording_changed = Mock()
        self.audio_ready = Mock()
        self.error = Mock()
        self.streams = []

        def make_stream(**kwargs):
            stream = FakeInputStream(**kwargs)
            self.streams.append(stream)
            return stream

        self.sounddevice = types.SimpleNamespace(
            InputStream=make_stream,
            CallbackStop=type("CallbackStop", (Exception,), {}),
        )
        self.controller = PushToTalk(
            self.recording_changed,
            self.audio_ready,
            self.error,
            hotkey="f8",
        )

    def test_records_mono_audio_until_recording_is_toggled_off(self):
        with patch.dict(sys.modules, {"sounddevice": self.sounddevice}):
            self.controller.start_recording()
            self.streams[0].feed(
                np.ones((MIN_RECORDING_SAMPLES + 100, 1), dtype=np.float32)
            )
            self.controller.stop_recording()

        np.testing.assert_array_equal(
            self.audio_ready.call_args.args[0],
            np.ones(MIN_RECORDING_SAMPLES + 100, dtype=np.float32),
        )
        self.assertEqual(
            self.recording_changed.call_args_list,
            [unittest.mock.call(True), unittest.mock.call(False)],
        )
        self.assertFalse(self.streams[0].active)

    def test_short_recording_is_not_sent_for_transcription(self):
        with patch.dict(sys.modules, {"sounddevice": self.sounddevice}):
            self.controller.start_recording()
            self.streams[0].feed(np.ones((100, 1), dtype=np.float32))
            self.controller.stop_recording()

        self.audio_ready.assert_not_called()
        self.error.assert_called_once_with("錄音太短了，請按住F8多說一點")

    def test_recording_is_limited_to_sixty_seconds(self):
        with patch.dict(sys.modules, {"sounddevice": self.sounddevice}):
            self.controller.start_recording()
            self.streams[0].feed(
                np.ones((16_000 * 60, 1), dtype=np.float32)
            )
            self.controller.stop_recording()

        self.audio_ready.assert_called_once()
        self.assertEqual(len(self.audio_ready.call_args.args[0]), 16_000 * 60)
        self.error.assert_called_once_with("單次錄音已達 60 秒上限")

    def test_silence_does_not_end_recording_before_button_release(self):
        silence = np.zeros((16_000 * 3, 1), dtype=np.float32)

        with patch.dict(sys.modules, {"sounddevice": self.sounddevice}):
            self.controller.start_recording()
            self.streams[0].feed(silence)

        self.audio_ready.assert_not_called()
        self.assertTrue(self.controller._pressed)
        self.controller.stop_recording()
        self.audio_ready.assert_called_once()

    def test_partial_audio_is_sent_for_local_transcription_while_recording(self):
        partial_audio_ready = Event()
        partial_audio = []

        def on_partial_audio(audio):
            partial_audio.append(audio)
            partial_audio_ready.set()

        controller = PushToTalk(
            self.recording_changed,
            self.audio_ready,
            self.error,
            hotkey="f8",
            on_partial_audio=on_partial_audio,
        )
        audio = np.full((16_000 * 2, 1), 0.1, dtype=np.float32)

        with patch.dict(sys.modules, {"sounddevice": self.sounddevice}):
            controller.start_recording()
            self.streams[0].feed(audio)
            self.assertTrue(partial_audio_ready.wait(timeout=2))
            self.assertTrue(controller._pressed)
            controller.stop_recording()

        self.assertEqual(len(partial_audio), 1)
        self.assertEqual(len(partial_audio[0]), 16_000 * 2)

    def test_utterance_is_emitted_after_silence_without_stopping_recording(self):
        utterance_ready = Event()
        utterances = []

        def on_utterance_audio(audio, recording_id, started_at, ended_at):
            utterances.append((audio, recording_id, started_at, ended_at))
            utterance_ready.set()

        controller = PushToTalk(
            self.recording_changed,
            self.audio_ready,
            self.error,
            hotkey="f8",
            on_utterance_audio=on_utterance_audio,
        )
        speech = np.full((16_000, 1), 0.1, dtype=np.float32)
        silence = np.zeros((int(16_000 * 0.5), 1), dtype=np.float32)

        with patch.dict(sys.modules, {"sounddevice": self.sounddevice}):
            controller.start_recording()
            self.streams[0].feed(speech)
            self.streams[0].feed(silence)
            self.assertTrue(utterance_ready.wait(timeout=2))
            self.assertTrue(controller._pressed)
            controller.stop_recording()

        self.assertEqual(len(utterances), 1)
        self.assertEqual(utterances[0][1], 1)
        self.assertGreaterEqual(len(utterances[0][0]), 16_000)
        self.assertLessEqual(utterances[0][2], utterances[0][3])
        self.audio_ready.assert_not_called()
        self.error.assert_not_called()

    def test_short_noise_burst_does_not_create_an_utterance(self):
        on_utterance_audio = Mock()
        controller = PushToTalk(
            self.recording_changed,
            self.audio_ready,
            self.error,
            hotkey="f8",
            on_utterance_audio=on_utterance_audio,
        )
        noise_burst = np.full((int(16_000 * 0.08), 1), 0.1, dtype=np.float32)
        silence = np.zeros((int(16_000 * 0.5), 1), dtype=np.float32)

        with patch.dict(sys.modules, {"sounddevice": self.sounddevice}):
            controller.start_recording()
            self.streams[0].feed(noise_burst)
            self.streams[0].feed(silence)
            controller.stop_recording()

        on_utterance_audio.assert_not_called()
        self.error.assert_not_called()

    def test_releasing_button_flushes_the_last_unpaused_utterance(self):
        utterance_ready = Event()
        utterances = []

        def on_utterance_audio(audio, recording_id, started_at, ended_at):
            utterances.append((audio, recording_id, started_at, ended_at))
            utterance_ready.set()

        controller = PushToTalk(
            self.recording_changed,
            self.audio_ready,
            self.error,
            hotkey="f8",
            on_utterance_audio=on_utterance_audio,
        )

        with patch.dict(sys.modules, {"sounddevice": self.sounddevice}):
            controller.start_recording()
            self.streams[0].feed(
                np.full((16_000, 1), 0.1, dtype=np.float32)
            )
            controller.stop_recording()
            self.assertTrue(utterance_ready.wait(timeout=2))

        self.assertEqual(len(utterances), 1)
        self.assertEqual(utterances[0][1], 1)
        self.audio_ready.assert_not_called()
        self.error.assert_not_called()

    def test_continuous_utterance_mode_keeps_recording_past_sixty_seconds(self):
        controller = PushToTalk(
            self.recording_changed,
            self.audio_ready,
            self.error,
            hotkey="f8",
            on_utterance_audio=Mock(),
        )

        with patch.dict(sys.modules, {"sounddevice": self.sounddevice}):
            controller.start_recording()
            self.streams[0].feed(
                np.zeros((16_000 * 60, 1), dtype=np.float32)
            )
            self.assertTrue(controller._pressed)
            self.assertTrue(self.streams[0].active)
            controller.stop_recording()

        self.error.assert_not_called()

    def test_continuous_mode_ignores_audio_while_reply_is_playing(self):
        should_pause = [True]
        utterance_ready = Event()
        utterances = []

        def on_utterance_audio(audio, recording_id, started_at, ended_at):
            utterances.append(audio)
            utterance_ready.set()

        controller = PushToTalk(
            self.recording_changed,
            self.audio_ready,
            self.error,
            hotkey="f8",
            on_utterance_audio=on_utterance_audio,
            should_pause_utterance_detection=lambda: should_pause[0],
        )
        speech = np.full((16_000, 1), 0.1, dtype=np.float32)
        silence = np.zeros((int(16_000 * 0.5), 1), dtype=np.float32)

        with patch.dict(sys.modules, {"sounddevice": self.sounddevice}):
            controller.start_recording()
            self.streams[0].feed(speech)
            self.streams[0].feed(silence)
            self.assertEqual(utterances, [])

            should_pause[0] = False
            self.streams[0].feed(speech)
            self.streams[0].feed(silence)
            self.assertTrue(utterance_ready.wait(timeout=2))
            controller.stop_recording()

        self.assertEqual(len(utterances), 1)

    def test_toggle_recording_starts_then_stops(self):
        with patch.dict(sys.modules, {"sounddevice": self.sounddevice}):
            self.controller.toggle_recording()
            self.assertTrue(self.controller._pressed)
            self.controller.toggle_recording()

        self.assertFalse(self.controller._pressed)
        self.recording_changed.assert_has_calls(
            [unittest.mock.call(True), unittest.mock.call(False)]
        )

    def test_duplicate_press_does_not_open_another_stream(self):
        with patch.dict(sys.modules, {"sounddevice": self.sounddevice}):
            self.controller.start_recording()
            self.controller.start_recording()

        self.assertEqual(len(self.streams), 1)

    def test_close_discards_an_active_recording(self):
        with patch.dict(sys.modules, {"sounddevice": self.sounddevice}):
            self.controller.start_recording()
            self.streams[0].feed(
                np.ones((MIN_RECORDING_SAMPLES + 100, 1), dtype=np.float32)
            )
            self.controller.close()
            self.controller.start_recording()

        self.audio_ready.assert_not_called()
        self.assertEqual(len(self.streams), 1)

    def test_keyboard_listener_toggles_recording_on_key_down_only(self):
        key = object()
        listener = Mock()
        keyboard = types.SimpleNamespace(
            Key=types.SimpleNamespace(f8=key),
            Listener=Mock(return_value=listener),
        )
        with patch.dict(sys.modules, {"pynput": types.SimpleNamespace(keyboard=keyboard)}):
            self.controller.start_listener()

        keyboard.Listener.assert_called_once()
        callbacks = keyboard.Listener.call_args.kwargs
        callbacks["on_press"](key)
        callbacks["on_press"](key)
        self.assertTrue(self.controller._pressed)
        callbacks["on_release"](key)
        self.assertTrue(self.controller._pressed)
        callbacks["on_press"](key)
        self.assertFalse(self.controller._pressed)
        callbacks["on_release"](key)
        self.recording_changed.assert_has_calls(
        [unittest.mock.call(True), unittest.mock.call(False)]
        )
        listener.start.assert_called_once()

    def test_mouse_forward_side_button_toggles_on_click_not_release(self):
        back_button = object()
        forward_button = object()
        listener = Mock()
        mouse = types.SimpleNamespace(
            Button=types.SimpleNamespace(x1=back_button, x2=forward_button),
            Listener=Mock(return_value=listener),
        )
        controller = PushToTalk(
            self.recording_changed,
            self.audio_ready,
            self.error,
            hotkey="mouse_x2",
        )

        with patch.dict(
            sys.modules,
            {
                "pynput": types.SimpleNamespace(mouse=mouse),
                "sounddevice": self.sounddevice,
            },
        ):
            controller.start_listener()
            on_click = mouse.Listener.call_args.kwargs["on_click"]
            on_click(0, 0, back_button, True)
            on_click(0, 0, forward_button, True)
            on_click(0, 0, forward_button, False)
            self.assertTrue(controller._pressed)
            self.streams[0].feed(
                np.ones(MIN_RECORDING_SAMPLES + 100, dtype=np.float32)[:, None]
            )
            on_click(0, 0, forward_button, True)
            on_click(0, 0, forward_button, False)

        self.assertEqual(len(self.streams), 1)
        self.audio_ready.assert_called_once()
        self.recording_changed.assert_has_calls(
            [unittest.mock.call(True), unittest.mock.call(False)]
        )
        listener.start.assert_called_once()

    def test_unavailable_microphone_is_reported(self):
        broken_sounddevice = types.SimpleNamespace(
            InputStream=Mock(side_effect=RuntimeError("no input device"))
        )
        with patch.dict(sys.modules, {"sounddevice": broken_sounddevice}):
            self.controller.start_recording()

        self.recording_changed.assert_has_calls(
            [unittest.mock.call(True), unittest.mock.call(False)]
        )
        self.error.assert_called_once_with("無法開啟麥克風：no input device")


if __name__ == "__main__":
    unittest.main()

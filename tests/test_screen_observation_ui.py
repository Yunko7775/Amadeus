import unittest
from unittest.mock import Mock, patch

from PIL import Image

from desktop import desktop_app
from shared.screen_observation import has_significant_screen_change


class ScreenChangeDetectionTests(unittest.TestCase):
    def test_first_frame_and_resolution_changes_count_as_changes(self):
        image = Image.new("RGB", (320, 180), "black")

        self.assertTrue(has_significant_screen_change(None, image))
        self.assertTrue(
            has_significant_screen_change(
                image,
                Image.new("RGB", (640, 360), "black"),
            )
        )

    def test_unchanged_or_minor_changes_do_not_trigger_analysis(self):
        previous = Image.new("RGB", (320, 180), "black")
        minor_change = previous.copy()
        for x in range(10):
            for y in range(10):
                minor_change.putpixel((x, y), (255, 255, 255))

        self.assertFalse(has_significant_screen_change(previous, previous.copy()))
        self.assertFalse(has_significant_screen_change(previous, minor_change))

    def test_significant_changes_trigger_analysis(self):
        previous = Image.new("RGB", (320, 180), "black")
        current = Image.new("RGB", (320, 180), "white")

        self.assertTrue(has_significant_screen_change(previous, current))


class ScreenObservationUiTests(unittest.TestCase):
    def make_app(self):
        app = desktop_app.Live2DChatApp.__new__(desktop_app.Live2DChatApp)
        app.master = Mock()
        app.auto_screen_active = True
        app.closing = False
        app.screen_interval = 60
        app.screen_observation_after_id = "timer-1"
        app.last_observation_image = None
        app.screen_observation_generation = 0
        app.activity_generation = 0
        app.is_screen_processing = False
        app.proactive_request_active = False
        app.speech_active = False
        app.btn_auto_screen = Mock()
        app.show_voice_status = Mock()
        app.hide_voice_status = Mock()
        app.display_message = Mock()
        app.process_and_stream = Mock()
        app.live2d = Mock()
        return app

    def test_start_schedules_capture_on_the_tk_event_loop(self):
        app = self.make_app()
        app.auto_screen_active = False

        app.toggle_auto_screen()

        app.master.after.assert_called_once_with(
            0,
            app.capture_and_analyze_screen,
        )
        app.btn_auto_screen.config.assert_called_once()

    def test_stop_cancels_pending_capture(self):
        app = self.make_app()

        app.toggle_auto_screen()

        app.master.after_cancel.assert_called_once_with("timer-1")
        self.assertFalse(app.auto_screen_active)
        self.assertIsNone(app.screen_observation_after_id)

    def test_unchanged_screen_skips_model_call_and_retries_later(self):
        app = self.make_app()
        app.last_observation_image = Image.new("RGB", (320, 180), "black")

        with patch.object(
            desktop_app.ImageGrab,
            "grab",
            return_value=app.last_observation_image.copy(),
        ), patch.object(desktop_app.threading, "Thread") as thread:
            app.capture_and_analyze_screen()

        thread.assert_not_called()
        self.assertFalse(app.is_screen_processing)
        app.hide_voice_status.assert_not_called()
        app.show_voice_status.assert_not_called()
        app.master.after.assert_called_once_with(
            60_000,
            app.capture_and_analyze_screen,
        )
        app.master.withdraw.assert_called_once()
        app.master.deiconify.assert_called_once()

    def test_screen_capture_restores_the_original_keyboard_focus(self):
        app = self.make_app()
        focused_widget = Mock()
        focused_widget.winfo_exists.return_value = True
        app.master.focus_get.return_value = focused_widget
        app.last_observation_image = Image.new("RGB", (320, 180), "black")

        with patch.object(
            desktop_app.ImageGrab,
            "grab",
            return_value=app.last_observation_image.copy(),
        ):
            app.capture_and_analyze_screen()

        focused_widget.focus_set.assert_called_once_with()

    def test_changed_screen_starts_one_background_request(self):
        app = self.make_app()
        app.last_observation_image = Image.new("RGB", (320, 180), "black")
        screenshot = Image.new("RGB", (320, 180), "white")

        with patch.object(
            desktop_app.ImageGrab,
            "grab",
            return_value=screenshot,
        ), patch.object(desktop_app.threading, "Thread") as thread:
            app.capture_and_analyze_screen()

        self.assertTrue(app.is_screen_processing)
        self.assertIsNot(app.last_observation_image, screenshot)
        thread.assert_called_once()
        self.assertEqual(
            thread.call_args.kwargs["target"],
            app.send_screen_observation,
        )
        self.assertEqual(
            thread.call_args.kwargs["args"][1],
            app.screen_observation_generation,
        )
        self.assertEqual(
            thread.call_args.kwargs["args"][2],
            app.activity_generation,
        )
        thread.return_value.start.assert_called_once()

    def test_pending_result_is_discarded_after_observation_is_stopped(self):
        app = self.make_app()
        app.auto_screen_active = False
        app.is_screen_processing = True
        app.screen_observation_generation = 1

        app.finish_screen_observation(
            {
                "reply": "舊畫面的回覆",
                "japanese_reply": "古い画面への返事",
                "expression": "happy",
            },
            generation=0,
            activity_generation=0,
        )

        self.assertFalse(app.is_screen_processing)
        app.display_message.assert_not_called()
        app.process_and_stream.assert_not_called()
        app.live2d.set_expression.assert_not_called()

    def test_failed_analysis_retries_same_screen_next_interval(self):
        app = self.make_app()
        app.last_observation_image = Image.new("RGB", (320, 180), "black")

        app.finish_screen_observation(
            None,
            RuntimeError("API unavailable"),
            generation=0,
            activity_generation=0,
        )

        self.assertIsNone(app.last_observation_image)
        app.show_voice_status.assert_called_once_with(
            "觀測分析失敗，請檢查後端與 API 狀態"
        )

    def test_analysis_is_discarded_after_user_activity(self):
        app = self.make_app()
        app.activity_generation = 2
        app.is_screen_processing = True

        app.finish_screen_observation(
            {
                "reply": "分析回覆",
                "japanese_reply": "返事",
                "expression": "happy",
            },
            generation=0,
            activity_generation=1,
        )

        self.assertFalse(app.is_screen_processing)
        app.display_message.assert_not_called()
        app.process_and_stream.assert_not_called()


if __name__ == "__main__":
    unittest.main()

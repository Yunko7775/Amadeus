import unittest
from datetime import datetime
from unittest.mock import Mock, call, patch

from desktop import desktop_app


class ProactiveUiTests(unittest.TestCase):
    def make_app(self):
        app = desktop_app.Live2DChatApp.__new__(desktop_app.Live2DChatApp)
        app.proactive_request_active = True
        app.closing = False
        app.activity_generation = 3
        app.last_interaction_at = 0
        app.proactive_status_after_id = None
        app.proactive_started_at = None
        app.master = Mock()
        app.show_voice_status = Mock()
        return app

    def test_screen_observation_does_not_block_idle_proactive_calls(self):
        app = self.make_app()
        app.auto_screen_active = True
        app.is_screen_processing = False
        app.proactive_request_active = False
        app.speech_active = False
        app.user_input = Mock()
        app.user_input.get.return_value = ""

        self.assertFalse(app.is_proactive_busy())

    def test_screen_processing_and_speech_still_defer_proactive_calls(self):
        app = self.make_app()
        app.auto_screen_active = True
        app.is_screen_processing = True
        app.proactive_request_active = False
        app.speech_active = False
        app.user_input = Mock()
        app.user_input.get.return_value = ""

        self.assertTrue(app.is_proactive_busy())

        app.is_screen_processing = False
        app.speech_active = True
        self.assertTrue(app.is_proactive_busy())

    def test_manual_trigger_explains_when_backend_has_no_message(self):
        app = self.make_app()

        app.show_proactive_message(
            {"reply": "", "japanese_reply": "", "expression": "neutral"},
            generation=3,
            force=True,
        )

        self.assertFalse(app.proactive_request_active)
        app.show_voice_status.assert_called_once_with(
            "這次沒有找到值得分享的話題"
        )

    def test_manual_trigger_reports_backend_request_failure(self):
        app = self.make_app()

        app.finish_proactive_request(force=True, error="connection refused")

        self.assertFalse(app.proactive_request_active)
        app.show_voice_status.assert_called_once_with(
            "主動訊息失敗，請檢查後端與 API 狀態"
        )

    def test_manual_trigger_explains_request_timeout(self):
        app = self.make_app()

        app.finish_proactive_request(force=True, error="timeout")

        self.assertFalse(app.proactive_request_active)
        app.show_voice_status.assert_called_once_with(
            "主動訊息處理逾時，請稍後再試"
        )

    def test_proactive_button_is_disabled_while_request_is_running(self):
        app = self.make_app()
        app.btn_proactive = Mock()

        app.set_proactive_button_busy(True)

        app.btn_proactive.config.assert_called_once_with(
            state=desktop_app.tk.DISABLED,
            text="處理中...",
        )

        app.btn_proactive.reset_mock()
        app.set_proactive_button_busy(False)

        app.btn_proactive.config.assert_called_once_with(
            state=desktop_app.tk.NORMAL,
            text="主動聊聊",
        )

    def test_manual_trigger_does_not_start_a_duplicate_request(self):
        app = self.make_app()

        app.start_proactive_request(force=True)

        app.show_voice_status.assert_not_called()

    def test_manual_request_tells_backend_to_open_a_conversation(self):
        app = self.make_app()
        response = Mock()
        response.status_code = 200
        response.json.return_value = {"reply": "現在在忙什麼？"}

        with patch.object(
            desktop_app.requests,
            "post",
            return_value=response,
        ) as post:
            app.request_proactive_message(generation=3, force=True)

        self.assertEqual(
            post.call_args.kwargs["json"],
            {"user_id": "user_001", "force": True},
        )

    def test_daily_greeting_request_is_marked_for_backend(self):
        app = self.make_app()
        response = Mock()
        response.status_code = 200
        response.json.return_value = {"reply": "早安。"}

        with patch.object(
            desktop_app.requests,
            "post",
            return_value=response,
        ) as post:
            app.request_proactive_message(generation=3, force=True, greeting=True)

        self.assertEqual(
            post.call_args.kwargs["json"],
            {"user_id": "user_001", "force": True, "greeting": True},
        )

    def test_daily_greeting_is_skipped_if_already_sent_today(self):
        app = self.make_app()
        app.closing = False

        with patch.object(
            desktop_app,
            "get_application_state",
            return_value=datetime.now().astimezone().date().isoformat(),
        ), patch.object(app, "start_proactive_request") as start_request:
            app.start_daily_greeting()

        start_request.assert_not_called()

    def test_daily_greeting_waits_until_the_app_is_idle(self):
        app = self.make_app()
        app.closing = False
        app.is_proactive_busy = Mock(return_value=True)

        with patch.object(desktop_app, "get_application_state", return_value=None), \
                patch.object(app, "start_proactive_request") as start_request:
            app.start_daily_greeting()

        app.master.after.assert_called_once_with(60_000, app.start_daily_greeting)
        start_request.assert_not_called()

    def test_daily_greeting_starts_only_after_database_has_no_sent_date(self):
        app = self.make_app()
        app.closing = False
        app.is_proactive_busy = Mock(return_value=False)

        with patch.object(desktop_app, "get_application_state", return_value=None), \
                patch.object(app, "start_proactive_request") as start_request:
            app.start_daily_greeting()

        start_request.assert_called_once_with(force=True, greeting=True)

    def test_midnight_calendar_rollover_waits_for_the_first_chat(self):
        app = self.make_app()
        app.closing = False
        app._daily_greeting_calendar_date = "2026-07-24"
        app._greeting_waiting_for_first_chat = False
        now = datetime(2026, 7, 25, 0, 0).astimezone()

        with patch.object(
            desktop_app,
            "daily_greeting_context",
            return_value={"period": "深夜", "holiday": None},
        ) as greeting_context:
            changed = app.refresh_daily_greeting_calendar(now)

        self.assertTrue(changed)
        self.assertEqual(app._daily_greeting_calendar_date, "2026-07-25")
        self.assertTrue(app._greeting_waiting_for_first_chat)
        greeting_context.assert_called_once_with(now)

    def test_daily_greeting_persists_date_after_displaying_reply(self):
        app = self.make_app()
        app.live2d = Mock()
        app.display_message = Mock()
        app.set_proactive_button_busy = Mock()
        app.clear_proactive_status_timer = Mock()

        with patch.object(desktop_app, "set_application_state") as save_state:
            app.show_proactive_message(
                {
                    "reply": "早安。",
                    "japanese_reply": "",
                    "expression": "happy",
                },
                generation=3,
                force=True,
                greeting=True,
            )

        self.assertEqual(
            app.display_message.call_args_list,
            [
                call("早安。", auto_hide=False),
                call("早安。", auto_hide=True),
            ],
        )
        save_state.assert_called_once_with(
            "daily_greeting_date",
            datetime.now().astimezone().date().isoformat(),
        )

    def test_manual_reply_is_shown_even_if_user_activity_changed(self):
        app = self.make_app()
        app.activity_generation = 4
        app.live2d = Mock()
        app.display_message = Mock()

        app.show_proactive_message(
            {
                "reply": "新查到一個有趣的結果。",
                "japanese_reply": "",
                "expression": "surprised",
            },
            generation=3,
            force=True,
        )

        self.assertEqual(
            app.display_message.call_args_list,
            [
                call(
                    "新查到一個有趣的結果。",
                    auto_hide=False,
                ),
                call(
                    "新查到一個有趣的結果。",
                    auto_hide=True,
                ),
            ],
        )
        app.live2d.set_expression.assert_called_once_with("surprised")
        app.show_voice_status.assert_not_called()

    def test_stale_automatic_reply_is_still_discarded(self):
        app = self.make_app()
        app.activity_generation = 4
        app.live2d = Mock()
        app.display_message = Mock()

        app.show_proactive_message(
            {
                "reply": "自動訊息。（neutral）",
                "japanese_reply": "",
                "expression": "neutral",
            },
            generation=3,
            force=False,
        )

        app.display_message.assert_not_called()


if __name__ == "__main__":
    unittest.main()

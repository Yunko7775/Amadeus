import unittest
from types import SimpleNamespace
from unittest.mock import patch

import config

config.API_KEY = "test-key"

import flask_backend
from proactive import should_send_proactive


class ProactiveTests(unittest.TestCase):
    def test_proactive_message_waits_for_configured_idle_period(self):
        self.assertFalse(should_send_proactive(3599, 0, 3600, False, False))
        self.assertTrue(should_send_proactive(3600, 0, 3600, False, False))

    def test_proactive_message_is_skipped_when_busy_or_already_sent(self):
        self.assertFalse(should_send_proactive(4000, 0, 3600, True, False))
        self.assertFalse(should_send_proactive(4000, 0, 3600, False, True))

    @patch.object(flask_backend, "translate_to_japanese", return_value="日本語の一言。")
    @patch.object(
        flask_backend.client.models,
        "generate_content",
        return_value=SimpleNamespace(text="你今天忙了很久吧，休息一下也行。（neutral）"),
    )
    def test_proactive_endpoint_returns_persona_reply_and_speech_text(
        self,
        generate_content,
        _translate,
    ):
        with flask_backend.app.test_client() as client:
            response = client.post("/proactive", json={"user_id": "test"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.get_json(),
            {
                "reply": "你今天忙了很久吧，休息一下也行。",
                "japanese_reply": "日本語の一言。",
                "expression": "neutral",
            },
        )
        self.assertIn("最多兩句", generate_content.call_args.kwargs["contents"][0].parts[0].text)


if __name__ == "__main__":
    unittest.main()

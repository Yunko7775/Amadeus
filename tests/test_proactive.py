import io
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from shared import config

config.API_KEY = "test-key"

from backend import memory_manager, flask_backend
from shared.proactive import should_send_proactive


class ProactiveTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_patch = patch.object(
            memory_manager,
            "MEMORY_DB_PATH",
            Path(self.temp_dir.name) / "memory.sqlite3",
        )
        self.legacy_patch = patch.object(
            memory_manager,
            "LEGACY_MEMORY_FILE",
            Path(self.temp_dir.name) / "missing-user-memory.json",
        )
        self.database_patch.start()
        self.legacy_patch.start()
        flask_backend.conversation_history.clear()

    def tearDown(self):
        self.database_patch.stop()
        self.legacy_patch.stop()
        self.temp_dir.cleanup()

    @patch.object(flask_backend, "search_web")
    @patch.object(flask_backend, "translate_to_japanese", return_value="日本語。")
    @patch.object(flask_backend, "append_assistant_history")
    @patch.object(
        flask_backend.proactive_client.models,
        "generate_content",
        return_value=SimpleNamespace(
            text=(
                '{"action":"check_in",'
                '"reply":"你現在在忙什麼？有空的話陪我聊兩句吧。（neutral）"}'
            )
        ),
    )
    def test_manual_proactive_trigger_returns_a_simple_check_in(
        self,
        generate_content,
        append_history,
        _translate,
        search,
    ):
        with patch.object(flask_backend, "build_full_prompt", return_value="CONTEXT"):
            with patch.object(flask_backend, "get_conversation_history", return_value=[]):
                with patch.object(
                    flask_backend,
                    "get_recent_agent_explorations",
                    return_value=[],
                ):
                    with flask_backend.app.test_client() as client:
                        response = client.post(
                            "/proactive",
                            json={"user_id": "test", "force": True},
                        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.get_json(),
            {
                "reply": "你現在在忙什麼？有空的話陪我聊兩句吧。",
                "japanese_reply": "日本語。",
                "expression": "neutral",
            },
        )
        prompt = generate_content.call_args.kwargs["contents"][0].parts[0].text
        self.assertIn('"action":"check_in"', prompt)
        self.assertIn('"action":"start_exploration"', prompt)
        self.assertIn("不可選 silent", prompt)
        search.assert_not_called()
        append_history.assert_called_once_with(
            "test",
            "你現在在忙什麼？有空的話陪我聊兩句吧。",
        )

    @patch.object(flask_backend, "search_web")
    @patch.object(flask_backend, "translate_to_japanese", return_value="日本語。")
    @patch.object(flask_backend, "append_assistant_history")
    @patch.object(
        flask_backend.proactive_client.models,
        "generate_content",
        side_effect=[
            SimpleNamespace(text='{"action":"silent"}'),
            SimpleNamespace(text='{"action":"silent"}'),
        ],
    )
    def test_manual_check_in_has_safe_fallback_if_model_keeps_silent(
        self,
        generate_content,
        append_history,
        _translate,
        search,
    ):
        with patch.object(flask_backend, "build_full_prompt", return_value="CONTEXT"):
            with patch.object(flask_backend, "get_conversation_history", return_value=[]):
                with patch.object(
                    flask_backend,
                    "get_recent_agent_explorations",
                    return_value=[],
                ):
                    with flask_backend.app.test_client() as client:
                        response = client.post(
                            "/proactive",
                            json={"user_id": "test", "force": True},
                        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("你現在在忙什麼？", response.get_json()["reply"])
        self.assertEqual(generate_content.call_count, 2)
        append_history.assert_called_once()
        search.assert_not_called()

    @patch.object(flask_backend, "record_agent_exploration")
    @patch.object(flask_backend, "search_web", return_value=[
        {
            "title": "New evidence",
            "url": "https://example.com/water",
            "snippet": "Evidence about subsurface water.",
        }
    ])
    @patch.object(flask_backend, "translate_to_japanese", return_value="日本語。")
    @patch.object(flask_backend, "append_assistant_history")
    @patch.object(
        flask_backend.proactive_client.models,
        "generate_content",
        side_effect=[
            SimpleNamespace(
                text=(
                    '{"action":"continue_exploration","note_index":1,'
                    '"query":"火星地下液態水研究"}'
                )
            ),
            SimpleNamespace(
                text=(
                    '{"share":"火星地下水又有了新線索。（surprised）",'
                    '"summary":"新研究發現地下水活動線索。",'
                    '"next_question":"這種活動可能持續多久？"}'
                )
            ),
        ],
    )
    def test_manual_trigger_can_continue_a_pending_exploration(
        self,
        generate_content,
        append_history,
        _translate,
        search,
        record,
    ):
        note = {
            "topic": "火星上的水",
            "query": "火星水研究",
            "summary": "火星曾存在液態水環境。",
            "sources": [],
            "next_question": "火星地下是否仍有液態水？",
            "explored_at": "2026-10-30",
        }
        with patch.object(flask_backend, "build_full_prompt", return_value="CONTEXT"):
            with patch.object(flask_backend, "get_conversation_history", return_value=[]):
                with patch.object(
                    flask_backend,
                    "get_recent_agent_explorations",
                    return_value=[note],
                ):
                    with flask_backend.app.test_client() as client:
                        response = client.post(
                            "/proactive",
                            json={"user_id": "test", "force": True},
                        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["reply"], "火星地下水又有了新線索。")
        search.assert_called_once_with("火星地下液態水研究", limit=3)
        record.assert_called_once()
        append_history.assert_called_once_with(
            "test",
            "火星地下水又有了新線索。",
        )
        decision_prompt = generate_content.call_args_list[0].kwargs[
            "contents"
        ][0].parts[0].text
        self.assertIn('"action":"continue_exploration"', decision_prompt)
        self.assertIn('"note_index":筆記編號', decision_prompt)

    def test_proactive_force_flag_must_be_boolean(self):
        with flask_backend.app.test_client() as client:
            response = client.post(
                "/proactive",
                json={"user_id": "test", "force": "true"},
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error"], "invalid_force_value")

    def test_proactive_greeting_flag_must_be_boolean(self):
        with flask_backend.app.test_client() as client:
            response = client.post(
                "/proactive",
                json={"user_id": "test", "greeting": "true"},
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error"], "invalid_greeting_value")

    def test_chat_birthday_wish_is_saved_for_the_current_year(self):
        class BirthdayDateTime(datetime):
            @classmethod
            def now(cls, tz=None):
                value = cls(2026, 7, 25, 12, tzinfo=timezone.utc)
                return value.astimezone(tz) if tz else value

        with (
            patch.object(flask_backend, "datetime", BirthdayDateTime),
            patch.object(flask_backend, "set_application_state") as save_state,
            patch.object(flask_backend, "build_full_prompt", return_value="context"),
            patch.object(flask_backend, "ENABLE_WEB_SEARCH", False),
            patch.object(flask_backend, "build_interaction_tools", return_value=[]),
            patch.object(
                flask_backend.client.models,
                "generate_content",
                return_value=SimpleNamespace(
                    text=(
                        "【中文回覆】\n"
                        "生日快樂，紅莉栖。（happy）\n"
                        "【日文語音稿】\n"
                        "誕生日おめでとう、紅莉栖。"
                    )
                ),
            ),
            patch.object(flask_backend, "translate_to_japanese", return_value=""),
            patch.object(
                flask_backend,
                "update_conversation_history",
                return_value="turn",
            ),
            patch.object(flask_backend, "_index_conversation_turn_safely"),
            patch.object(flask_backend, "_schedule_memory_consolidation"),
        ):
            with flask_backend.app.test_client() as client:
                response = client.post(
                    "/ask",
                    json={"user_id": "user_001", "message": "紅莉栖，生日快樂！"},
                )

        self.assertEqual(response.status_code, 200)
        save_state.assert_called_once_with(
            "kurisu_birthday_wished_2026",
            "2026-07-25",
        )

    def test_explicit_user_birthday_flushes_memory_consolidation(self):
        with (
            patch.object(flask_backend, "build_full_prompt", return_value="context"),
            patch.object(flask_backend, "ENABLE_WEB_SEARCH", False),
            patch.object(flask_backend, "build_interaction_tools", return_value=[]),
            patch.object(
                flask_backend.client.models,
                "generate_content",
                return_value=SimpleNamespace(
                    text=(
                        "【中文回覆】\n"
                        "好，我會記住。（happy）\n"
                        "【日文語音稿】\n"
                        "うん、覚えておくね。"
                    )
                ),
            ),
            patch.object(flask_backend, "translate_to_japanese", return_value=""),
            patch.object(
                flask_backend,
                "update_conversation_history",
                return_value="turn",
            ),
            patch.object(flask_backend, "_index_conversation_turn_safely"),
            patch.object(
                flask_backend,
                "_schedule_memory_consolidation",
            ) as schedule_memory,
        ):
            with flask_backend.app.test_client() as client:
                response = client.post(
                    "/ask",
                    json={
                        "user_id": "user_001",
                        "message": "我生日是7月25日，記住喔",
                    },
                )

        self.assertEqual(response.status_code, 200)
        schedule_memory.assert_called_once_with(
            "user_001",
            flush_pending=True,
        )

    @patch.object(flask_backend, "daily_greeting_context", return_value={
        "period": "早上",
        "holiday": "Lunar New Year",
    })
    @patch.object(flask_backend, "translate_to_japanese", return_value="おはよう。")
    @patch.object(flask_backend, "append_assistant_history")
    @patch.object(
        flask_backend.proactive_client.models,
        "generate_content",
        return_value=SimpleNamespace(
            text='{"reply":"早安，新年快樂。（happy）"}'
        ),
    )
    def test_daily_greeting_uses_local_holiday_context_without_memory(
        self,
        generate_content,
        append_history,
        _translate,
        _greeting_context,
    ):
        with patch.object(flask_backend, "build_full_prompt") as build_prompt:
            with flask_backend.app.test_client() as client:
                response = client.post(
                    "/proactive",
                    json={"user_id": "test", "greeting": True},
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.get_json(),
            {
                "reply": "早安，新年快樂。",
                "japanese_reply": "おはよう。",
                "expression": "happy",
            },
        )
        prompt = generate_content.call_args.kwargs["contents"][0].parts[0].text
        self.assertIn("現在是早上", prompt)
        self.assertIn("Lunar New Year", prompt)
        build_prompt.assert_not_called()
        append_history.assert_called_once_with("test", "早安，新年快樂。")

    def test_proactive_reply_is_appended_without_inventing_a_user_message(self):
        user_id = "proactive-history-test"
        flask_backend.conversation_history[user_id] = [
            {"role": "user", "content": "之前的問題"},
            {"role": "assistant", "content": "之前的回答"},
        ]

        flask_backend.append_assistant_history(user_id, "主動查證的新發現")

        self.assertEqual(
            flask_backend.conversation_history[user_id],
            [
                {"role": "user", "content": "之前的問題"},
                {"role": "assistant", "content": "之前的回答"},
                {"role": "assistant", "content": "主動查證的新發現"},
            ],
        )
        del flask_backend.conversation_history[user_id]

    def test_conversation_history_is_restored_from_sqlite_after_cache_reset(self):
        with redirect_stdout(io.StringIO()):
            flask_backend.update_conversation_history(
                "persistent-test",
                "把 Amadeus 移植到手機很困難",
                "可以先從簡化介面開始。",
            )
        del flask_backend.conversation_history["persistent-test"]

        with redirect_stdout(io.StringIO()):
            history = flask_backend.get_conversation_history("persistent-test")
        self.assertEqual(
            history,
            [
                {"role": "user", "content": "把 Amadeus 移植到手機很困難"},
                {"role": "assistant", "content": "可以先從簡化介面開始。"},
            ],
        )

    def test_proactive_client_retries_transient_errors_with_bounded_timeout(self):
        options = flask_backend.proactive_client._api_client._http_options
        self.assertEqual(options.timeout, 30_000)
        self.assertEqual(
            options.retry_options.attempts,
            4,
        )
        self.assertIn(
            503,
            options.retry_options.http_status_codes,
        )

    @patch.object(
        flask_backend.proactive_client.models,
        "generate_content",
        side_effect=TimeoutError("model timed out"),
    )
    def test_proactive_model_timeout_returns_gateway_timeout(self, _generate_content):
        with patch.object(flask_backend, "build_full_prompt", return_value="CONTEXT"):
            with patch.object(flask_backend, "get_conversation_history", return_value=[]):
                with flask_backend.app.test_client() as client:
                    response = client.post("/proactive", json={"user_id": "test"})

        self.assertEqual(response.status_code, 504)
        self.assertEqual(
            response.get_json()["error"],
            "proactive_model_timeout",
        )

    def test_proactive_message_waits_for_configured_idle_period(self):
        self.assertFalse(should_send_proactive(7199, 0, 7200, False))
        self.assertTrue(should_send_proactive(7200, 0, 7200, False))

    def test_proactive_message_repeats_after_each_idle_interval(self):
        self.assertFalse(
            should_send_proactive(
                14399,
                0,
                7200,
                False,
                last_proactive_at=7200,
            )
        )
        self.assertTrue(
            should_send_proactive(
                14400,
                0,
                7200,
                False,
                last_proactive_at=7200,
            )
        )

    def test_recent_user_activity_restarts_the_idle_interval(self):
        self.assertFalse(
            should_send_proactive(
                4000,
                3500,
                3600,
                False,
                last_proactive_at=0,
            )
        )

    def test_proactive_message_is_skipped_while_busy(self):
        self.assertFalse(should_send_proactive(4000, 0, 3600, True))

    def test_manual_proactive_trigger_bypasses_idle_and_sent_checks_but_not_busy(self):
        self.assertTrue(
            should_send_proactive(
                10,
                10,
                1800,
                busy=False,
                force=True,
            )
        )
        self.assertFalse(
            should_send_proactive(
                10,
                10,
                1800,
                busy=True,
                force=True,
            )
        )

    @patch.object(flask_backend, "translate_to_japanese", return_value="日本語の一言。")
    @patch.object(
        flask_backend.proactive_client.models,
        "generate_content",
        return_value=SimpleNamespace(
            text=(
                '{"action":"continue","evidence":"I am playing a game",'
                '"reply":"你今天忙了很久吧，休息一下也行。（neutral）"}'
            )
        ),
    )
    def test_proactive_endpoint_returns_persona_reply_and_speech_text(
        self,
        generate_content,
        _translate,
    ):
        with patch.object(
            flask_backend,
            "build_full_prompt",
            return_value="RECENT CONTEXT: user is playing a game",
        ) as build_prompt:
            with patch.object(
                flask_backend,
                "get_conversation_history",
                return_value=[
                    {"role": "user", "content": "[系统自动屏幕观测]"},
                    {"role": "assistant", "content": "I see a game on your screen"},
                    {"role": "user", "content": "I am playing a game"},
                ],
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
        prompt = generate_content.call_args.kwargs["contents"][0].parts[0].text
        self.assertIn("待查問題筆記", prompt)
        self.assertIn("角色設定", prompt)
        self.assertIn("RECENT CONTEXT: user is playing a game", prompt)
        self.assertIn('"action":"start_exploration"', prompt)
        self.assertIn('"action":"continue_exploration"', prompt)
        self.assertIn('"action":"check_in"', prompt)
        self.assertIn("提出一個具體、可查證且你沒有把握的問題", prompt)
        self.assertIn("不能在搜尋前因為尚未看到可靠資料而選 silent", prompt)
        self.assertIn("應先從角色興趣提出一個可查證的未知問題", prompt)
        build_prompt.assert_called_once()
        self.assertEqual(build_prompt.call_args.args[1], "")
        self.assertEqual(
            build_prompt.call_args.kwargs["memory_query"],
            "I am playing a game\n最近討論的遊戲、專案、興趣、喜好、未完成問題、計畫、待續話題",
        )
        self.assertFalse(
            build_prompt.call_args.kwargs["include_screen_observations"]
        )

    @patch.object(flask_backend, "translate_to_japanese")
    @patch.object(
        flask_backend.proactive_client.models,
        "generate_content",
        return_value=SimpleNamespace(
            text=(
                '{"action":"continue","evidence":"你提到命運石之門的新遊戲",'
                '"reply":"要不要聊聊時間跳躍？（neutral）"}'
            )
        ),
    )
    def test_proactive_cannot_claim_user_said_something_without_history(
        self,
        _generate_content,
        translate,
    ):
        with patch.object(flask_backend, "build_full_prompt", return_value="CONTEXT") as build_prompt:
            with patch.object(flask_backend, "get_conversation_history", return_value=[]):
                with patch.object(
                    flask_backend,
                    "get_recent_agent_explorations",
                    return_value=[],
                ):
                    with flask_backend.app.test_client() as client:
                        response = client.post("/proactive", json={"user_id": "test"})

        self.assertEqual(response.status_code, 502)
        self.assertEqual(
            response.get_json()["error"],
            "proactive_decision_unavailable",
        )
        self.assertEqual(build_prompt.call_args.args[1], "")
        translate.assert_not_called()

    @patch.object(flask_backend, "translate_to_japanese")
    @patch.object(
        flask_backend.proactive_client.models,
        "generate_content",
        return_value=SimpleNamespace(
            text=(
                '{"action":"continue","evidence":"命運石之門的新遊戲",'
                '"reply":"時間跳躍理論確實有趣。（neutral）"}'
            )
        ),
    )
    def test_proactive_rejects_evidence_not_found_in_user_messages(
        self,
        _generate_content,
        translate,
    ):
        with patch.object(flask_backend, "build_full_prompt", return_value="CONTEXT"):
            with patch.object(
                flask_backend,
                "get_conversation_history",
                return_value=[
                    {"role": "user", "content": "最近在看動畫。"},
                    {"role": "assistant", "content": "哪部動畫？"},
                ],
            ):
                with patch.object(
                    flask_backend,
                    "get_recent_agent_explorations",
                    return_value=[],
                ):
                    with flask_backend.app.test_client() as client:
                        response = client.post("/proactive", json={"user_id": "test"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["reply"], "")
        translate.assert_not_called()

    def test_automatic_screen_observation_and_its_reply_are_filtered_out(self):
        history = [
            {"role": "user", "content": "I am working on a project"},
            {"role": "assistant", "content": "Tell me more"},
            {"role": "user", "content": "[系统自动屏幕观测]"},
            {"role": "assistant", "content": "The screen shows a private message"},
            {"role": "user", "content": "I am playing a game"},
        ]

        self.assertEqual(
            flask_backend._exclude_automatic_screen_observations(history),
            [
                {"role": "user", "content": "I am working on a project"},
                {"role": "assistant", "content": "Tell me more"},
                {"role": "user", "content": "I am playing a game"},
            ],
        )

    @patch.object(flask_backend, "translate_to_japanese")
    @patch.object(
        flask_backend.proactive_client.models,
        "generate_content",
        return_value=SimpleNamespace(text='{"action":"silent"}'),
    )
    def test_proactive_endpoint_stays_silent_without_a_worthwhile_topic(
        self,
        _generate_content,
        translate,
    ):
        with patch.object(flask_backend, "build_full_prompt", return_value="CONTEXT"):
            with patch.object(
                flask_backend,
                "get_conversation_history",
                return_value=[{"role": "user", "content": "今天有點累。"}],
            ):
                with flask_backend.app.test_client() as client:
                    response = client.post("/proactive", json={"user_id": "test"})

        self.assertEqual(
            response.get_json(),
            {"reply": "", "japanese_reply": "", "expression": "neutral"},
        )
        translate.assert_not_called()

    @patch.object(flask_backend, "search_web", return_value=[])
    @patch.object(
        flask_backend.proactive_client.models,
        "generate_content",
        side_effect=[
            SimpleNamespace(text='{"action":"silent"}'),
            SimpleNamespace(
                text=(
                    '{"action":"continue_exploration","note_index":1,'
                    '"query":"火星地下液態水研究"}'
                )
            ),
        ],
    )
    def test_silent_decision_retries_pending_note_when_recent_chat_exists(
        self,
        generate_content,
        search,
    ):
        note = {
            "topic": "火星上的水",
            "query": "火星水研究",
            "summary": "火星曾存在液態水環境。",
            "sources": [],
            "next_question": "火星地下是否仍有液態水？",
            "explored_at": "2026-10-30",
        }
        with patch.object(flask_backend, "build_full_prompt", return_value="CONTEXT"):
            with patch.object(
                flask_backend,
                "get_conversation_history",
                return_value=[
                    {"role": "user", "content": "沒辦法啊我只能說確實如此"},
                ],
            ):
                with patch.object(
                    flask_backend,
                    "get_recent_agent_explorations",
                    return_value=[note],
                ):
                    with flask_backend.app.test_client() as client:
                        response = client.post(
                            "/proactive",
                            json={"user_id": "test"},
                        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["reply"], "")
        self.assertEqual(generate_content.call_count, 2)
        initial_prompt = generate_content.call_args_list[0].kwargs[
            "contents"
        ][0].parts[0].text
        self.assertIn('"note_index":筆記編號', initial_prompt)
        retry_prompt = generate_content.call_args_list[1].kwargs[
            "contents"
        ][0].parts[0].text
        self.assertIn("目前有尚未完成的待查筆記", retry_prompt)
        self.assertIn("不得輸出 silent", retry_prompt)
        search.assert_called_once_with("火星地下液態水研究", limit=3)

    @patch.object(flask_backend, "search_web", return_value=[])
    @patch.object(
        flask_backend.proactive_client.models,
        "generate_content",
        side_effect=[
            SimpleNamespace(text='{"action":"silent"}'),
            SimpleNamespace(text=(
                '{"action":"start_exploration","interest":"時間旅行與物理學",'
                '"topic":"原子鐘如何測量時間膨脹","query":"原子鐘 時間膨脹 實驗"}'
            )),
        ],
    )
    def test_silent_decision_is_retried_when_exploration_is_required(
        self,
        generate_content,
        search,
    ):
        with patch.object(flask_backend, "build_full_prompt", return_value="CONTEXT"):
            with patch.object(flask_backend, "get_conversation_history", return_value=[]):
                with patch.object(
                    flask_backend,
                    "get_recent_agent_explorations",
                    return_value=[],
                ):
                    with flask_backend.app.test_client() as client:
                        response = client.post("/proactive", json={"user_id": "test"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["reply"], "")
        self.assertEqual(generate_content.call_count, 2)
        self.assertIn(
            "不得輸出 silent",
            generate_content.call_args.kwargs["contents"][0].parts[0].text,
        )
        self.assertIn(
            "不要從人工列出的興趣清單挑選",
            generate_content.call_args_list[0].kwargs["contents"][0].parts[0].text,
        )
        search.assert_called_once_with("原子鐘 時間膨脹 實驗", limit=3)

    @patch.object(flask_backend, "record_agent_exploration")
    @patch.object(flask_backend, "search_web", return_value=[
        {
            "title": "Subsurface water",
            "url": "https://example.com/water",
            "snippet": "A new observation about subsurface water.",
        }
    ])
    @patch.object(flask_backend, "translate_to_japanese", return_value="日本語。")
    @patch.object(
        flask_backend.proactive_client.models,
        "generate_content",
        side_effect=[
            SimpleNamespace(
                text=(
                    '{"action":"continue_exploration","note_index":1,'
                    '"query":"火星地下液態水研究"}'
                )
            ),
            SimpleNamespace(
                text=(
                    '{"share":"後續研究又有了新線索。（surprised）",'
                    '"summary":"新觀測提供火星地下水活動的線索。",'
                    '"next_question":"這種水活動可能持續多久？"}'
                )
            ),
        ],
    )
    def test_exploration_continues_from_saved_note(
        self,
        generate_content,
        _translate,
        search,
        record,
    ):
        existing_note = {
            "topic": "火星上的水",
            "query": "火星水研究",
            "summary": "火星曾存在液態水環境。",
            "sources": [
                {"title": "Old source", "url": "https://example.com/old", "snippet": ""}
            ],
            "next_question": "火星地下是否仍有液態水？",
            "explored_at": "2026-10-30",
        }
        with patch.object(
            flask_backend,
            "get_recent_agent_explorations",
            return_value=[existing_note],
        ):
            with patch.object(flask_backend, "build_full_prompt", return_value="CONTEXT"):
                with patch.object(flask_backend, "get_conversation_history", return_value=[]):
                    with flask_backend.app.test_client() as client:
                        response = client.post("/proactive", json={"user_id": "test"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["reply"], "後續研究又有了新線索。")
        search.assert_called_once_with("火星地下液態水研究", limit=3)
        record.assert_called_once()
        decision_prompt = generate_content.call_args_list[0].kwargs[
            "contents"
        ][0].parts[0].text
        self.assertIn("1. 主題：火星上的水", decision_prompt)
        self.assertIn('"note_index":筆記編號', decision_prompt)
        research_prompt = generate_content.call_args.kwargs["contents"][0].parts[0].text
        self.assertIn("火星曾存在液態水環境。", research_prompt)
        self.assertIn("火星地下是否仍有液態水？", research_prompt)
        self.assertIn("https://example.com/old", research_prompt)

    @patch.object(flask_backend, "search_web", return_value=[])
    @patch.object(
        flask_backend.proactive_client.models,
        "generate_content",
        side_effect=[
            SimpleNamespace(
                text=(
                    '{"action":"continue_exploration","note_index":4,'
                    '"query":"火星地下液態水研究"}'
                )
            ),
            SimpleNamespace(
                text=(
                    '{"action":"continue_exploration","note_index":1,'
                    '"query":"火星地下液態水研究"}'
                )
            )
        ],
    )
    def test_continuation_retries_note_index_outside_pending_list(
        self,
        generate_content,
        search,
    ):
        existing_note = {
            "topic": "火星上的水",
            "query": "火星水研究",
            "summary": "火星曾存在液態水環境。",
            "sources": [],
            "next_question": "火星地下是否仍有液態水？",
            "explored_at": "2026-10-30",
        }
        with patch.object(
            flask_backend,
            "get_recent_agent_explorations",
            return_value=[existing_note],
        ):
            with patch.object(flask_backend, "build_full_prompt", return_value="CONTEXT"):
                with patch.object(flask_backend, "get_conversation_history", return_value=[]):
                    with flask_backend.app.test_client() as client:
                        response = client.post("/proactive", json={"user_id": "test"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["reply"], "")
        self.assertEqual(generate_content.call_count, 2)
        self.assertIn(
            "note_index 無效",
            generate_content.call_args_list[1].kwargs["contents"][0].parts[0].text,
        )
        search.assert_called_once_with("火星地下液態水研究", limit=3)

    @patch.object(flask_backend, "search_web", return_value=[])
    @patch.object(
        flask_backend.proactive_client.models,
        "generate_content",
        side_effect=[
            SimpleNamespace(text=(
                '{"action":"start_exploration","interest":"腦科學",'
                '"topic":"BCI腦機接口最新進展","query":"BCI最新研究"}'
            )),
            SimpleNamespace(text=(
                '{"action":"start_exploration","interest":"記憶研究",'
                '"topic":"睡眠如何鞏固記憶","query":"睡眠 記憶鞏固 研究"}'
            )),
        ],
    )
    def test_exploration_limit_forces_a_new_topic_after_three_notes(
        self,
        generate_content,
        search,
    ):
        existing_notes = [
            {
                "topic": "BCI腦機接口最新進展",
                "query": f"BCI研究 {index}",
                "summary": f"研究摘要 {index}",
                "sources": [],
                "next_question": f"後續問題 {index}",
                "explored_at": f"2026-10-31 14:5{index}:00",
            }
            for index in range(3)
        ]
        with patch.object(
            flask_backend,
            "get_recent_agent_explorations",
            return_value=existing_notes,
        ):
            with patch.object(flask_backend, "build_full_prompt", return_value="CONTEXT"):
                with patch.object(flask_backend, "get_conversation_history", return_value=[]):
                    with flask_backend.app.test_client() as client:
                        response = client.post("/proactive", json={"user_id": "test"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["reply"], "")
        self.assertEqual(generate_content.call_count, 2)
        first_prompt = generate_content.call_args_list[0].kwargs[
            "contents"
        ][0].parts[0].text
        retry_prompt = generate_content.call_args_list[1].kwargs[
            "contents"
        ][0].parts[0].text
        self.assertIn("唯一允許的 action 是 start_exploration", first_prompt)
        self.assertIn("BCI腦機接口最新進展", first_prompt)
        self.assertIn("上一次選擇的主題已達探索上限", retry_prompt)
        search.assert_called_once_with("睡眠 記憶鞏固 研究", limit=3)

    @patch.object(flask_backend, "record_agent_exploration")
    @patch.object(flask_backend, "search_web", return_value=[
        {
            "title": "Scientific source",
            "url": "https://example.com/science",
            "snippet": "Research describes the mechanism and its limitations.",
        }
    ])
    @patch.object(flask_backend, "translate_to_japanese", return_value="日本語。")
    @patch.object(
        flask_backend.proactive_client.models,
        "generate_content",
        side_effect=[
            SimpleNamespace(
                text=(
                    '{"action":"start_exploration",'
                    '"interest":"時間旅行與物理學",'
                    '"topic":"時間膨脹如何被實際測量",'
                    '"query":"原子鐘 時間膨脹 實驗測量"}'
                )
            ),
            SimpleNamespace(
                text=(
                    '{"share":"我查到原子鐘實驗能測出微小的時間差，'
                    '不過測量方法本身也有精度限制。（surprised）",'
                    '"summary":"原子鐘實驗可測量相對論造成的微小時間差；'
                    '結果受儀器精度與實驗條件限制。",'
                    '"next_question":"不同高度的原子鐘如何比較時間差？"}'
                )
            ),
        ],
    )
    def test_idle_agent_starts_an_interest_exploration_and_saves_first_note(
        self,
        generate_content,
        _translate,
        search,
        record,
    ):
        with patch.object(flask_backend, "get_recent_agent_explorations", return_value=[]):
            with patch.object(flask_backend, "build_full_prompt", return_value="CONTEXT"):
                with patch.object(flask_backend, "get_conversation_history", return_value=[]):
                    with patch.object(
                        flask_backend,
                        "append_assistant_history",
                    ) as append_history:
                        with flask_backend.app.test_client() as client:
                            response = client.post(
                                "/proactive",
                                json={"user_id": "test"},
                            )

        self.assertEqual(response.status_code, 200)
        self.assertIn("原子鐘實驗", response.get_json()["reply"])
        append_history.assert_called_once_with(
            "test",
            "我查到原子鐘實驗能測出微小的時間差，"
            "不過測量方法本身也有精度限制。",
        )
        search.assert_called_once_with("原子鐘 時間膨脹 實驗測量", limit=3)
        record.assert_called_once_with(
            "test",
            "時間膨脹如何被實際測量",
            "原子鐘 時間膨脹 實驗測量",
            summary=(
                "原子鐘實驗可測量相對論造成的微小時間差；"
                "結果受儀器精度與實驗條件限制。"
            ),
            sources=[
                {
                    "title": "Scientific source",
                    "url": "https://example.com/science",
                    "snippet": "Research describes the mechanism and its limitations.",
                }
            ],
            next_question="不同高度的原子鐘如何比較時間差？",
        )
        self.assertEqual(generate_content.call_count, 2)

    @patch.object(flask_backend, "search_web")
    @patch.object(
        flask_backend.proactive_client.models,
        "generate_content",
        return_value=SimpleNamespace(
            text='{"action":"explore","topic":"火星上的水","query":"火星水研究"}'
        ),
    )
    def test_legacy_unstructured_exploration_action_is_rejected(self, _generate, search):
        with patch.object(flask_backend, "get_recent_agent_explorations", return_value=[]):
            with patch.object(flask_backend, "build_full_prompt", return_value="CONTEXT"):
                with patch.object(flask_backend, "get_conversation_history", return_value=[]):
                    with flask_backend.app.test_client() as client:
                        response = client.post("/proactive", json={"user_id": "test"})

        self.assertEqual(response.status_code, 502)
        self.assertEqual(
            response.get_json()["error"],
            "proactive_decision_unavailable",
        )
        search.assert_not_called()

if __name__ == "__main__":
    unittest.main()

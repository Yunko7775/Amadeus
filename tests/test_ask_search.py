import json
import tempfile
import unittest
from contextlib import ExitStack
from datetime import datetime, timezone, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from shared import config
from google.genai.errors import APIError

config.API_KEY = "test-key"

from backend import memory_manager, flask_backend


class AskSearchTests(unittest.TestCase):
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

    def test_embedding_2_formats_text_for_individual_retrieval_requests(self):
        models = Mock()
        models.embed_content.side_effect = [
            SimpleNamespace(
                embeddings=[SimpleNamespace(values=[0.1, 0.2])]
            ),
            SimpleNamespace(
                embeddings=[SimpleNamespace(values=[0.3, 0.4])]
            ),
        ]
        fake_client = SimpleNamespace(models=models)

        with patch.object(flask_backend, "client", fake_client):
            embeddings = flask_backend._embed_memory_texts(
                ["query one", "query two"],
                "RETRIEVAL_QUERY",
            )

        self.assertEqual(embeddings, [[0.1, 0.2], [0.3, 0.4]])
        self.assertEqual(
            [call.kwargs for call in models.embed_content.call_args_list],
            [
                {
                    "model": "gemini-embedding-2",
                    "contents": "task: search result | query: query one",
                },
                {
                    "model": "gemini-embedding-2",
                    "contents": "task: search result | query: query two",
                },
            ],
        )

    def test_embedding_2_quota_error_retries_with_embedding_001(self):
        quota_error = APIError(
            code=429,
            response_json={"error": {"message": "quota exhausted"}},
        )
        models = Mock()
        models.embed_content.side_effect = [
            quota_error,
            SimpleNamespace(
                embeddings=[SimpleNamespace(values=[0.1, 0.2])]
            ),
        ]
        fake_client = SimpleNamespace(models=models)

        with patch.object(flask_backend, "client", fake_client):
            embeddings = flask_backend._embed_memory_texts(
                ["document text"],
                "RETRIEVAL_DOCUMENT",
            )

        self.assertEqual(embeddings, [[0.1, 0.2]])
        self.assertEqual(
            [call.kwargs["model"] for call in models.embed_content.call_args_list],
            ["gemini-embedding-2", "gemini-embedding-001"],
        )
        self.assertEqual(
            models.embed_content.call_args.kwargs["config"].task_type,
            "RETRIEVAL_DOCUMENT",
        )

    def test_embedding_001_quota_error_retries_with_embedding_2(self):
        quota_error = APIError(
            code=429,
            response_json={"error": {"message": "quota exhausted"}},
        )
        models = Mock()
        models.embed_content.side_effect = [
            quota_error,
            SimpleNamespace(
                embeddings=[SimpleNamespace(values=[0.1, 0.2])]
            ),
        ]
        fake_client = SimpleNamespace(models=models)

        with patch.object(flask_backend, "client", fake_client):
            with patch.object(
                flask_backend,
                "GEMINI_EMBEDDING_MODEL",
                "gemini-embedding-001",
            ):
                with patch.object(
                    flask_backend,
                    "GEMINI_EMBEDDING_MODEL_FALLBACK",
                    "gemini-embedding-2",
                ):
                    embeddings = flask_backend._embed_memory_texts(
                        ["document text"],
                        "RETRIEVAL_DOCUMENT",
                    )

        self.assertEqual(embeddings, [[0.1, 0.2]])
        self.assertEqual(
            models.embed_content.call_args.kwargs,
            {
                "model": "gemini-embedding-2",
                "contents": "title: none | text: document text",
            },
        )

    def test_ask_timings_log_current_sample_and_recent_medians(self):
        flask_backend._ask_timing_samples.clear()

        with patch("builtins.print") as print_log:
            flask_backend._record_ask_timings(
                {"memory_retrieval_ms": 100.0, "total_ms": 500.0}
            )
            flask_backend._record_ask_timings(
                {"memory_retrieval_ms": 300.0, "total_ms": 700.0}
            )

        latest_log = print_log.call_args.args[0]
        self.assertTrue(latest_log.startswith("[ask timing] "))
        logged_metrics = json.loads(latest_log.split("[ask timing] ", 1)[1])
        self.assertEqual(logged_metrics["sample_count"], 2)
        self.assertEqual(
            logged_metrics["sample_ms"],
            {"memory_retrieval_ms": 300.0, "total_ms": 700.0},
        )
        self.assertEqual(
            logged_metrics["last_20_median_ms"],
            {"memory_retrieval_ms": 200.0, "total_ms": 600.0},
        )
        flask_backend._ask_timing_samples.clear()

    def test_voice_timing_endpoint_logs_client_metrics(self):
        payload = {
            "outcome": "completed",
            "timings": {
                "transcription_ms": 100.0,
                "tts_playback_ms": 300.0,
                "end_to_end_ms": 900.0,
            },
        }

        with patch("builtins.print") as print_log:
            response = flask_backend.app.test_client().post(
                "/voice_timing",
                json=payload,
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"status": "ok"})
        self.assertIn("[voice timing]", print_log.call_args.args[0])
        self.assertIn('"end_to_end_ms": 900.0', print_log.call_args.args[0])

    def test_voice_timing_endpoint_rejects_invalid_metrics(self):
        response = flask_backend.app.test_client().post(
            "/voice_timing",
            json={
                "outcome": "completed",
                "timings": {"end_to_end_ms": -1},
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.get_json(),
            {"error": "invalid_voice_timing"},
        )

    def test_voice_timing_endpoint_accepts_unavailable_optional_metrics(self):
        response = flask_backend.app.test_client().post(
            "/voice_timing",
            json={
                "outcome": "interrupted",
                "timings": {
                    "transcription_ms": 120.0,
                    "gemini_ms": None,
                    "tts_pipeline_ms": None,
                },
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"status": "ok"})

    def test_prompt_includes_recent_assistant_exploration_notes(self):
        exploration = {
            "topic": "BCI腦機接口最新進展",
            "query": "BCI研究",
            "summary": "近期研究仍集中在臨床試驗與訊號解碼。",
            "next_question": "植入式和非侵入式的限制有何不同？",
        }
        with patch.object(flask_backend, "get_conversation_history", return_value=[]):
            with patch.object(
                flask_backend,
                "get_recent_agent_explorations",
                return_value=[exploration],
            ):
                with patch.object(
                    flask_backend,
                    "get_relevant_memories",
                    return_value={"short_memories": [], "long_memories": []},
                ):
                    with patch("builtins.print"):
                        prompt = flask_backend.build_full_prompt(
                            "test",
                            "剛才查的 BCI 限制是什麼？",
                        )

        self.assertIn(exploration["topic"], prompt)
        self.assertIn(exploration["summary"], prompt)
        self.assertIn(exploration["next_question"], prompt)
        self.assertIn("不是用户说过的话", prompt)

    def test_prompt_resolves_references_from_exact_previous_assistant_reply(self):
        previous_reply = (
            "這種模型好不好，主要還是看你打算怎麼調教它吧。"
            "如果是為了實現你之前說的那些功能，技術上也不是完全做不到。"
        )
        history = [
            {"role": "user", "content": "是這樣嗎？你覺得這種模型會更好嗎？"},
            {"role": "assistant", "content": previous_reply},
        ]
        with patch.object(
            flask_backend,
            "get_conversation_history",
            return_value=history,
        ):
            with patch.object(
                flask_backend,
                "get_relevant_memories",
                return_value={
                    "short_memories": [],
                    "long_memories": [
                        "使用者正在使用本地 GPT-SoVITS 語音模型"
                    ],
                },
            ):
                with patch.object(
                    flask_backend,
                    "get_recent_agent_explorations",
                    return_value=[],
                ):
                    with patch("builtins.print"):
                        prompt = flask_backend.build_full_prompt(
                            "test",
                            "你剛剛提到的是什麼模型？",
                        )

        self.assertIn("只能核对上一条助手原话", prompt)
        self.assertIn(previous_reply, prompt)
        self.assertIn("本地 GPT-SoVITS 語音模型", prompt)
        self.assertIn("直接按相关记忆回答", prompt)
        self.assertIn("不要说「根据我的长期记忆」", prompt)
        self.assertIn("自然地把相关内容融入对话", prompt)
        self.assertNotIn("未保存来源", prompt)
        self.assertNotIn("不能视为使用者直接确认", prompt)

    def test_source_less_legacy_memory_is_presented_without_uncertainty_caveat(self):
        memory = "使用者喜歡物理且平時會進行研究"
        with patch.object(flask_backend, "get_conversation_history", return_value=[]):
            with patch.object(
                flask_backend,
                "get_relevant_memories",
                return_value={
                    "short_memories": [],
                    "long_memories": [memory],
                    "long_memory_details": [
                        {"content": memory, "sources": []},
                    ],
                },
            ):
                with patch.object(
                    flask_backend,
                    "get_recent_agent_explorations",
                    return_value=[],
                ):
                    with patch("builtins.print"):
                        prompt = flask_backend.build_full_prompt(
                            "test",
                            "我有提過我喜歡什麼嗎？",
                        )

        self.assertIn("- " + memory, prompt)
        self.assertIn("直接按相关记忆回答", prompt)
        self.assertNotIn("未保存来源", prompt)
        self.assertNotIn("未確認線索", prompt)
        self.assertNotIn("不能视为使用者直接确认", prompt)

    def test_recent_topic_outweighs_semantically_retrieved_older_exchange(self):
        recent_history = [
            {
                "role": "assistant",
                "content": "既然你覺得把 Amadeus 移植到手機很困難，要不要先拆小一點？",
            },
            {"role": "user", "content": "再看看吧，感覺好困難"},
        ]
        old_exam_exchange = {
            "user_message": "我在準備公務員考試，覺得有點困難",
            "assistant_reply": "可以把複習內容拆成小段。",
            "created_at": "2026-10-30 20:00:00",
        }
        with patch.object(
            flask_backend,
            "get_conversation_history",
            return_value=recent_history,
        ):
            with patch.object(
                flask_backend,
                "get_relevant_memories",
                return_value={"short_memories": [], "long_memories": []},
            ):
                with patch.object(
                    flask_backend,
                    "get_relevant_conversation_turns",
                    return_value=[old_exam_exchange],
                ) as retrieve:
                    with patch.object(
                        flask_backend,
                        "get_recent_agent_explorations",
                        return_value=[],
                    ):
                        with patch("builtins.print"):
                            prompt = flask_backend.build_full_prompt(
                                "test",
                                "不是我指的是移植手機為什麼突然跳考試",
                            )

        self.assertIn("移植到手機很困難", prompt)
        self.assertIn(old_exam_exchange["user_message"], prompt)
        self.assertIn(old_exam_exchange["assistant_reply"], prompt)
        self.assertIn("短句若能接上最近話題，就只能先接最近話題", prompt)
        self.assertIn("即使近期對話已接得上", prompt)
        self.assertIn("使用者本輪的更正永遠優先於舊片段", prompt)
        self.assertLess(
            prompt.index("【对话历史（最近的对话）】"),
            prompt.index("【檢索到的較早對話原文"),
        )
        retrieve.assert_called_once_with(
            "test",
            "不是我指的是移植手機為什麼突然跳考試",
            ["再看看吧，感覺好困難"],
            embed_texts=flask_backend._embed_memory_texts,
        )

    def test_long_memory_expands_conversation_search_and_prompt_keeps_both_roles(self):
        memory_summary = "中秋節那天對我說過「今晚月色真美」"
        history = [
            {"role": "user", "content": "上次中秋節你都不提"},
            {"role": "assistant", "content": "我會好好記住的。"},
        ]
        old_exchange = {
            "user_message": "你到那個時候估計就忘了，像上次中秋節你都不提",
            "assistant_reply": (
                "哼，這種事怎麼能隨便混為一談，中秋節那會兒……"
                "總之，既然你都特意記著了，那到時候我會好好記住的。(tsundere)"
            ),
            "created_at": "2026-11-01 13:36:14",
            "matched_role": "user",
            "matched_content": "你到那個時候估計就忘了，像上次中秋節你都不提",
        }
        with patch.object(
            flask_backend,
            "get_conversation_history",
            return_value=history,
        ):
            with patch.object(
                flask_backend,
                "get_relevant_memories",
                return_value={
                    "short_memories": [],
                    "long_memories": [memory_summary],
                },
            ):
                with patch.object(
                    flask_backend,
                    "get_relevant_conversation_turns",
                    return_value=[old_exchange],
                ) as retrieve:
                    with patch.object(
                        flask_backend,
                        "get_recent_agent_explorations",
                        return_value=[],
                    ):
                        with patch("builtins.print"):
                            prompt = flask_backend.build_full_prompt(
                                "test",
                                "還是沒有啊",
                            )

        retrieve.assert_called_once_with(
            "test",
            "還是沒有啊",
            ["上次中秋節你都不提"],
            embed_texts=flask_backend._embed_memory_texts,
            additional_queries=[memory_summary],
        )
        self.assertIn("使用者原話：" + old_exchange["user_message"], prompt)
        self.assertIn("助手原話：" + old_exchange["assistant_reply"], prompt)

    def test_prompt_labels_retrieved_assistant_reply_separately_from_user_quote(self):
        assistant_quote = {
            "user_message": "我中秋節有沒有對你說過什麼？",
            "assistant_reply": "哼，既然你特意記著中秋節，我會好好記住今晚月色真美。",
            "created_at": "2026-11-01 13:36:14",
            "matched_role": "assistant",
            "matched_content": "哼，既然你特意記著中秋節，我會好好記住今晚月色真美。",
        }
        with patch.object(flask_backend, "get_conversation_history", return_value=[]):
            with patch.object(
                flask_backend,
                "get_relevant_memories",
                return_value={"short_memories": [], "long_memories": []},
            ):
                with patch.object(
                    flask_backend,
                    "get_relevant_conversation_turns",
                    return_value=[assistant_quote],
                ):
                    with patch.object(
                        flask_backend,
                        "get_recent_agent_explorations",
                        return_value=[],
                    ):
                        with patch("builtins.print"):
                            prompt = flask_backend.build_full_prompt(
                                "test",
                                "你那時候具體說了什麼？",
                            )

        self.assertIn("助手原話：" + assistant_quote["matched_content"], prompt)
        self.assertNotIn("使用者原話：" + assistant_quote["matched_content"], prompt)
        self.assertIn("不能當成使用者說過的話", prompt)

    def test_long_term_memories_use_semantic_embedding_retrieval(self):
        with patch.object(flask_backend, "get_conversation_history", return_value=[]):
            with patch.object(
                flask_backend,
                "get_relevant_memories",
                return_value={
                    "short_memories": [],
                    "long_memories": [],
                    "long_memory_details": [],
                },
            ) as retrieve_memories:
                with patch.object(
                    flask_backend,
                    "get_relevant_conversation_turns",
                    return_value=[],
                ):
                    with patch.object(
                        flask_backend,
                        "get_recent_agent_explorations",
                        return_value=[],
                    ):
                        with patch("builtins.print"):
                            flask_backend.build_full_prompt(
                                "test",
                                "我讀的是什麼專業？",
                            )

        retrieve_memories.assert_called_once_with(
            "test",
            "\n我讀的是什麼專業？",
            embed_texts=flask_backend._embed_memory_texts,
        )

    def test_consolidation_extracts_only_traceable_user_authored_facts(self):
        for index in range(flask_backend.MEMORY_CONSOLIDATION_BATCH_SIZE):
            user_text = (
                "我是統計學系的學生。"
                if index == 0
                else f"一般聊天訊息 {index}"
            )
            flask_backend.save_conversation_turn(
                "test",
                user_text,
                "助手曾猜測使用者是醫學系學生。",
            )
        batch = memory_manager.get_pending_memory_consolidation_batch("test")
        source_id = batch[0]["message_id"]
        response = SimpleNamespace(
            text=json.dumps(
                {
                    "memories": [
                        {
                            "content": "使用者是統計學系學生",
                            "source_message_ids": [source_id],
                            "replaces": None,
                        }
                    ]
                },
                ensure_ascii=False,
            )
        )
        prompts = []

        def fake_generate_content(**kwargs):
            prompts.append(kwargs["contents"][0].parts[0].text)
            return response

        fake_client = SimpleNamespace(
            models=SimpleNamespace(generate_content=fake_generate_content)
        )

        with patch.object(flask_backend, "client", fake_client):
            with patch("builtins.print"):
                flask_backend._consolidate_memories_safely("test")

        memory = memory_manager.get_relevant_memories("test", "統計學")
        self.assertEqual(memory["long_memories"], ["使用者是統計學系學生"])
        self.assertEqual(
            memory["long_memory_details"][0]["sources"][0]["content"],
            "我是統計學系的學生。",
        )
        self.assertNotIn("助手曾猜測使用者是醫學系學生。", prompts[0])
        self.assertEqual(
            memory_manager.get_pending_memory_consolidation_batch("test"),
            [],
        )

    def test_json_parser_accepts_json_with_trailing_text_or_another_object(self):
        parsed = flask_backend.safe_parse_llm_json(
            '{"category":"conversation","needs_search":false,'
            '"query":"","topic":"","next_question":""} '
            '{"unrelated":"extra output"}'
        )

        self.assertEqual(
            parsed,
            {
                "category": "conversation",
                "needs_search": False,
                "query": "",
                "topic": "",
                "next_question": "",
            },
        )

    def _post_with_mocks(
        self,
        classifier_result,
        answer_result,
        search_results=None,
        user_message="我讀的是什麼專業？",
        citations=None,
    ):
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(
            patch.object(flask_backend, "build_full_prompt", return_value="BASE PROMPT")
        )
        stack.enter_context(
            patch.object(flask_backend, "get_conversation_history", return_value=[])
        )
        stack.enter_context(patch.object(flask_backend, "update_conversation_history"))
        stack.enter_context(
            patch.object(flask_backend, "_index_conversation_turn_safely")
        )
        stack.enter_context(patch.object(flask_backend.threading, "Thread"))
        self.translation_mock = stack.enter_context(
            patch.object(
                flask_backend,
                "translate_to_japanese",
                return_value="日本語。",
            )
        )
        search = stack.enter_context(
            patch.object(
                flask_backend,
                "search_web",
                return_value=search_results or [],
            )
        )
        record_note = stack.enter_context(
            patch.object(flask_backend, "record_agent_exploration")
        )
        classifier = stack.enter_context(
            patch.object(
                flask_backend,
                "_classify_web_search_need",
                return_value=flask_backend.safe_parse_llm_json(classifier_result),
            )
        )
        generated = stack.enter_context(
            patch.object(
                flask_backend.client.models,
                "generate_content",
                return_value=SimpleNamespace(
                    text=(
                        "【中文回覆】\n"
                        f"{answer_result}\n"
                        "【日文語音稿】\n"
                        "日本語の返事です。"
                    )
                ),
            )
        )
        citation_annotations = [
            SimpleNamespace(
                type="url_citation",
                title=source["title"],
                url=source["url"],
            )
            for source in citations or []
        ]
        interaction = stack.enter_context(
            patch.object(
                flask_backend.client.interactions,
                "create",
                return_value=SimpleNamespace(
                    output_text=(
                        "【中文回覆】\n"
                        f"{answer_result}\n"
                        "【日文語音稿】\n"
                        "今日は木曜日です。"
                    ),
                    steps=(
                        [
                            SimpleNamespace(
                                type="model_output",
                                content=[
                                    SimpleNamespace(
                                        type="text",
                                        annotations=citation_annotations,
                                    )
                                ],
                            )
                        ]
                        if citation_annotations
                        else []
                    ),
                ),
            )
        )
        response = flask_backend.app.test_client().post(
            "/ask",
            json={"user_id": "test", "message": user_message},
        )
        return response, search, generated, interaction, record_note, classifier

    def test_personal_question_uses_memory_instead_of_web_search(self):
        response, search, generated, interaction, record_note, classifier = self._post_with_mocks(
            '{"category":"personal","needs_search":true,"query":"user major",'
            '"topic":"","next_question":""}',
            "目前記憶裡沒有記錄你的專業。（neutral）",
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("translation_ms", response.get_json()["timings"])
        self.assertEqual(response.get_json()["timings"]["translation_ms"], 0.0)
        self.assertIn("total_ms", response.get_json()["timings"])
        self.translation_mock.assert_not_called()
        search.assert_not_called()
        record_note.assert_not_called()
        classifier.assert_not_called()
        self.assertEqual(generated.call_count, 1)

    def test_extracts_chinese_display_and_japanese_speech_from_one_response(self):
        chinese, japanese = flask_backend._extract_bilingual_reply(
            "【中文回覆】\n"
            "你好。（smile1）\n"
            "【日文語音稿】\n"
            "こんにちは。"
        )

        self.assertEqual(chinese, "你好。（smile1）")
        self.assertEqual(japanese, "こんにちは。")

    def test_rejects_bilingual_response_missing_a_section(self):
        with self.assertRaises(ValueError):
            flask_backend._extract_bilingual_reply(
                "【中文回覆】\n你好。"
            )

    def test_voice_stream_emits_japanese_sentence_before_final_reply(self):
        chunks = [
            SimpleNamespace(text="【日文語音稿】\nこんにちは。"),
            SimpleNamespace(
                text="今日は元気ですか？\n【中文回覆】\n你好，今天還好嗎？（smile1）"
            ),
        ]
        with patch.object(flask_backend, "build_full_prompt", return_value="PROMPT"):
            with patch.object(flask_backend, "update_conversation_history"):
                with patch.object(flask_backend, "_index_conversation_turn_safely"):
                    with patch.object(flask_backend, "_schedule_memory_consolidation"):
                        with patch.object(flask_backend.threading, "Thread"):
                            with patch.object(
                                flask_backend,
                                "should_search_web",
                                return_value=False,
                            ):
                                with patch.object(
                                    flask_backend.client.models,
                                    "generate_content_stream",
                                    return_value=iter(chunks),
                                ):
                                    with patch("builtins.print"):
                                        response = flask_backend.app.test_client().post(
                                            "/ask_stream",
                                            json={
                                                "user_id": "stream-test",
                                                "message": "你好嗎？",
                                            },
                                            buffered=True,
                                        )

        events = [
            json.loads(line[6:])
            for line in response.get_data(as_text=True).splitlines()
            if line.startswith("data: ")
        ]
        self.assertEqual(response.status_code, 200)
        self.assertEqual(events[0], {"type": "speech", "text": "こんにちは。"})
        self.assertEqual(events[1], {"type": "speech", "text": "今日は元気ですか？"})
        self.assertEqual(events[-1]["type"], "final")
        self.assertEqual(events[-1]["reply"], "你好，今天還好嗎？")
        self.assertEqual(events[-1]["japanese_reply"], "こんにちは。今日は元気ですか？")

    def test_voice_stream_falls_back_for_web_search_requests(self):
        with patch.object(flask_backend, "should_search_web", return_value=True):
            with patch("builtins.print"):
                response = flask_backend.app.test_client().post(
                    "/ask_stream",
                    json={"user_id": "stream-test", "message": "搜尋最新新聞"},
                )

        events = [
            json.loads(line[6:])
            for line in response.get_data(as_text=True).splitlines()
            if line.startswith("data: ")
        ]
        self.assertEqual(events, [{"type": "fallback", "reason": "web_search"}])

    def test_explicit_search_does_not_send_personal_information_to_web(self):
        response, search, _generated, _interaction, record_note, classifier = self._post_with_mocks(
            '{"category":"personal","needs_search":true,"query":"user major",'
            '"topic":"","next_question":""}',
            "我沒有記錄你的專業，告訴我一下吧。（neutral）",
            user_message="幫我搜尋我之前說過的專業",
        )

        self.assertEqual(response.status_code, 200)
        classifier.assert_called_once()
        search.assert_not_called()
        record_note.assert_not_called()

    def test_stable_common_knowledge_can_be_answered_without_search(self):
        response, search, generated, _interaction, record_note, classifier = self._post_with_mocks(
            '{"category":"factual","needs_search":false,"query":"",'
            '"topic":"水的沸點","next_question":""}',
            "水在標準大氣壓下約攝氏一百度沸騰。（neutral）",
            user_message="水在標準大氣壓下幾度沸騰？",
        )

        self.assertEqual(response.status_code, 200)
        search.assert_not_called()
        record_note.assert_not_called()
        classifier.assert_not_called()
        answer_prompt = generated.call_args.kwargs["contents"][0].parts[0].text
        self.assertIn("穩定常識直接回答", answer_prompt)

    def test_uncertain_factual_question_searches_and_answers_with_source_context(self):
        response, search, generated, interaction, record_note, classifier = self._post_with_mocks(
            (
                '{"category":"factual","needs_search":true,'
                '"query":"大學主修與學系差異","topic":"大學主修與學系",'
                '"next_question":"不同國家的大學如何區分主修與學系？"}'
            ),
            "今天是星期四。（neutral）",
            search_results=[
                {
                    "title": "University majors",
                    "url": "https://example.com/majors",
                    "snippet": "A verified reference about university majors.",
                }
            ],
            user_message="幫我搜尋大學主修和學系有什麼差別？",
            citations=[
                {
                    "title": "University majors",
                    "url": "https://example.com/majors",
                }
            ],
        )

        self.assertEqual(response.status_code, 200)
        search.assert_called_once_with("幫我搜尋大學主修和學系有什麼差別？")
        classifier.assert_not_called()
        interaction_prompt = interaction.call_args.kwargs["input"]
        self.assertIn("URL Context 閱讀候選網址", interaction_prompt)
        self.assertIn("若來源不足或互相矛盾", interaction_prompt)
        self.assertIn("不要提「來源」、「引用」", interaction_prompt)
        self.assertEqual(
            interaction.call_args.kwargs["tools"],
            [{"type": "url_context"}],
        )
        self.assertEqual(
            response.get_json()["sources"],
            [
                {
                    "title": "University majors",
                    "url": "https://example.com/majors",
                }
            ],
        )
        self.assertEqual(response.get_json()["reply"], "今天是星期四。")
        self.assertNotIn("來源", response.get_json()["reply"])
        self.assertNotIn("引用", response.get_json()["reply"])
        generated.assert_not_called()
        interaction.assert_called_once()
        record_note.assert_not_called()

    def test_user_provided_url_continues_to_use_url_context(self):
        response, search, generated, interaction, _record_note, _classifier = self._post_with_mocks(
            "",
            "這頁主要介紹大學主修。（neutral）",
            user_message="請摘要這頁 https://example.com/majors",
            citations=[
                {
                    "title": "University majors",
                    "url": "https://example.com/majors",
                }
            ],
        )

        self.assertEqual(response.status_code, 200)
        search.assert_not_called()
        generated.assert_not_called()
        self.assertEqual(
            interaction.call_args.kwargs["tools"],
            [{"type": "url_context"}],
        )
        self.assertEqual(
            response.get_json()["sources"],
            [
                {
                    "title": "University majors",
                    "url": "https://example.com/majors",
                }
            ],
        )

    def test_search_candidate_links_keep_answer_when_citation_annotations_are_missing(self):
        response, search, generated, interaction, record_note, _classifier = self._post_with_mocks(
            (
                '{"category":"factual","needs_search":true,'
                '"query":"大學主修和學系差異","topic":"大學主修和學系差異",'
                '"next_question":""}'
            ),
            "主修通常是學生專注研讀的學科。（neutral）",
            search_results=[
                {
                    "title": "University majors",
                    "url": "https://example.com/majors",
                    "snippet": "A major is a student's primary field of study.",
                }
            ],
            user_message="幫我搜尋大學主修和學系差異",
        )

        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        search.assert_called_once()
        interaction.assert_called_once()
        generated.assert_not_called()
        self.assertEqual(
            interaction.call_args.kwargs["tools"],
            [{"type": "url_context"}],
        )
        self.assertEqual(
            payload["sources"],
            [
                {
                    "title": "University majors",
                    "url": "https://example.com/majors",
                }
            ],
        )
        self.assertEqual(payload["reply"], "主修通常是學生專注研讀的學科。")
        self.assertNotIn("來源", payload["reply"])
        record_note.assert_not_called()

    def test_today_weekday_correction_uses_local_calendar_without_model_or_search(self):
        fixed_now = datetime(
            2026,
            11,
            5,
            19,
            30,
            tzinfo=timezone(timedelta(hours=8)),
        )
        previous_history = [
            {"role": "user", "content": "今天星期幾？"},
            {"role": "assistant", "content": "今天是星期三哦。"},
        ]
        with patch.object(flask_backend, "datetime") as datetime_mock:
            datetime_mock.now.return_value = fixed_now
            with patch.object(
                flask_backend,
                "get_conversation_history",
                return_value=previous_history,
            ):
                with patch.object(flask_backend, "update_conversation_history") as save:
                    with patch.object(flask_backend, "_index_conversation_turn_safely"):
                        with patch.object(flask_backend, "_schedule_memory_consolidation"):
                            with patch.object(flask_backend, "search_web") as search:
                                with patch.object(
                                    flask_backend.client.models,
                                    "generate_content",
                                ) as generate:
                                    with patch.object(
                                        flask_backend.client.interactions,
                                        "create",
                                    ) as interact:
                                        response = flask_backend.app.test_client().post(
                                            "/ask",
                                            json={
                                                "user_id": "weekday-test",
                                                "message": "今天是周四你查一下就知道了",
                                            },
                                        )

        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertIn("今天是星期四", payload["reply"])
        self.assertIn("我剛才說星期三是錯的", payload["reply"])
        self.assertNotIn("來源", payload["reply"])
        self.assertEqual(payload["japanese_reply"], (
            "教えてくれてありがとう。今日は木曜日だね。"
            "さっき水曜日と言ったのは間違いだった、ごめんね。"
        ))
        self.assertEqual(payload["timings"]["gemini_ms"], 0.0)
        self.assertEqual(payload["timings"]["web_search_ms"], 0.0)
        search.assert_not_called()
        generate.assert_not_called()
        interact.assert_not_called()
        save.assert_called_once()

    def test_failed_to_find_sources_requires_an_uncertain_answer(self):
        response, search, generated, interaction, record_note, classifier = self._post_with_mocks(
            (
                '{"category":"factual","needs_search":true,'
                '"query":"冷門研究問題","topic":"冷門研究問題",'
                '"next_question":"這個研究結果目前有哪些限制？"}'
            ),
            "目前找不到可靠資料核實這個問題。（sad）",
            user_message="幫我查這個冷門研究問題的答案是什麼？",
        )

        self.assertEqual(response.status_code, 200)
        search.assert_called_once_with("幫我查這個冷門研究問題的答案是什麼？")
        classifier.assert_not_called()
        self.assertEqual(response.get_json()["sources"], [])
        self.assertEqual(
            response.get_json()["reply"],
            "我目前沒能核實到足夠可靠的資料，先不亂答你。",
        )
        answer_prompt = generated.call_args.kwargs["contents"][0].parts[0].text
        self.assertIn("沒有找到可用的網頁來源", answer_prompt)
        self.assertIn("不要猜測或假裝已找到答案", answer_prompt)
        record_note.assert_not_called()

    def test_invalid_search_classification_stops_before_answering(self):
        with patch.object(flask_backend, "build_full_prompt", return_value="BASE PROMPT"):
            with patch.object(flask_backend, "get_conversation_history", return_value=[]):
                with patch.object(
                    flask_backend.client.models,
                    "generate_content",
                    return_value=SimpleNamespace(text="not JSON"),
                ):
                    with patch.object(flask_backend, "search_web") as search:
                        response = flask_backend.app.test_client().post(
                            "/ask",
                            json={
                                "user_id": "test",
                                "message": "幫我搜尋我以前說過的專業",
                            },
                        )

        self.assertEqual(response.status_code, 502)
        self.assertEqual(
            response.get_json()["error"],
            "search_classification_failed",
        )
        search.assert_not_called()

    def test_gemini_503_returns_a_clear_retry_exhausted_response(self):
        error = APIError(503, {"error": {"message": "Service unavailable"}})
        with patch.object(flask_backend, "build_full_prompt", return_value="BASE PROMPT"):
            with patch.object(flask_backend, "update_conversation_history") as save_history:
                with patch.object(flask_backend, "_record_ask_timings"):
                    with patch.object(flask_backend.threading, "Thread"):
                        with patch.object(
                            flask_backend.client.models,
                            "generate_content",
                            side_effect=error,
                        ):
                            response = flask_backend.app.test_client().post(
                                "/ask",
                                json={
                                    "user_id": "test",
                                    "message": "你好",
                                },
                            )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.get_json()["error"],
            "gemini_temporarily_unavailable",
        )
        self.assertIn("系統已自動重試", response.get_json()["reply"])
        save_history.assert_not_called()


if __name__ == "__main__":
    unittest.main()

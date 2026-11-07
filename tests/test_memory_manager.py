import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from backend import memory_manager


class MemoryManagerTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)
        self.database_path = self.data_dir / "memory.sqlite3"
        self.legacy_file = self.data_dir / "user_memory.json"
        self.database_patch = patch.object(
            memory_manager, "MEMORY_DB_PATH", self.database_path
        )
        self.legacy_patch = patch.object(
            memory_manager, "LEGACY_MEMORY_FILE", self.legacy_file
        )
        self.database_patch.start()
        self.legacy_patch.start()

    def tearDown(self):
        self.database_patch.stop()
        self.legacy_patch.stop()
        self.temp_dir.cleanup()

    def test_short_memory_replaces_previous_value_and_long_memory_deduplicates(self):
        memory_manager.add_memory("user", "SHORT", "temporary one")
        memory_manager.add_memory("user", "SHORT", "temporary two")
        memory_manager.add_memory("user", "LONG", "favorite color is blue")
        memory_manager.add_memory("user", "LONG", "favorite color is blue")

        self.assertEqual(
            memory_manager.get_user_memory("user"),
            {
                "profile": {},
                "long_memories": ["favorite color is blue"],
                "short_memories": ["temporary two"],
            },
        )

    def test_application_state_persists_and_updates(self):
        self.assertIsNone(
            memory_manager.get_application_state("daily_greeting_date")
        )

        memory_manager.set_application_state("daily_greeting_date", "2026-11-04")
        memory_manager.set_application_state("daily_greeting_date", "2026-11-05")

        self.assertEqual(
            memory_manager.get_application_state("daily_greeting_date"),
            "2026-11-05",
        )

    def test_migrates_legacy_json_without_deleting_it(self):
        self.legacy_file.write_text(
            json.dumps(
                {
                    "user": {
                        "profile": {"name": "Amadeus"},
                        "long_memories": ["likes science"],
                        "short_memories": ["today's note"],
                    }
                }
            ),
            encoding="utf-8",
        )

        migrated = memory_manager.get_user_memory("user")

        self.assertEqual(migrated["profile"], {"name": "Amadeus"})
        self.assertEqual(migrated["long_memories"], ["likes science"])
        self.assertEqual(migrated["short_memories"], ["today's note"])
        self.assertTrue(self.legacy_file.exists())

    def test_clears_short_memories_for_every_user(self):
        memory_manager.add_memory("one", "SHORT", "temporary")
        memory_manager.add_memory("one", "LONG", "persistent")
        memory_manager.add_memory("two", "SHORT", "temporary")

        memory_manager.clear_short_memory_on_start()

        self.assertEqual(memory_manager.get_user_memory("one")["short_memories"], [])
        self.assertEqual(
            memory_manager.get_user_memory("one")["long_memories"], ["persistent"]
        )
        self.assertEqual(memory_manager.get_user_memory("two")["short_memories"], [])

    def test_selects_memories_relevant_to_the_current_question(self):
        memory_manager.add_memory("user", "LONG", "我喜欢古典音乐")
        memory_manager.add_memory("user", "LONG", "我喜欢摇滚音乐")
        memory_manager.add_memory("user", "LONG", "我养了一只猫")

        with closing(sqlite3.connect(self.database_path)) as connection:
            with connection:
                connection.execute(
                    """
                    UPDATE memories SET last_used_at = '2020-01-01 00:00:00'
                    WHERE content = '我喜欢古典音乐'
                    """
                )
                connection.execute(
                    """
                    UPDATE memories SET last_used_at = '2026-10-31 00:00:00'
                    WHERE content = '我喜欢摇滚音乐'
                    """
                )

        selected = memory_manager.get_relevant_memories(
            "user",
            "你喜欢什么音乐？",
            limit=1,
        )

        self.assertEqual(selected["long_memories"], ["我喜欢摇滚音乐"])
        self.assertEqual(selected["short_memories"], [])

    def test_semantic_search_finds_memories_without_shared_keywords(self):
        memory_manager.add_memory(
            "user",
            "LONG",
            "使用者是物理系出身，後來轉向人工智慧研究",
        )
        memory_manager.add_memory(
            "user",
            "LONG",
            "使用者喜歡收集動畫周邊",
        )
        requests = []

        def fake_embed_texts(texts, task_type):
            requests.append((list(texts), task_type))
            if task_type == "RETRIEVAL_QUERY":
                return [[1.0, 0.0]]
            return [
                [1.0, 0.0] if "物理系" in text else [0.0, 1.0]
                for text in texts
            ]

        selected = memory_manager.get_relevant_memories(
            "user",
            "我讀的是什麼專業？",
            embed_texts=fake_embed_texts,
        )

        self.assertEqual(
            selected["long_memories"],
            ["使用者是物理系出身，後來轉向人工智慧研究"],
        )
        self.assertEqual(
            [task_type for _, task_type in requests],
            ["RETRIEVAL_QUERY", "RETRIEVAL_DOCUMENT"],
        )

        memory_manager.get_relevant_memories(
            "user",
            "我以前是學什麼的？",
            embed_texts=fake_embed_texts,
        )
        self.assertEqual(len(requests), 3)
        self.assertEqual(requests[-1][1], "RETRIEVAL_QUERY")

    def test_semantic_search_reindexes_embeddings_with_old_dimensions(self):
        memory_manager.add_memory(
            "user",
            "LONG",
            "使用者是物理系出身",
        )
        with closing(sqlite3.connect(self.database_path)) as connection:
            with connection:
                connection.execute(
                    "UPDATE memories SET embedding = ? WHERE content = ?",
                    (json.dumps([1.0, 0.0]), "使用者是物理系出身"),
                )

        requests = []

        def fake_embed_texts(texts, task_type):
            requests.append((list(texts), task_type))
            return [[1.0, 0.0, 0.0] for _ in texts]

        selected = memory_manager.get_relevant_memories(
            "user",
            "我讀的是什麼專業？",
            embed_texts=fake_embed_texts,
        )

        self.assertEqual(selected["long_memories"], ["使用者是物理系出身"])
        self.assertEqual(
            requests,
            [
                (["我讀的是什麼專業？"], "RETRIEVAL_QUERY"),
                (["使用者是物理系出身"], "RETRIEVAL_DOCUMENT"),
            ],
        )

    def test_persists_conversation_messages_with_roles_and_timestamps(self):
        turn_id = memory_manager.save_conversation_turn(
            "user",
            "把 Amadeus 移植到手機感覺很困難",
            "可以先從簡化手機介面開始。",
        )

        messages = memory_manager.get_recent_conversation_messages("user")

        self.assertEqual([message["role"] for message in messages], ["user", "assistant"])
        self.assertEqual(
            [message["content"] for message in messages],
            [
                "把 Amadeus 移植到手機感覺很困難",
                "可以先從簡化手機介面開始。",
            ],
        )
        self.assertEqual({message["turn_id"] for message in messages}, {turn_id})
        self.assertEqual({message["source"] for message in messages}, {"chat"})
        self.assertTrue(all(message["created_at"] for message in messages))

    def test_conversation_retrieval_returns_original_exchange_and_skips_recent_turns(self):
        old_turn = memory_manager.save_conversation_turn(
            "user",
            "我想把 Amadeus 移植到手機",
            "可以先評估手機介面的最小改動。",
        )
        unrelated_turn = memory_manager.save_conversation_turn(
            "user",
            "我在準備公務員考試",
            "可以把複習內容拆成小段。",
        )
        current_turn = memory_manager.save_conversation_turn(
            "user",
            "現在這個方案有點困難",
            "先拆分問題再逐步處理。",
        )
        memory_manager.index_conversation_turn(
            "user",
            old_turn,
            lambda texts, _task: [[1.0, 0.0] for _ in texts],
        )
        memory_manager.index_conversation_turn(
            "user",
            unrelated_turn,
            lambda texts, _task: [[0.0, 1.0] for _ in texts],
        )
        memory_manager.index_conversation_turn(
            "user",
            current_turn,
            lambda texts, _task: [[0.8, 0.2] for _ in texts],
        )

        retrieved = memory_manager.get_relevant_conversation_turns(
            "user",
            "移植到手機怎麼做",
            recent_user_messages=["現在這個方案有點困難"],
            embed_texts=lambda _texts, task: (
                [[1.0, 0.0]] if task == "RETRIEVAL_QUERY" else []
            ),
        )

        self.assertEqual(len(retrieved), 1)
        self.assertEqual(
            retrieved[0]["user_message"],
            "我想把 Amadeus 移植到手機",
        )
        self.assertEqual(
            retrieved[0]["assistant_reply"],
            "可以先評估手機介面的最小改動。",
        )
        self.assertTrue(retrieved[0]["created_at"])

    def test_conversation_retrieval_matches_assistant_reply_embeddings(self):
        turn_id = memory_manager.save_conversation_turn(
            "user",
            "那天聊完了就先休息。",
            "哼，既然你特意記著中秋節，我會好好記住今晚月色真美。",
        )

        def fake_embed_texts(texts, task_type):
            if task_type == "RETRIEVAL_QUERY":
                return [
                    [0.0, 1.0] if "中秋節那句話" not in text else [1.0, 0.0]
                    for text in texts
                ]
            return [
                [1.0, 0.0] if "今晚月色真美" in text else [0.0, 1.0]
                for text in texts
            ]

        memory_manager.index_conversation_turn(
            "user",
            turn_id,
            fake_embed_texts,
        )

        retrieved = memory_manager.get_relevant_conversation_turns(
            "user",
            "現在在聊別的事情",
            recent_user_messages=[],
            embed_texts=fake_embed_texts,
            additional_queries=["中秋節那句話是什麼？"],
        )

        self.assertEqual(len(retrieved), 1)
        self.assertEqual(retrieved[0]["matched_role"], "assistant")
        self.assertIn("今晚月色真美", retrieved[0]["matched_content"])

    def test_clears_persisted_conversation_for_user_only(self):
        memory_manager.save_conversation_turn("one", "first", "reply")
        memory_manager.save_conversation_turn("two", "second", "reply")

        memory_manager.clear_conversation_messages("one")

        self.assertEqual(memory_manager.get_recent_conversation_messages("one"), [])
        self.assertEqual(
            [item["content"] for item in memory_manager.get_recent_conversation_messages("two")],
            ["second", "reply"],
        )

    def test_memory_consolidation_batches_only_new_user_chat_messages(self):
        for index in range(19):
            memory_manager.save_conversation_turn("user", f"message-{index}", "reply")
        memory_manager.save_conversation_message(
            "user",
            "user",
            "[系統持續屏幕觀測]",
            source="screen_observation",
        )
        self.assertEqual(
            memory_manager.get_pending_memory_consolidation_batch("user"), []
        )

        memory_manager.save_conversation_turn("user", "message-19", "reply")
        batch = memory_manager.get_pending_memory_consolidation_batch("user")
        self.assertEqual(len(batch), 20)
        self.assertTrue(all(item["content"].startswith("message-") for item in batch))

        memory_manager.mark_memory_consolidation_complete(
            "user",
            batch[-1]["message_id"],
        )
        self.assertEqual(
            memory_manager.get_pending_memory_consolidation_batch("user"), []
        )

    def test_explicit_memory_flush_can_consolidate_a_partial_batch(self):
        memory_manager.save_conversation_turn(
            "user",
            "我生日是7月25日，記住喔",
            "好，我會記住。",
        )

        self.assertEqual(
            memory_manager.get_pending_memory_consolidation_batch("user"),
            [],
        )
        batch = memory_manager.get_pending_memory_consolidation_batch(
            "user",
            allow_partial=True,
        )

        self.assertEqual(len(batch), 1)
        self.assertEqual(batch[0]["content"], "我生日是7月25日，記住喔")

    def test_long_memory_keeps_user_quote_provenance_and_updates_in_place(self):
        turn_id = memory_manager.save_conversation_turn(
            "user",
            "我是統計學系的學生。",
            "了解。",
        )
        source_message = memory_manager.get_recent_conversation_messages("user")[0]
        memory_manager.upsert_long_memory(
            "user",
            "使用者是統計學背景",
            [source_message["message_id"]],
        )
        follow_up = memory_manager.save_conversation_turn(
            "user",
            "我現在已經從統計學系畢業了。",
            "恭喜。",
        )
        second_source = memory_manager.get_recent_conversation_messages("user")[-2]
        memory_manager.upsert_long_memory(
            "user",
            "使用者是統計學背景",
            [second_source["message_id"]],
        )
        memory_manager.upsert_long_memory(
            "user",
            "使用者已從統計學系畢業",
            [source_message["message_id"]],
            replaces="使用者是統計學背景",
        )

        memory = memory_manager.get_relevant_memories("user", "統計學")

        self.assertEqual(
            memory["long_memories"],
            ["使用者已從統計學系畢業"],
        )
        self.assertEqual(
            memory["long_memory_details"][0]["sources"][0]["content"],
            "我是統計學系的學生。",
        )
        self.assertEqual(
            memory["long_memory_details"][0]["sources"][0]["created_at"],
            source_message["created_at"],
        )
        self.assertEqual(source_message["turn_id"], turn_id)
        self.assertEqual(
            {source["content"] for source in memory["long_memory_details"][0]["sources"]},
            {"我是統計學系的學生。", "我現在已經從統計學系畢業了。"},
        )
        self.assertEqual(second_source["turn_id"], follow_up)

    def test_long_term_memories_are_not_automatically_pruned(self):
        for index in range(1001):
            memory_manager.add_memory("user", "LONG", f"memory-{index:03}")

        memories = memory_manager.get_user_memory("user")["long_memories"]
        self.assertEqual(len(memories), 1001)
        self.assertIn("memory-000", memories)
        self.assertIn("memory-1000", memories)

    def test_upgrades_an_existing_sqlite_schema_without_losing_memories(self):
        with closing(sqlite3.connect(self.database_path)) as connection:
            with connection:
                connection.execute(
                    """
                    CREATE TABLE memories (
                        user_id TEXT NOT NULL,
                        memory_type TEXT NOT NULL,
                        content TEXT NOT NULL,
                        UNIQUE (user_id, memory_type, content)
                    )
                    """
                )
                connection.execute(
                    "INSERT INTO memories VALUES ('user', 'LONG', 'keep this memory')"
                )

        memory = memory_manager.get_user_memory("user")

        self.assertEqual(memory["long_memories"], ["keep this memory"])
        with closing(sqlite3.connect(self.database_path)) as connection:
            columns = {
                row[1]
                for row in connection.execute("PRAGMA table_info(memories)")
            }
        self.assertTrue(
            {
                "created_at",
                "last_used_at",
                "access_count",
                "source_message_ids",
            }
            <= columns
        )


    def test_agent_explorations_are_persistent_and_scoped_by_user(self):
        memory_manager.record_agent_exploration(
            "one",
            "Quantum clocks",
            "quantum clock",
            summary="Atomic clocks can measure relativistic time shifts.",
            sources=[
                {
                    "title": "Clock research",
                    "url": "https://example.com/clock",
                    "snippet": "A precise clock measurement.",
                }
            ],
            next_question="How precise can future clocks become?",
        )
        memory_manager.record_agent_exploration("one", "Neural networks", "brain research")
        memory_manager.record_agent_exploration("two", "Space", "space research")

        explorations = memory_manager.get_recent_agent_explorations("one")
        self.assertEqual([item["topic"] for item in explorations], ["Neural networks", "Quantum clocks"])
        self.assertEqual(
            explorations[1],
            {
                "topic": "Quantum clocks",
                "query": "quantum clock",
                "summary": "Atomic clocks can measure relativistic time shifts.",
                "sources": [
                    {
                        "title": "Clock research",
                        "url": "https://example.com/clock",
                        "snippet": "A precise clock measurement.",
                    }
                ],
                "next_question": "How precise can future clocks become?",
                "explored_at": explorations[1]["explored_at"],
            },
        )
        self.assertEqual(
            [item["topic"] for item in memory_manager.get_recent_agent_explorations("two")],
            ["Space"],
        )

    def test_existing_exploration_table_is_upgraded_with_note_fields(self):
        with closing(sqlite3.connect(self.database_path)) as connection:
            with connection:
                connection.execute(
                    """
                    CREATE TABLE agent_explorations (
                        user_id TEXT NOT NULL,
                        topic TEXT NOT NULL,
                        query TEXT NOT NULL,
                        explored_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE (user_id, topic, query)
                    )
                    """
                )
                connection.execute(
                    """
                    INSERT INTO agent_explorations (user_id, topic, query)
                    VALUES ('user', 'Old topic', 'old query')
                    """
                )

        exploration = memory_manager.get_recent_agent_explorations("user")[0]

        self.assertEqual(exploration["topic"], "Old topic")
        self.assertEqual(exploration["summary"], "")
        self.assertEqual(exploration["sources"], [])
        self.assertEqual(exploration["next_question"], "")


if __name__ == "__main__":
    unittest.main()

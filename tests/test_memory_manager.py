import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import memory_manager


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

    def test_caps_long_memories_and_prunes_the_oldest_least_used(self):
        for index in range(memory_manager.MAX_LONG_MEMORIES + 1):
            memory_manager.add_memory("user", "LONG", f"memory-{index:03}")

        with closing(sqlite3.connect(self.database_path)) as connection:
            with connection:
                connection.execute(
                    """
                    UPDATE memories
                    SET last_used_at = '2000-01-01 00:00:00',
                        created_at = '2000-01-01 00:00:00'
                    WHERE content = 'memory-000'
                    """
                )

        memory_manager.add_memory(
            "user",
            "LONG",
            f"memory-{memory_manager.MAX_LONG_MEMORIES + 1:03}",
        )

        memories = memory_manager.get_user_memory("user")["long_memories"]
        self.assertEqual(len(memories), memory_manager.MAX_LONG_MEMORIES)
        self.assertNotIn("memory-000", memories)
        self.assertIn(f"memory-{memory_manager.MAX_LONG_MEMORIES + 1:03}", memories)

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
        self.assertTrue({"created_at", "last_used_at", "access_count"} <= columns)


if __name__ == "__main__":
    unittest.main()

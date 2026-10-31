import json
import math
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from settings import DATA_DIR, ROOT_DIR


MEMORY_DB_PATH = DATA_DIR / "user_memory.sqlite3"
LEGACY_MEMORY_FILE = ROOT_DIR / "user_memory.json"
_MIGRATION_NAME = "user_memory_json"
MAX_LONG_MEMORIES = 1000
MAX_PROMPT_MEMORIES = 8
MEMORY_DECAY_HALF_LIFE_DAYS = 90


def _tokenize(text: str) -> set[str]:
    tokens: set[str] = set()
    for token in re.findall(r"[a-z0-9]+|[\u3400-\u9fff]+", text.casefold()):
        if re.fullmatch(r"[\u3400-\u9fff]+", token):
            if len(token) == 1:
                tokens.add(token)
            else:
                tokens.update(token[index:index + 2] for index in range(len(token) - 1))
        else:
            tokens.add(token)
    return tokens


def _memory_score(content: str, last_used_at: str, query_tokens: set[str]) -> float:
    memory_tokens = _tokenize(content)
    if not query_tokens or not memory_tokens:
        return 0.0

    overlap = len(query_tokens & memory_tokens)
    if not overlap:
        return 0.0

    relevance = overlap / math.sqrt(len(query_tokens) * len(memory_tokens))
    last_used = datetime.fromisoformat(last_used_at)
    if last_used.tzinfo is None:
        last_used = last_used.replace(tzinfo=timezone.utc)
    age_days = max(
        0.0,
        (datetime.now(timezone.utc) - last_used.astimezone(timezone.utc)).total_seconds()
        / 86_400,
    )
    recency = 0.5 ** (age_days / MEMORY_DECAY_HALF_LIFE_DAYS)
    return relevance * (0.25 + 0.75 * recency)


def _connect() -> sqlite3.Connection:
    MEMORY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(MEMORY_DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                user_id TEXT NOT NULL,
                memory_type TEXT NOT NULL CHECK (memory_type IN ('SHORT', 'LONG')),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                last_used_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER NOT NULL DEFAULT 0,
                UNIQUE (user_id, memory_type, content)
            )
            """
        )
        columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(memories)")
        }
        for column, definition in (
            ("created_at", "TEXT NOT NULL DEFAULT ''"),
            ("last_used_at", "TEXT NOT NULL DEFAULT ''"),
            ("access_count", "INTEGER NOT NULL DEFAULT 0"),
        ):
            if column not in columns:
                connection.execute(f"ALTER TABLE memories ADD COLUMN {column} {definition}")
        connection.execute(
            """
            UPDATE memories
            SET created_at = CURRENT_TIMESTAMP
            WHERE created_at = ''
            """
        )
        connection.execute(
            """
            UPDATE memories
            SET last_used_at = created_at
            WHERE last_used_at = ''
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS migrations (
                name TEXT PRIMARY KEY,
                completed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                profile_json TEXT NOT NULL
            )
            """
        )
        connection.commit()
        _migrate_legacy_memory(connection)
        return connection
    except BaseException:
        connection.close()
        raise


@contextmanager
def _database_connection() -> Iterator[sqlite3.Connection]:
    connection = _connect()
    try:
        yield connection
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()


def _migrate_legacy_memory(connection: sqlite3.Connection) -> None:
    migrated = connection.execute(
        "SELECT 1 FROM migrations WHERE name = ?",
        (_MIGRATION_NAME,),
    ).fetchone()
    if migrated or not LEGACY_MEMORY_FILE.exists():
        return

    with LEGACY_MEMORY_FILE.open("r", encoding="utf-8") as memory_file:
        legacy_data = json.load(memory_file)
    if not isinstance(legacy_data, dict):
        raise ValueError(f"Legacy memory file must contain a JSON object: {LEGACY_MEMORY_FILE}")

    with connection:
        for user_id, user_data in legacy_data.items():
            if not isinstance(user_data, dict):
                raise ValueError(f"Invalid memory record for user {user_id!r}")
            profile = user_data.get("profile", {})
            if not isinstance(profile, dict):
                raise ValueError(f"Invalid profile for user {user_id!r}")
            connection.execute(
                """
                INSERT OR IGNORE INTO user_profiles (user_id, profile_json)
                VALUES (?, ?)
                """,
                (str(user_id), json.dumps(profile, ensure_ascii=False)),
            )
            for memory_type, field in (
                ("SHORT", "short_memories"),
                ("LONG", "long_memories"),
            ):
                memories = user_data.get(field, [])
                if not isinstance(memories, list) or not all(
                    isinstance(content, str) for content in memories
                ):
                    raise ValueError(f"Invalid {field} for user {user_id!r}")
                connection.executemany(
                    """
                    INSERT OR IGNORE INTO memories (user_id, memory_type, content)
                    VALUES (?, ?, ?)
                    """,
                    ((str(user_id), memory_type, content) for content in memories),
                )
        connection.execute(
            "INSERT INTO migrations (name) VALUES (?)",
            (_MIGRATION_NAME,),
        )


def get_user_memory(user_id: str) -> dict[str, Any]:
    with _database_connection() as connection:
        rows = connection.execute(
            """
            SELECT memory_type, content
            FROM memories
            WHERE user_id = ?
            ORDER BY rowid
            """,
            (user_id,),
        ).fetchall()
        profile_row = connection.execute(
            "SELECT profile_json FROM user_profiles WHERE user_id = ?",
            (user_id,),
        ).fetchone()

    user_memory: dict[str, Any] = {
        "profile": json.loads(profile_row["profile_json"]) if profile_row else {},
        "long_memories": [],
        "short_memories": [],
    }
    for row in rows:
        key = "long_memories" if row["memory_type"] == "LONG" else "short_memories"
        user_memory[key].append(row["content"])
    return user_memory


def get_relevant_memories(
    user_id: str,
    query: str,
    limit: int = MAX_PROMPT_MEMORIES,
) -> dict[str, Any]:
    query_tokens = _tokenize(query)
    if limit < 0:
        raise ValueError("Memory limit cannot be negative")

    with _database_connection() as connection:
        long_memories = connection.execute(
            """
            SELECT rowid, content, last_used_at
            FROM memories
            WHERE user_id = ? AND memory_type = 'LONG'
            """,
            (user_id,),
        ).fetchall()
        short_memories = connection.execute(
            """
            SELECT content
            FROM memories
            WHERE user_id = ? AND memory_type = 'SHORT'
            ORDER BY rowid
            """,
            (user_id,),
        ).fetchall()
        profile_row = connection.execute(
            "SELECT profile_json FROM user_profiles WHERE user_id = ?",
            (user_id,),
        ).fetchone()

        ranked_memories = sorted(
            (
                (
                    _memory_score(row["content"], row["last_used_at"], query_tokens),
                    row,
                )
                for row in long_memories
            ),
            key=lambda item: (item[0], item[1]["last_used_at"]),
            reverse=True,
        )
        selected_memories = [
            row for score, row in ranked_memories if score > 0
        ][:limit]
        if selected_memories:
            connection.executemany(
                """
                UPDATE memories
                SET last_used_at = CURRENT_TIMESTAMP,
                    access_count = access_count + 1
                WHERE rowid = ?
                """,
                ((row["rowid"],) for row in selected_memories),
            )

    return {
        "profile": json.loads(profile_row["profile_json"]) if profile_row else {},
        "long_memories": [row["content"] for row in selected_memories],
        "short_memories": [row["content"] for row in short_memories],
    }


def add_memory(user_id: str, mem_type: str, content: str) -> None:
    if mem_type not in ("SHORT", "LONG"):
        raise ValueError(f"Unsupported memory type: {mem_type}")

    with _database_connection() as connection:
        if mem_type == "SHORT":
            connection.execute(
                "DELETE FROM memories WHERE user_id = ? AND memory_type = 'SHORT'",
                (user_id,),
            )
        connection.execute(
            """
            INSERT INTO memories (user_id, memory_type, content)
            VALUES (?, ?, ?)
            ON CONFLICT (user_id, memory_type, content)
            DO UPDATE SET
                last_used_at = CURRENT_TIMESTAMP,
                access_count = access_count + 1
            """,
            (user_id, mem_type, content),
        )
        if mem_type == "LONG":
            connection.execute(
                """
                DELETE FROM memories
                WHERE user_id = ? AND memory_type = 'LONG'
                  AND rowid IN (
                      SELECT rowid
                      FROM memories
                      WHERE user_id = ? AND memory_type = 'LONG'
                      ORDER BY access_count ASC, last_used_at ASC, created_at ASC, rowid ASC
                      LIMIT MAX(
                          (SELECT COUNT(*) FROM memories
                           WHERE user_id = ? AND memory_type = 'LONG') - ?,
                          0
                      )
                  )
                """,
                (user_id, user_id, user_id, MAX_LONG_MEMORIES),
            )


def clear_short_memory_on_start() -> None:
    with _database_connection() as connection:
        connection.execute("DELETE FROM memories WHERE memory_type = 'SHORT'")

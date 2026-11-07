import json
import math
import re
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

from google.genai.errors import APIError
from shared.settings import DATA_DIR, ROOT_DIR


MEMORY_DB_PATH = DATA_DIR / "user_memory.sqlite3"
LEGACY_MEMORY_FILE = ROOT_DIR / "user_memory.json"
_MIGRATION_NAME = "user_memory_json"
MAX_PROMPT_MEMORIES = 8
MEMORY_DECAY_HALF_LIFE_DAYS = 90
MAX_AGENT_EXPLORATIONS = 100
MAX_EMBEDDING_BATCH_SIZE = 100
MIN_SEMANTIC_SIMILARITY = 0.35
MIN_CONVERSATION_SIMILARITY = 0.45
MAX_CONVERSATION_RETRIEVAL_CANDIDATES = 2000
EmbeddingProvider = Callable[[Sequence[str], str], Sequence[Sequence[float]]]


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
    return relevance * _memory_recency_weight(last_used_at)


def _memory_recency_weight(last_used_at: str) -> float:
    last_used = datetime.fromisoformat(last_used_at)
    if last_used.tzinfo is None:
        last_used = last_used.replace(tzinfo=timezone.utc)
    age_days = max(
        0.0,
        (datetime.now(timezone.utc) - last_used.astimezone(timezone.utc)).total_seconds()
        / 86_400,
    )
    recency = 0.5 ** (age_days / MEMORY_DECAY_HALF_LIFE_DAYS)
    return 0.25 + 0.75 * recency


def _validated_embedding(values: Sequence[float]) -> list[float]:
    embedding = [float(value) for value in values]
    if not embedding or not all(math.isfinite(value) for value in embedding):
        raise ValueError("Embedding response must contain finite values")
    return embedding


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Embedding vectors have inconsistent dimensions")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


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
                embedding TEXT,
                source_message_ids TEXT NOT NULL DEFAULT '[]',
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
            ("embedding", "TEXT"),
            ("source_message_ids", "TEXT NOT NULL DEFAULT '[]'"),
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
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS application_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_messages (
                message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                turn_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'chat',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                embedding TEXT
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_consolidation_state (
                user_id TEXT PRIMARY KEY,
                last_processed_message_id INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_conversation_messages_user
            ON conversation_messages (user_id, message_id DESC)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_conversation_messages_turn
            ON conversation_messages (user_id, turn_id, role)
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_explorations (
                user_id TEXT NOT NULL,
                topic TEXT NOT NULL,
                query TEXT NOT NULL,
                summary TEXT NOT NULL DEFAULT '',
                sources_json TEXT NOT NULL DEFAULT '[]',
                next_question TEXT NOT NULL DEFAULT '',
                explored_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (user_id, topic, query)
            )
            """
        )
        exploration_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(agent_explorations)")
        }
        for column, definition in (
            ("summary", "TEXT NOT NULL DEFAULT ''"),
            ("sources_json", "TEXT NOT NULL DEFAULT '[]'"),
            ("next_question", "TEXT NOT NULL DEFAULT ''"),
        ):
            if column not in exploration_columns:
                connection.execute(
                    f"ALTER TABLE agent_explorations ADD COLUMN {column} {definition}"
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


def get_application_state(key: str) -> str | None:
    with _database_connection() as connection:
        row = connection.execute(
            "SELECT value FROM application_state WHERE key = ?",
            (key,),
        ).fetchone()
    return row["value"] if row else None


def set_application_state(key: str, value: str) -> None:
    with _database_connection() as connection:
        connection.execute(
            """
            INSERT INTO application_state (key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )


def save_conversation_turn(
    user_id: str,
    user_message: str,
    assistant_reply: str,
    source: str = "chat",
) -> str:
    turn_id = uuid.uuid4().hex
    with _database_connection() as connection:
        connection.executemany(
            """
            INSERT INTO conversation_messages
                (user_id, turn_id, role, content, source)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                (user_id, turn_id, "user", user_message, source),
                (user_id, turn_id, "assistant", assistant_reply, source),
            ),
        )
    return turn_id


def save_conversation_message(
    user_id: str,
    role: str,
    content: str,
    source: str = "proactive",
) -> str:
    if role not in {"user", "assistant"}:
        raise ValueError(f"Unsupported conversation role: {role}")
    turn_id = uuid.uuid4().hex
    with _database_connection() as connection:
        connection.execute(
            """
            INSERT INTO conversation_messages
                (user_id, turn_id, role, content, source)
            VALUES (?, ?, ?, ?, ?)
            """,
            (user_id, turn_id, role, content, source),
        )
    return turn_id


def index_conversation_turn(
    user_id: str,
    turn_id: str,
    embed_texts: EmbeddingProvider,
) -> None:
    with _database_connection() as connection:
        messages = connection.execute(
            """
            SELECT message_id, content
            FROM conversation_messages
            WHERE user_id = ? AND turn_id = ? AND source = 'chat'
              AND embedding IS NULL
            ORDER BY message_id
            """,
            (user_id, turn_id),
        ).fetchall()
    if not messages:
        return

    embeddings = embed_texts(
        [message["content"] for message in messages],
        "RETRIEVAL_DOCUMENT",
    )
    if len(embeddings) != len(messages):
        raise ValueError("Embedding provider returned an unexpected result count")

    with _database_connection() as connection:
        for message, values in zip(messages, embeddings):
            connection.execute(
                """
                UPDATE conversation_messages
                SET embedding = ?
                WHERE user_id = ? AND message_id = ?
                """,
                (
                    json.dumps(_validated_embedding(values)),
                    user_id,
                    message["message_id"],
                ),
            )


def get_recent_conversation_messages(
    user_id: str,
    limit: int = 20,
) -> list[dict[str, Any]]:
    if limit < 0:
        raise ValueError("Conversation message limit cannot be negative")
    with _database_connection() as connection:
        rows = connection.execute(
            """
            SELECT message_id, turn_id, role, content, source, created_at
            FROM conversation_messages
            WHERE user_id = ?
            ORDER BY message_id DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()
    return [dict(row) for row in reversed(rows)]


def get_pending_memory_consolidation_batch(
    user_id: str,
    batch_size: int = 20,
    allow_partial: bool = False,
) -> list[dict[str, Any]]:
    if batch_size <= 0:
        raise ValueError("Memory consolidation batch size must be positive")

    with _database_connection() as connection:
        state = connection.execute(
            """
            SELECT last_processed_message_id
            FROM memory_consolidation_state
            WHERE user_id = ?
            """,
            (user_id,),
        ).fetchone()
        last_processed_message_id = (
            state["last_processed_message_id"] if state else 0
        )
        rows = connection.execute(
            """
            SELECT message_id, content, created_at
            FROM conversation_messages
            WHERE user_id = ?
              AND message_id > ?
              AND role = 'user'
              AND source = 'chat'
              AND content NOT LIKE '[系统自动屏幕观测]%'
              AND content NOT LIKE '[系統持續屏幕觀測]%'
            ORDER BY message_id
            LIMIT ?
            """,
            (user_id, last_processed_message_id, batch_size),
        ).fetchall()

    if not rows or (len(rows) < batch_size and not allow_partial):
        return []
    return [dict(row) for row in rows]


def mark_memory_consolidation_complete(
    user_id: str,
    last_processed_message_id: int,
) -> None:
    if last_processed_message_id < 0:
        raise ValueError("Processed conversation message ID cannot be negative")

    with _database_connection() as connection:
        connection.execute(
            """
            INSERT INTO memory_consolidation_state
                (user_id, last_processed_message_id)
            VALUES (?, ?)
            ON CONFLICT (user_id)
            DO UPDATE SET last_processed_message_id = MAX(
                last_processed_message_id,
                excluded.last_processed_message_id
            )
            """,
            (user_id, last_processed_message_id),
        )


def get_relevant_conversation_turns(
    user_id: str,
    query: str,
    recent_user_messages: Sequence[str],
    embed_texts: EmbeddingProvider | None = None,
    limit: int = 5,
    additional_queries: Sequence[str] = (),
) -> list[dict[str, str]]:
    if limit < 0:
        raise ValueError("Conversation retrieval limit cannot be negative")
    queries = [item.strip() for item in (query, *additional_queries) if item.strip()]
    if not queries or limit == 0:
        return []

    with _database_connection() as connection:
        rows = connection.execute(
            """
            SELECT user_message.message_id, user_message.turn_id,
                   user_message.content AS user_message,
                   user_message.created_at, user_message.embedding,
                   assistant_message.content AS assistant_reply,
                   assistant_message.embedding AS assistant_embedding
            FROM conversation_messages AS user_message
            LEFT JOIN conversation_messages AS assistant_message
              ON assistant_message.user_id = user_message.user_id
             AND assistant_message.turn_id = user_message.turn_id
             AND assistant_message.role = 'assistant'
            WHERE user_message.user_id = ?
              AND user_message.role = 'user'
              AND user_message.source = 'chat'
              AND user_message.content NOT LIKE '[系统自动屏幕观测]%'
              AND user_message.content NOT LIKE '[系統持續屏幕觀測]%'
            ORDER BY user_message.message_id DESC
            LIMIT ?
            """,
            (user_id, MAX_CONVERSATION_RETRIEVAL_CANDIDATES),
        ).fetchall()

    recent_messages = set(recent_user_messages)
    candidates = [
        dict(row)
        for row in rows
        if row["user_message"] not in recent_messages
    ]
    if not candidates:
        return []

    query_token_sets = [_tokenize(item) for item in queries]

    def keyword_match(
        row: dict[str, Any],
        query_tokens: set[str],
    ) -> tuple[float, str, str]:
        return max(
            (
                (
                    _memory_score(
                        row[key] or "",
                        row["created_at"],
                        query_tokens,
                    ),
                    role,
                    row[key] or "",
                )
                for role, key in (
                    ("user", "user_message"),
                    ("assistant", "assistant_reply"),
                )
            ),
            key=lambda item: item[0],
        )

    def select_ranked_matches(
        ranked_by_query: list[list[tuple[float, int, dict[str, Any]]]],
    ) -> list[dict[str, Any]]:
        selected_rows = []
        seen_message_ids = set()
        for rank in range(max((len(items) for items in ranked_by_query), default=0)):
            for ranked in reversed(ranked_by_query):
                if rank >= len(ranked):
                    continue
                _, message_id, row = ranked[rank]
                if message_id in seen_message_ids:
                    continue
                seen_message_ids.add(message_id)
                selected_rows.append(row)
                if len(selected_rows) == limit:
                    return selected_rows
        return selected_rows

    selected: list[dict[str, Any]] = []
    if embed_texts is not None and any(
        row["embedding"] or row["assistant_embedding"] for row in candidates
    ):
        try:
            query_result = embed_texts(queries, "RETRIEVAL_QUERY")
            if len(query_result) != len(queries):
                raise ValueError(
                    "Embedding provider returned an unexpected result count"
                )
            query_embeddings = [
                _validated_embedding(values) for values in query_result
            ]
            ranked_by_query = [[] for _ in queries]
            for row in candidates:
                message_embeddings = []
                for role, content, raw_embedding in (
                    ("user", row["user_message"], row["embedding"]),
                    ("assistant", row["assistant_reply"], row["assistant_embedding"]),
                ):
                    if not content or not raw_embedding:
                        continue
                    message_embeddings.append((
                        _validated_embedding(json.loads(raw_embedding)),
                        role,
                        content,
                    ))
                for query_index, query_embedding in enumerate(query_embeddings):
                    comparable_embeddings = [
                        item
                        for item in message_embeddings
                        if len(item[0]) == len(query_embedding)
                    ]
                    if not comparable_embeddings:
                        continue
                    similarity, matched_role, matched_content = max(
                        (
                            _cosine_similarity(query_embedding, embedding),
                            role,
                            content,
                        )
                        for embedding, role, content in comparable_embeddings
                    )
                    if similarity < MIN_CONVERSATION_SIMILARITY:
                        continue
                    score = similarity * _memory_recency_weight(row["created_at"])
                    ranked_by_query[query_index].append((
                        score,
                        row["message_id"],
                        {
                            **row,
                            "matched_role": matched_role,
                            "matched_content": matched_content,
                        },
                    ))
            for ranked in ranked_by_query:
                ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
            selected = select_ranked_matches(ranked_by_query)
        except (APIError, ValueError, TypeError) as error:
            print(
                "[memory] Conversation semantic retrieval failed; "
                f"falling back to keyword matching: {error}"
            )

    if not selected:
        ranked_by_query = [[] for _ in queries]
        for row in candidates:
            for query_index, query_tokens in enumerate(query_token_sets):
                score, matched_role, matched_content = keyword_match(
                    row,
                    query_tokens,
                )
                if score > 0:
                    ranked_by_query[query_index].append((
                        score,
                        row["message_id"],
                        {
                            **row,
                            "matched_role": matched_role,
                            "matched_content": matched_content,
                        },
                    ))
        for ranked in ranked_by_query:
            ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        selected = select_ranked_matches(ranked_by_query)

    return [
        {
            "user_message": row["user_message"],
            "assistant_reply": row["assistant_reply"] or "",
            "created_at": row["created_at"],
            "matched_role": row["matched_role"],
            "matched_content": row["matched_content"],
        }
        for row in selected
    ]


def clear_conversation_messages(user_id: str) -> None:
    with _database_connection() as connection:
        connection.execute(
            "DELETE FROM conversation_messages WHERE user_id = ?",
            (user_id,),
        )


def get_relevant_memories(
    user_id: str,
    query: str,
    limit: int = MAX_PROMPT_MEMORIES,
    embed_texts: EmbeddingProvider | None = None,
) -> dict[str, Any]:
    query_tokens = _tokenize(query)
    if limit < 0:
        raise ValueError("Memory limit cannot be negative")

    with _database_connection() as connection:
        long_memories = connection.execute(
            """
            SELECT rowid, content, last_used_at, embedding, source_message_ids
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

        long_memories = [dict(row) for row in long_memories]

    semantic_embeddings: dict[int, list[float]] = {}
    if embed_texts is not None and query.strip() and long_memories:
        try:
            query_result = embed_texts([query], "RETRIEVAL_QUERY")
            if len(query_result) != 1:
                raise ValueError("Embedding provider returned an unexpected result count")
            query_embedding = _validated_embedding(query_result[0])

            missing = [
                row
                for row in long_memories
                if not row["embedding"]
                or len(
                    _validated_embedding(json.loads(row["embedding"]))
                ) != len(query_embedding)
            ]
            for start in range(0, len(missing), MAX_EMBEDDING_BATCH_SIZE):
                batch = missing[start:start + MAX_EMBEDDING_BATCH_SIZE]
                document_results = embed_texts(
                    [row["content"] for row in batch],
                    "RETRIEVAL_DOCUMENT",
                )
                if len(document_results) != len(batch):
                    raise ValueError(
                        "Embedding provider returned an unexpected result count"
                    )
                with _database_connection() as connection:
                    for row, values in zip(batch, document_results):
                        embedding = _validated_embedding(values)
                        connection.execute(
                            """
                            UPDATE memories
                            SET embedding = ?
                            WHERE rowid = ? AND user_id = ?
                              AND memory_type = 'LONG' AND content = ?
                            """,
                            (
                                json.dumps(embedding),
                                row["rowid"],
                                user_id,
                                row["content"],
                            ),
                        )
                        row["embedding"] = json.dumps(embedding)

            for row in long_memories:
                embedding = (
                    _validated_embedding(json.loads(row["embedding"]))
                    if row["embedding"]
                    else None
                )
                if embedding is not None:
                    semantic_embeddings[row["rowid"]] = embedding

            ranked_memories = []
            for row in long_memories:
                embedding = semantic_embeddings.get(row["rowid"])
                if embedding is None or len(embedding) != len(query_embedding):
                    continue
                similarity = _cosine_similarity(query_embedding, embedding)
                if similarity >= MIN_SEMANTIC_SIMILARITY:
                    score = similarity * _memory_recency_weight(row["last_used_at"])
                    ranked_memories.append((score, row))
            ranked_memories.sort(
                key=lambda item: (item[0], item[1]["last_used_at"]),
                reverse=True,
            )
            selected_memories = [row for _, row in ranked_memories[:limit]]
        except (APIError, ValueError, TypeError) as error:
            print(
                "[memory] Semantic retrieval failed; falling back to keyword matching: "
                f"{error}"
            )
            selected_memories = _rank_keyword_memories(long_memories, query_tokens, limit)
    else:
        selected_memories = _rank_keyword_memories(long_memories, query_tokens, limit)

    if selected_memories:
        with _database_connection() as connection:
            connection.executemany(
                """
                UPDATE memories
                SET last_used_at = CURRENT_TIMESTAMP,
                    access_count = access_count + 1
                WHERE rowid = ?
                """,
                ((row["rowid"],) for row in selected_memories),
            )

    long_memory_details = []
    source_ids = {
        int(message_id)
        for row in selected_memories
        for message_id in json.loads(row["source_message_ids"] or "[]")
        if isinstance(message_id, int) and not isinstance(message_id, bool)
    }
    source_messages: dict[int, dict[str, str]] = {}
    if source_ids:
        placeholders = ",".join("?" for _ in source_ids)
        with _database_connection() as connection:
            source_rows = connection.execute(
                f"""
                SELECT message_id, content, created_at
                FROM conversation_messages
                WHERE user_id = ? AND role = 'user'
                  AND message_id IN ({placeholders})
                """,
                (user_id, *sorted(source_ids)),
            ).fetchall()
        source_messages = {
            row["message_id"]: {
                "content": row["content"],
                "created_at": row["created_at"],
            }
            for row in source_rows
        }

    for row in selected_memories:
        row_source_ids = json.loads(row["source_message_ids"] or "[]")
        long_memory_details.append({
            "content": row["content"],
            "sources": [
                {
                    "message_id": message_id,
                    **source_messages[message_id],
                }
                for message_id in row_source_ids
                if message_id in source_messages
            ],
        })

    return {
        "profile": json.loads(profile_row["profile_json"]) if profile_row else {},
        "long_memories": [row["content"] for row in selected_memories],
        "long_memory_details": long_memory_details,
        "short_memories": [row["content"] for row in short_memories],
    }


def _rank_keyword_memories(
    memories: Sequence[dict[str, Any]],
    query_tokens: set[str],
    limit: int,
) -> list[dict[str, Any]]:
    ranked_memories = sorted(
        (
            (_memory_score(row["content"], row["last_used_at"], query_tokens), row)
            for row in memories
        ),
        key=lambda item: (item[0], item[1]["last_used_at"]),
        reverse=True,
    )
    return [row for score, row in ranked_memories if score > 0][:limit]


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


def upsert_long_memory(
    user_id: str,
    content: str,
    source_message_ids: Sequence[int],
    replaces: str | None = None,
) -> None:
    if not content.strip():
        raise ValueError("Long-term memory content cannot be empty")
    if not source_message_ids or any(
        isinstance(message_id, bool) or not isinstance(message_id, int)
        or message_id <= 0
        for message_id in source_message_ids
    ):
        raise ValueError("Long-term memories require valid source message IDs")

    with _database_connection() as connection:
        existing_sources: set[int] = set()
        existing_content = connection.execute(
            """
            SELECT source_message_ids
            FROM memories
            WHERE user_id = ? AND memory_type = 'LONG' AND content = ?
            """,
            (user_id, content),
        ).fetchone()
        if existing_content is not None:
            existing_sources.update(
                message_id
                for message_id in json.loads(
                    existing_content["source_message_ids"] or "[]"
                )
                if isinstance(message_id, int) and not isinstance(message_id, bool)
            )
        if replaces and replaces != content:
            existing = connection.execute(
                """
                SELECT source_message_ids
                FROM memories
                WHERE user_id = ? AND memory_type = 'LONG' AND content = ?
                """,
                (user_id, replaces),
            ).fetchone()
            if existing is None:
                raise ValueError("Replacement memory does not exist")
            existing_sources = {
                message_id
                for message_id in json.loads(existing["source_message_ids"] or "[]")
                if isinstance(message_id, int) and not isinstance(message_id, bool)
            }
            connection.execute(
                """
                DELETE FROM memories
                WHERE user_id = ? AND memory_type = 'LONG' AND content = ?
                """,
                (user_id, replaces),
            )

        merged_sources = sorted(existing_sources | set(source_message_ids))
        connection.execute(
            """
            INSERT INTO memories
                (user_id, memory_type, content, source_message_ids)
            VALUES (?, 'LONG', ?, ?)
            ON CONFLICT (user_id, memory_type, content)
            DO UPDATE SET
                source_message_ids = excluded.source_message_ids,
                last_used_at = CURRENT_TIMESTAMP
            """,
            (user_id, content, json.dumps(merged_sources)),
        )


def get_recent_agent_explorations(
    user_id: str,
    limit: int = 20,
) -> list[dict[str, str]]:
    if limit < 0:
        raise ValueError("Exploration limit cannot be negative")

    with _database_connection() as connection:
        rows = connection.execute(
            """
            SELECT topic, query, summary, sources_json, next_question, explored_at
            FROM agent_explorations
            WHERE user_id = ?
            ORDER BY explored_at DESC, rowid DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()
    explorations = []
    for row in rows:
        exploration = dict(row)
        exploration["sources"] = json.loads(exploration.pop("sources_json"))
        explorations.append(exploration)
    return explorations


def record_agent_exploration(
    user_id: str,
    topic: str,
    query: str,
    summary: str = "",
    sources: Sequence[dict[str, str]] = (),
    next_question: str = "",
) -> None:
    topic = topic.strip()
    query = query.strip()
    if not topic or not query:
        raise ValueError("Exploration topic and query cannot be empty")

    with _database_connection() as connection:
        connection.execute(
            """
            INSERT INTO agent_explorations (
                user_id, topic, query, summary, sources_json, next_question
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (user_id, topic, query) DO UPDATE SET
                summary = excluded.summary,
                sources_json = excluded.sources_json,
                next_question = excluded.next_question,
                explored_at = CURRENT_TIMESTAMP
            """,
            (
                user_id,
                topic,
                query,
                summary.strip(),
                json.dumps(list(sources), ensure_ascii=False),
                next_question.strip(),
            ),
        )
        connection.execute(
            """
            DELETE FROM agent_explorations
            WHERE user_id = ?
              AND rowid NOT IN (
                  SELECT rowid
                  FROM agent_explorations
                  WHERE user_id = ?
                  ORDER BY explored_at DESC, rowid DESC
                  LIMIT ?
              )
            """,
            (user_id, user_id, MAX_AGENT_EXPLORATIONS),
        )


def clear_short_memory_on_start() -> None:
    with _database_connection() as connection:
        connection.execute("DELETE FROM memories WHERE memory_type = 'SHORT'")

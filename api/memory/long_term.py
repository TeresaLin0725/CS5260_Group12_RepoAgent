"""
Long-term Memory — SQLite-backed persistent knowledge base.

Responsibilities:
  - Persist memory entries across server restarts
  - Full-text search over stored knowledge
  - Weight-based ranking and time-decay scoring
  - Automatic cleanup of expired entries
  - Thread-safe with WAL mode for concurrent reads
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from api.memory.models import (
    MemoryEntry,
    MemoryQuery,
    MemoryStats,
    MemoryTier,
    MemoryType,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_DB_DIR = os.path.join(str(Path.home()), ".adalflow", "memory")
_DEFAULT_DB_NAME = "long_term_memory.db"

# Time-decay half-life: memories lose half their effective weight after
# this many days of not being accessed.
DECAY_HALF_LIFE_DAYS = 30.0

# Score formula: effective_score = weight * decay_factor + access_bonus
ACCESS_BONUS_FACTOR = 0.05   # bonus per access_count (capped)
MAX_ACCESS_BONUS = 2.0


def _decay_factor(last_accessed: datetime, half_life_days: float = DECAY_HALF_LIFE_DAYS) -> float:
    """Exponential decay based on days since last access."""
    days = (datetime.utcnow() - last_accessed).total_seconds() / 86400.0
    if days <= 0:
        return 1.0
    return 0.5 ** (days / half_life_days)


def _effective_score(entry: MemoryEntry) -> float:
    """Compute ranking score combining weight, recency, and access count."""
    decay = _decay_factor(entry.last_accessed_at)
    bonus = min(entry.access_count * ACCESS_BONUS_FACTOR, MAX_ACCESS_BONUS)
    return entry.weight * decay + bonus


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS memories (
    id              TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL,
    repo_id         TEXT NOT NULL,
    memory_type     TEXT NOT NULL,
    key             TEXT NOT NULL,
    value           TEXT NOT NULL,
    weight          REAL NOT NULL DEFAULT 1.0,
    access_count    INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    last_accessed_at TEXT NOT NULL,
    expiry_at       TEXT,
    tier            TEXT NOT NULL DEFAULT 'long_term',
    metadata        TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_memories_user_repo
    ON memories(user_id, repo_id);
CREATE INDEX IF NOT EXISTS idx_memories_type
    ON memories(memory_type);
CREATE INDEX IF NOT EXISTS idx_memories_key
    ON memories(key);
CREATE INDEX IF NOT EXISTS idx_memories_weight
    ON memories(weight DESC);
CREATE INDEX IF NOT EXISTS idx_memories_expiry
    ON memories(expiry_at);

-- Full-text search virtual table for knowledge lookups
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    id,
    key,
    value_text,
    content='memories',
    content_rowid='rowid'
);

-- Triggers to keep FTS table in sync
CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(id, key, value_text)
    VALUES (new.id, new.key, new.value);
END;

CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, id, key, value_text)
    VALUES ('delete', old.id, old.key, old.value);
END;

CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, id, key, value_text)
    VALUES ('delete', old.id, old.key, old.value);
    INSERT INTO memories_fts(id, key, value_text)
    VALUES (new.id, new.key, new.value);
END;
"""


class LongTermMemory:
    """
    SQLite-backed persistent memory store with full-text search.

    Thread-safe — uses one connection per thread via threading.local().
    WAL mode enables concurrent reads while a write is in progress.
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            os.makedirs(_DEFAULT_DB_DIR, exist_ok=True)
            db_path = os.path.join(_DEFAULT_DB_DIR, _DEFAULT_DB_NAME)
        self._db_path = db_path
        self._local = threading.local()
        self._init_lock = threading.Lock()
        # Ensure schema exists on the first connection
        self._initialize_db()
        logger.info(f"Long-term memory initialized at {self._db_path}")

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._db_path, timeout=10)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=-8000")  # 8 MB page cache
            self._local.conn = conn
        return conn

    def _initialize_db(self) -> None:
        with self._init_lock:
            conn = self._get_conn()
            conn.executescript(_SCHEMA_SQL)
            conn.commit()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def store(self, entry: MemoryEntry) -> MemoryEntry:
        """Insert or replace a memory entry."""
        entry.tier = MemoryTier.LONG_TERM
        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO memories
               (id, user_id, repo_id, memory_type, key, value, weight,
                access_count, created_at, updated_at, last_accessed_at,
                expiry_at, tier, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                entry.id,
                entry.user_id,
                entry.repo_id,
                entry.memory_type.value,
                entry.key,
                json.dumps(entry.value, ensure_ascii=False),
                entry.weight,
                entry.access_count,
                entry.created_at.isoformat(),
                entry.updated_at.isoformat(),
                entry.last_accessed_at.isoformat(),
                entry.expiry_at.isoformat() if entry.expiry_at else None,
                entry.tier.value,
                json.dumps(entry.metadata, ensure_ascii=False),
            ),
        )
        conn.commit()
        return entry

    def store_batch(self, entries: List[MemoryEntry]) -> int:
        """Bulk-insert entries. Returns count stored."""
        conn = self._get_conn()
        rows = []
        for e in entries:
            e.tier = MemoryTier.LONG_TERM
            rows.append((
                e.id, e.user_id, e.repo_id, e.memory_type.value, e.key,
                json.dumps(e.value, ensure_ascii=False), e.weight, e.access_count,
                e.created_at.isoformat(), e.updated_at.isoformat(),
                e.last_accessed_at.isoformat(),
                e.expiry_at.isoformat() if e.expiry_at else None,
                e.tier.value,
                json.dumps(e.metadata, ensure_ascii=False),
            ))
        conn.executemany(
            """INSERT OR REPLACE INTO memories
               (id, user_id, repo_id, memory_type, key, value, weight,
                access_count, created_at, updated_at, last_accessed_at,
                expiry_at, tier, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        conn.commit()
        return len(rows)

    def retrieve(self, query: MemoryQuery) -> List[MemoryEntry]:
        """Retrieve entries matching the query, ranked by effective score."""
        conn = self._get_conn()
        clauses = ["user_id = ?", "repo_id = ?"]
        params: list = [query.user_id, query.repo_id]

        if query.memory_types:
            placeholders = ",".join("?" for _ in query.memory_types)
            clauses.append(f"memory_type IN ({placeholders})")
            params.extend(t.value for t in query.memory_types)

        if query.key_prefix:
            clauses.append("key LIKE ?")
            params.append(f"{query.key_prefix}%")

        if query.min_weight > 0:
            clauses.append("weight >= ?")
            params.append(query.min_weight)

        if not query.include_expired:
            clauses.append("(expiry_at IS NULL OR expiry_at > ?)")
            params.append(datetime.utcnow().isoformat())

        where = " AND ".join(clauses)
        # Fetch more than limit so we can re-rank by effective score
        fetch_limit = max(query.limit * 3, 50)
        params.append(fetch_limit)

        rows = conn.execute(
            f"SELECT * FROM memories WHERE {where} ORDER BY weight DESC LIMIT ?",
            params,
        ).fetchall()

        entries = [self._row_to_entry(r) for r in rows]
        # Re-rank by effective score (weight * decay + access bonus)
        entries.sort(key=_effective_score, reverse=True)

        # Touch accessed entries (update last_accessed_at)
        result = entries[: query.limit]
        for e in result:
            e.touch()
            self._update_access(conn, e.id, e.last_accessed_at, e.access_count)
        conn.commit()
        return result

    def search_text(self, user_id: str, repo_id: str, query_text: str, limit: int = 10) -> List[MemoryEntry]:
        """Full-text search across stored knowledge."""
        conn = self._get_conn()
        # Tokenize query into individual terms and join with OR for broad matching
        tokens = [t.strip() for t in query_text.split() if t.strip()]
        if not tokens:
            return []
        # FTS5 query: each token as a separate term joined by OR
        fts_query = " OR ".join(t.replace('"', '""') for t in tokens)
        try:
            rows = conn.execute(
                """SELECT m.* FROM memories m
                   JOIN memories_fts f ON m.id = f.id
                   WHERE f.memories_fts MATCH ?
                     AND m.user_id = ? AND m.repo_id = ?
                     AND (m.expiry_at IS NULL OR m.expiry_at > ?)
                   ORDER BY rank
                   LIMIT ?""",
                (fts_query, user_id, repo_id, datetime.utcnow().isoformat(), limit),
            ).fetchall()
        except Exception as e:
            logger.debug(f"FTS search failed, falling back to LIKE: {e}")
            # Fallback to LIKE-based search
            like_pattern = f"%{query_text[:100]}%"
            rows = conn.execute(
                """SELECT * FROM memories
                   WHERE user_id = ? AND repo_id = ?
                     AND (key LIKE ? OR value LIKE ?)
                     AND (expiry_at IS NULL OR expiry_at > ?)
                   ORDER BY weight DESC
                   LIMIT ?""",
                (user_id, repo_id, like_pattern, like_pattern,
                 datetime.utcnow().isoformat(), limit),
            ).fetchall()
        entries = [self._row_to_entry(r) for r in rows]
        for e in entries:
            e.touch()
            self._update_access(conn, e.id, e.last_accessed_at, e.access_count)
        conn.commit()
        return entries

    def get_by_id(self, entry_id: str) -> Optional[MemoryEntry]:
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM memories WHERE id = ?", (entry_id,)).fetchone()
        return self._row_to_entry(row) if row else None

    def update_weight(self, entry_id: str, new_weight: float) -> Optional[MemoryEntry]:
        conn = self._get_conn()
        now = datetime.utcnow().isoformat()
        conn.execute(
            "UPDATE memories SET weight = ?, updated_at = ? WHERE id = ?",
            (new_weight, now, entry_id),
        )
        conn.commit()
        return self.get_by_id(entry_id)

    def increment_weight(self, entry_id: str, delta: float = 0.1) -> Optional[MemoryEntry]:
        conn = self._get_conn()
        now = datetime.utcnow().isoformat()
        conn.execute(
            "UPDATE memories SET weight = MIN(10.0, weight + ?), updated_at = ? WHERE id = ?",
            (delta, now, entry_id),
        )
        conn.commit()
        return self.get_by_id(entry_id)

    def delete(self, entry_id: str) -> bool:
        conn = self._get_conn()
        cursor = conn.execute("DELETE FROM memories WHERE id = ?", (entry_id,))
        conn.commit()
        return cursor.rowcount > 0

    def delete_by_user(self, user_id: str, repo_id: Optional[str] = None) -> int:
        conn = self._get_conn()
        if repo_id:
            cursor = conn.execute(
                "DELETE FROM memories WHERE user_id = ? AND repo_id = ?",
                (user_id, repo_id),
            )
        else:
            cursor = conn.execute("DELETE FROM memories WHERE user_id = ?", (user_id,))
        conn.commit()
        return cursor.rowcount

    def cleanup_expired(self, user_id: Optional[str] = None) -> int:
        conn = self._get_conn()
        now = datetime.utcnow().isoformat()
        if user_id:
            cursor = conn.execute(
                "DELETE FROM memories WHERE expiry_at IS NOT NULL AND expiry_at < ? AND user_id = ?",
                (now, user_id),
            )
        else:
            cursor = conn.execute(
                "DELETE FROM memories WHERE expiry_at IS NOT NULL AND expiry_at < ?",
                (now,),
            )
        conn.commit()
        deleted = cursor.rowcount
        if deleted > 0:
            logger.info(f"Cleaned up {deleted} expired long-term memories")
        return deleted

    def cleanup_decayed(self, min_effective_score: float = 0.1) -> int:
        """Remove entries whose effective score has decayed below threshold."""
        conn = self._get_conn()
        rows = conn.execute("SELECT * FROM memories").fetchall()
        to_delete = []
        for row in rows:
            entry = self._row_to_entry(row)
            if _effective_score(entry) < min_effective_score:
                to_delete.append(entry.id)
        if to_delete:
            placeholders = ",".join("?" for _ in to_delete)
            conn.execute(f"DELETE FROM memories WHERE id IN ({placeholders})", to_delete)
            conn.commit()
            logger.info(f"Cleaned up {len(to_delete)} decayed long-term memories")
        return len(to_delete)

    def get_stats(self, user_id: str, repo_id: str) -> MemoryStats:
        conn = self._get_conn()
        now = datetime.utcnow().isoformat()
        rows = conn.execute(
            """SELECT * FROM memories
               WHERE user_id = ? AND repo_id = ?
                 AND (expiry_at IS NULL OR expiry_at > ?)""",
            (user_id, repo_id, now),
        ).fetchall()

        if not rows:
            return MemoryStats(
                total_count=0, by_type={}, by_tier={},
                oldest_memory=None, newest_memory=None,
                total_weight=0.0, avg_weight=0.0,
            )

        by_type: Dict[str, int] = {}
        by_tier: Dict[str, int] = {}
        total_weight = 0.0
        oldest = None
        newest = None

        for r in rows:
            mt = r["memory_type"]
            by_type[mt] = by_type.get(mt, 0) + 1
            tier = r["tier"]
            by_tier[tier] = by_tier.get(tier, 0) + 1
            total_weight += r["weight"]
            created = datetime.fromisoformat(r["created_at"])
            updated = datetime.fromisoformat(r["updated_at"])
            if oldest is None or created < oldest:
                oldest = created
            if newest is None or updated > newest:
                newest = updated

        return MemoryStats(
            total_count=len(rows),
            by_type=by_type,
            by_tier=by_tier,
            oldest_memory=oldest,
            newest_memory=newest,
            total_weight=total_weight,
            avg_weight=total_weight / len(rows),
        )

    def count(self, user_id: Optional[str] = None, repo_id: Optional[str] = None) -> int:
        conn = self._get_conn()
        clauses = []
        params: list = []
        if user_id:
            clauses.append("user_id = ?")
            params.append(user_id)
        if repo_id:
            clauses.append("repo_id = ?")
            params.append(repo_id)
        where = " AND ".join(clauses) if clauses else "1=1"
        row = conn.execute(f"SELECT COUNT(*) as cnt FROM memories WHERE {where}", params).fetchone()
        return row["cnt"]

    # ------------------------------------------------------------------
    # Preference helpers (same interface as original MemoryManager)
    # ------------------------------------------------------------------

    def get_preferences(self, user_id: str, repo_id: str) -> Dict[str, Any]:
        query = MemoryQuery(
            user_id=user_id, repo_id=repo_id,
            memory_types=[MemoryType.PREFERENCE],
            limit=100,
        )
        prefs = {}
        for entry in self.retrieve(query):
            prefs[entry.key] = entry.value
        return prefs

    def set_preference(self, user_id: str, repo_id: str, key: str, value: Any) -> MemoryEntry:
        conn = self._get_conn()
        val = value if isinstance(value, dict) else {"value": value}
        now = datetime.utcnow()

        # Check if preference already exists
        row = conn.execute(
            """SELECT id FROM memories
               WHERE user_id = ? AND repo_id = ? AND memory_type = ? AND key = ?""",
            (user_id, repo_id, MemoryType.PREFERENCE.value, key),
        ).fetchone()

        if row:
            conn.execute(
                """UPDATE memories SET value = ?, updated_at = ?, last_accessed_at = ?
                   WHERE id = ?""",
                (json.dumps(val, ensure_ascii=False), now.isoformat(), now.isoformat(), row["id"]),
            )
            conn.commit()
            return self.get_by_id(row["id"])  # type: ignore
        else:
            entry = MemoryEntry.create(
                user_id=user_id, repo_id=repo_id,
                memory_type=MemoryType.PREFERENCE,
                key=key, value=val,
                tier=MemoryTier.LONG_TERM,
            )
            return self.store(entry)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> MemoryEntry:
        data = dict(row)
        return MemoryEntry.from_dict(data)

    @staticmethod
    def _update_access(conn: sqlite3.Connection, entry_id: str, ts: datetime, count: int) -> None:
        conn.execute(
            "UPDATE memories SET last_accessed_at = ?, access_count = ? WHERE id = ?",
            (ts.isoformat(), count, entry_id),
        )

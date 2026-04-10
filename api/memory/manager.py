"""
Multi-tier Memory Manager — orchestrates short-term, episodic, and long-term memory.

Architecture:
  ┌────────────────┐    ┌──────────────────┐    ┌─────────────────┐
  │  Short-term    │───>│  Episodic        │───>│  Long-term      │
  │  (Working)     │    │  (Session)       │    │  (Knowledge DB) │
  │  In-memory     │    │  In-memory +     │    │  SQLite + FTS5  │
  │  Sliding window│    │  Consolidation   │    │  Persistent     │
  └────────────────┘    └──────────────────┘    └─────────────────┘

The MemoryManager provides a unified API that is backward-compatible
with the original interface while adding multi-tier capabilities.
"""

from __future__ import annotations

import atexit
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from api.memory.models import (
    MemoryEntry,
    MemoryQuery,
    MemoryStats,
    MemoryTier,
    MemoryType,
    ConversationTurn,
    SessionContext,
)
from api.memory.short_term import ShortTermMemory
from api.memory.episodic import EpisodicMemory
from api.memory.long_term import LongTermMemory
from api.memory.consolidation import ConsolidationEngine

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Unified multi-tier memory manager.

    Backward-compatible: all existing store/retrieve/set_preference calls
    continue to work. New callers can use tier-specific APIs for finer control.

    Responsibilities:
      - Route entries to the appropriate tier based on MemoryType
      - Provide a merged view when querying across tiers
      - Manage the consolidation lifecycle (episodic → long-term)
      - Expose working-memory helpers for conversation context
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        max_short_term_turns: int = 50,
        max_short_term_tokens: int = 32_000,
        max_episodic_sessions: int = 200,
        session_ttl_hours: int = 24,
        consolidation_interval_seconds: int = 300,
        enable_consolidation: bool = True,
    ):
        # --- Tier instances ---
        self.short_term = ShortTermMemory(
            max_turns=max_short_term_turns,
            max_tokens=max_short_term_tokens,
        )
        self.episodic = EpisodicMemory(
            max_sessions=max_episodic_sessions,
            session_ttl_hours=session_ttl_hours,
        )
        self.long_term = LongTermMemory(db_path=db_path)

        # --- Consolidation engine ---
        self._consolidation = ConsolidationEngine(
            episodic=self.episodic,
            long_term=self.long_term,
            interval_seconds=consolidation_interval_seconds,
        )
        if enable_consolidation:
            self._consolidation.start()

        # --- In-memory buffer for INTERACTION/CONTEXT (backward compat) ---
        self._volatile: Dict[str, MemoryEntry] = {}
        self._volatile_index: Dict[str, List[str]] = {}

        logger.info(
            "MemoryManager initialized: short_term(turns=%d,tokens=%d) "
            "episodic(sessions=%d,ttl=%dh) long_term(sqlite) consolidation=%s",
            max_short_term_turns, max_short_term_tokens,
            max_episodic_sessions, session_ttl_hours,
            "ON" if enable_consolidation else "OFF",
        )

    # ================================================================
    # Backward-compatible API (drop-in replacement)
    # ================================================================

    def store(self, entry: MemoryEntry) -> MemoryEntry:
        """
        Store a memory entry — automatically routes to the right tier.

        - PREFERENCE, INSIGHT, KNOWLEDGE, SUMMARY → long-term (persistent)
        - INTERACTION, CONTEXT → volatile in-memory buffer
        """
        if entry.memory_type in (
            MemoryType.PREFERENCE,
            MemoryType.INSIGHT,
            MemoryType.KNOWLEDGE,
            MemoryType.SUMMARY,
        ):
            entry.tier = MemoryTier.LONG_TERM
            return self.long_term.store(entry)
        else:
            # INTERACTION / CONTEXT stay in volatile memory
            entry.tier = MemoryTier.SHORT_TERM
            self._volatile[entry.id] = entry
            idx_key = f"{entry.user_id}:{entry.repo_id}:{entry.key}"
            self._volatile_index.setdefault(idx_key, []).append(entry.id)
            return entry

    def retrieve(self, query: MemoryQuery) -> List[MemoryEntry]:
        """
        Retrieve memories across tiers, merged and ranked by weight.

        If query.tier is specified, only that tier is searched.
        Otherwise, searches volatile + long-term and merges results.
        """
        results: List[MemoryEntry] = []

        if query.tier is None or query.tier == MemoryTier.SHORT_TERM:
            results.extend(self._retrieve_volatile(query))

        if query.tier is None or query.tier == MemoryTier.LONG_TERM:
            results.extend(self.long_term.retrieve(query))

        # Deduplicate by id
        seen_ids = set()
        unique = []
        for e in results:
            if e.id not in seen_ids:
                seen_ids.add(e.id)
                unique.append(e)

        # Sort by weight descending
        unique.sort(key=lambda m: m.weight, reverse=True)
        return unique[: query.limit]

    def update_weight(self, entry_id: str, new_weight: float) -> Optional[MemoryEntry]:
        # Try volatile first
        if entry_id in self._volatile:
            entry = self._volatile[entry_id]
            entry.weight = new_weight
            entry.updated_at = datetime.utcnow()
            return entry
        # Then long-term
        return self.long_term.update_weight(entry_id, new_weight)

    def increment_weight(self, entry_id: str, delta: float = 0.1) -> Optional[MemoryEntry]:
        if entry_id in self._volatile:
            entry = self._volatile[entry_id]
            entry.weight = min(10.0, entry.weight + delta)
            entry.updated_at = datetime.utcnow()
            return entry
        return self.long_term.increment_weight(entry_id, delta)

    def get_preferences(self, user_id: str, repo_id: str) -> Dict[str, Any]:
        """Get user preferences from long-term storage."""
        return self.long_term.get_preferences(user_id, repo_id)

    def set_preference(self, user_id: str, repo_id: str, key: str, value: Any) -> MemoryEntry:
        """Set a user preference (persisted to long-term)."""
        return self.long_term.set_preference(user_id, repo_id, key, value)

    def delete(self, entry_id: str) -> bool:
        if entry_id in self._volatile:
            self._volatile.pop(entry_id)
            return True
        return self.long_term.delete(entry_id)

    def cleanup_expired(self, user_id: Optional[str] = None) -> int:
        # Volatile cleanup
        volatile_deleted = 0
        to_delete = [
            eid for eid, e in self._volatile.items()
            if e.is_expired() and (user_id is None or e.user_id == user_id)
        ]
        for eid in to_delete:
            self._volatile.pop(eid)
            volatile_deleted += 1

        # Long-term cleanup
        lt_deleted = self.long_term.cleanup_expired(user_id)
        return volatile_deleted + lt_deleted

    def get_stats(self, user_id: str, repo_id: str) -> MemoryStats:
        """Merged stats across all tiers."""
        lt_stats = self.long_term.get_stats(user_id, repo_id)

        # Add volatile stats
        volatile_entries = [
            e for e in self._volatile.values()
            if e.user_id == user_id and e.repo_id == repo_id and not e.is_expired()
        ]
        total = lt_stats.total_count + len(volatile_entries)
        by_type = dict(lt_stats.by_type)
        by_tier = dict(lt_stats.by_tier)
        by_tier["volatile"] = len(volatile_entries)

        for e in volatile_entries:
            by_type[e.memory_type.value] = by_type.get(e.memory_type.value, 0) + 1

        total_weight = lt_stats.total_weight + sum(e.weight for e in volatile_entries)
        avg_weight = total_weight / total if total > 0 else 0.0

        oldest = lt_stats.oldest_memory
        newest = lt_stats.newest_memory
        if volatile_entries:
            vol_oldest = min(e.created_at for e in volatile_entries)
            vol_newest = max(e.updated_at for e in volatile_entries)
            oldest = min(oldest, vol_oldest) if oldest else vol_oldest
            newest = max(newest, vol_newest) if newest else vol_newest

        return MemoryStats(
            total_count=total,
            by_type=by_type,
            by_tier=by_tier,
            oldest_memory=oldest,
            newest_memory=newest,
            total_weight=total_weight,
            avg_weight=avg_weight,
        )

    # ================================================================
    # Short-term (Working Memory) API
    # ================================================================

    def add_conversation_turn(
        self,
        session_key: str,
        user_query: str,
        assistant_response: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ConversationTurn:
        """Add a dialog turn to the working memory window."""
        return self.short_term.add_turn(session_key, user_query, assistant_response, metadata)

    def get_conversation_context(self, session_key: str, last_n: Optional[int] = None) -> str:
        """Get formatted working-memory context for prompt injection."""
        return self.short_term.get_context_string(session_key, last_n)

    def get_conversation_turns(self, session_key: str, last_n: Optional[int] = None) -> List[ConversationTurn]:
        """Get raw conversation turns from working memory."""
        return self.short_term.get_window(session_key, last_n)

    # ================================================================
    # Episodic (Session) Memory API
    # ================================================================

    def track_session_turn(
        self,
        session_key: str,
        user_id: str,
        repo_id: str,
        user_query: str,
        assistant_response: str,
        topics: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SessionContext:
        """Track a conversation turn in the episodic session."""
        return self.episodic.add_turn(
            session_key, user_id, repo_id,
            user_query, assistant_response,
            topics, metadata,
        )

    def get_session_context(
        self,
        session_key: str,
        current_query: str,
        max_turns: int = 5,
    ) -> Dict[str, Any]:
        """Get episodic context for the current query."""
        return self.episodic.get_context_for_query(session_key, current_query, max_turns)

    def close_session(self, session_key: str) -> Optional[SessionContext]:
        """Close an episodic session and trigger consolidation."""
        session = self.episodic.close_session(session_key)
        if session:
            # Immediately promote insights from this session
            insights = self.episodic.extract_insights_for_promotion(session)
            for entry in insights:
                if entry.weight >= 1.5:
                    self.long_term.store(entry)
        self.short_term.clear_session(session_key)
        return session

    # ================================================================
    # Long-term Knowledge Base API
    # ================================================================

    def search_knowledge(self, user_id: str, repo_id: str, query_text: str, limit: int = 10) -> List[MemoryEntry]:
        """Full-text search across the persistent knowledge base."""
        return self.long_term.search_text(user_id, repo_id, query_text, limit)

    def store_knowledge(
        self,
        user_id: str,
        repo_id: str,
        key: str,
        value: Dict[str, Any],
        weight: float = 2.0,
        expiry_days: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryEntry:
        """Explicitly store a knowledge entry in the long-term database."""
        entry = MemoryEntry.create(
            user_id=user_id,
            repo_id=repo_id,
            memory_type=MemoryType.KNOWLEDGE,
            key=key,
            value=value,
            weight=weight,
            expiry_days=expiry_days,
            metadata=metadata,
            tier=MemoryTier.LONG_TERM,
        )
        return self.long_term.store(entry)

    def get_user_insights(self, user_id: str, repo_id: str, limit: int = 20) -> List[MemoryEntry]:
        """Retrieve consolidated insights for a user."""
        return self.long_term.retrieve(MemoryQuery(
            user_id=user_id,
            repo_id=repo_id,
            memory_types=[MemoryType.INSIGHT, MemoryType.SUMMARY],
            limit=limit,
        ))

    def get_frequent_topics(self, user_id: str, min_count: int = 3) -> List[Dict[str, Any]]:
        """Get topics the user frequently asks about."""
        return self.episodic.get_frequent_topics(user_id, min_count)

    # ================================================================
    # Lifecycle
    # ================================================================

    def consolidate_now(self) -> Dict[str, int]:
        """Manually trigger a consolidation cycle."""
        return self._consolidation.consolidate_now()

    def shutdown(self) -> None:
        """Graceful shutdown — flush episodic data to long-term."""
        logger.info("MemoryManager shutting down — flushing data...")
        self._consolidation.consolidate_now()
        self._consolidation.stop()
        logger.info("MemoryManager shutdown complete")

    # ================================================================
    # Internal
    # ================================================================

    def _retrieve_volatile(self, query: MemoryQuery) -> List[MemoryEntry]:
        """Filter volatile in-memory entries by query criteria."""
        results = []
        for entry in self._volatile.values():
            if entry.user_id != query.user_id or entry.repo_id != query.repo_id:
                continue
            if query.memory_types and entry.memory_type not in query.memory_types:
                continue
            if query.key_prefix and not entry.key.startswith(query.key_prefix):
                continue
            if entry.weight < query.min_weight:
                continue
            if entry.is_expired() and not query.include_expired:
                continue
            results.append(entry)
        results.sort(key=lambda m: m.weight, reverse=True)
        return results[: query.limit]


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

_memory_manager: Optional[MemoryManager] = None


def get_memory_manager(**kwargs) -> MemoryManager:
    """Get or create the global memory manager singleton."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager(**kwargs)
        atexit.register(_memory_manager.shutdown)
    return _memory_manager

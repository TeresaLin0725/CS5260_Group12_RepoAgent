"""
Memory Consolidation Engine — promotes episodic memories to long-term storage.

Responsibilities:
  - Periodically scan episodic sessions for promotion candidates
  - Extract insights and summaries from mature sessions
  - Deduplicate knowledge before writing to long-term store
  - Decay stale long-term entries
  - Run as a background maintenance task
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Optional

from api.memory.models import MemoryEntry, MemoryQuery, MemoryTier, MemoryType

if TYPE_CHECKING:
    from api.memory.episodic import EpisodicMemory
    from api.memory.long_term import LongTermMemory

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# How often the consolidation loop runs (seconds)
DEFAULT_INTERVAL_SECONDS = 300  # 5 minutes

# Minimum weight to promote an episodic insight to long-term
MIN_PROMOTION_WEIGHT = 1.5

# Maximum age for long-term decay cleanup (days without access)
DECAY_CLEANUP_THRESHOLD = 0.1  # effective score threshold


class ConsolidationEngine:
    """
    Background engine that consolidates episodic memory into long-term storage.

    Lifecycle:
      1. Scan episodic sessions for maturity
      2. Extract insights and summaries
      3. Deduplicate against existing long-term entries
      4. Store new knowledge
      5. Clean up decayed long-term entries periodically
    """

    def __init__(
        self,
        episodic: EpisodicMemory,
        long_term: LongTermMemory,
        interval_seconds: int = DEFAULT_INTERVAL_SECONDS,
    ):
        self._episodic = episodic
        self._long_term = long_term
        self._interval = interval_seconds
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background consolidation loop."""
        if self._running:
            return
        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="memory-consolidation",
            daemon=True,
        )
        self._thread.start()
        logger.info("Memory consolidation engine started (interval=%ds)", self._interval)

    def stop(self) -> None:
        """Signal the background loop to stop."""
        self._stop_event.set()
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)
        logger.info("Memory consolidation engine stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # Core consolidation logic
    # ------------------------------------------------------------------

    def consolidate_now(self) -> Dict[str, int]:
        """
        Run one consolidation cycle immediately. Returns stats about what happened.
        Can be called externally (e.g. before shutdown).
        """
        stats = {
            "sessions_scanned": 0,
            "entries_promoted": 0,
            "entries_deduplicated": 0,
            "expired_cleaned": 0,
            "decayed_cleaned": 0,
        }

        try:
            # 1. Get sessions ready for consolidation
            sessions = self._episodic.get_sessions_for_consolidation()
            stats["sessions_scanned"] = len(sessions)

            for session in sessions:
                # 2. Extract promotable insights
                insights = self._episodic.extract_insights_for_promotion(session)

                for entry in insights:
                    if entry.weight < MIN_PROMOTION_WEIGHT:
                        continue

                    # 3. Deduplicate — check if similar entry exists in long-term
                    if self._is_duplicate(entry):
                        stats["entries_deduplicated"] += 1
                        # Reinforce existing entry instead
                        self._reinforce_existing(entry)
                        continue

                    # 4. Store in long-term
                    self._long_term.store(entry)
                    stats["entries_promoted"] += 1

            # 5. Cleanup expired long-term entries
            stats["expired_cleaned"] = self._long_term.cleanup_expired()

            # 6. Cleanup decayed entries (run less frequently)
            stats["decayed_cleaned"] = self._long_term.cleanup_decayed(DECAY_CLEANUP_THRESHOLD)

            # 7. Cleanup expired episodic sessions
            expired_sessions = self._episodic.cleanup_expired()
            for expired in expired_sessions:
                insights = self._episodic.extract_insights_for_promotion(expired)
                for entry in insights:
                    if entry.weight >= MIN_PROMOTION_WEIGHT and not self._is_duplicate(entry):
                        self._long_term.store(entry)
                        stats["entries_promoted"] += 1

            if stats["entries_promoted"] > 0:
                logger.info(
                    "Consolidation: promoted=%d deduplicated=%d expired_cleaned=%d",
                    stats["entries_promoted"],
                    stats["entries_deduplicated"],
                    stats["expired_cleaned"],
                )

        except Exception:
            logger.exception("Error during memory consolidation")

        return stats

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def _is_duplicate(self, entry: MemoryEntry) -> bool:
        """Check if a similar entry already exists in long-term storage."""
        existing = self._long_term.retrieve(MemoryQuery(
            user_id=entry.user_id,
            repo_id=entry.repo_id,
            memory_types=[entry.memory_type],
            key_prefix=entry.key[:20],  # Prefix match for similar keys
            limit=5,
        ))
        for ex in existing:
            # Same key is a duplicate
            if ex.key == entry.key:
                return True
            # Very similar value content
            ex_str = str(ex.value)
            entry_str = str(entry.value)
            if ex_str and entry_str:
                # Simple overlap check — ratio of common chars
                shorter = min(len(ex_str), len(entry_str))
                if shorter > 0:
                    common = sum(1 for a, b in zip(ex_str, entry_str) if a == b)
                    if common / shorter > 0.8:
                        return True
        return False

    def _reinforce_existing(self, entry: MemoryEntry) -> None:
        """Boost weight of an existing similar entry instead of duplicating."""
        existing = self._long_term.retrieve(MemoryQuery(
            user_id=entry.user_id,
            repo_id=entry.repo_id,
            memory_types=[entry.memory_type],
            key_prefix=entry.key[:20],
            limit=1,
        ))
        if existing:
            self._long_term.increment_weight(existing[0].id, delta=0.2)

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        """Background thread that periodically runs consolidation."""
        while not self._stop_event.is_set():
            try:
                self.consolidate_now()
            except Exception:
                logger.exception("Unhandled error in consolidation loop")
            self._stop_event.wait(self._interval)

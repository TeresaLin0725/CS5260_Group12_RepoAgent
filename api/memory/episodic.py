"""
Episodic (Intermediate) Memory — session-scoped cross-query context.

Responsibilities:
  - Track multi-turn conversation context within a session
  - Automatic summarization of long conversations
  - Extract recurring topics/patterns for promotion to long-term
  - Bridge between short-term working memory and long-term knowledge
  - Thread-safe operations
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import uuid4

from api.memory.models import (
    ConversationTurn,
    MemoryEntry,
    MemoryTier,
    MemoryType,
    SessionContext,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MAX_SESSIONS = 200       # Max concurrent sessions in memory
DEFAULT_SESSION_TTL_HOURS = 24   # Session expiry after last activity
DEFAULT_SUMMARY_THRESHOLD = 10   # Summarize after this many turns
DEFAULT_MAX_TOPICS = 20          # Max tracked topics per session


class EpisodicMemory:
    """
    Session-scoped intermediate memory with automatic consolidation hooks.

    Each user+repo session accumulates conversation turns and extracted
    topics. When a session is closed or exceeds thresholds, the memory
    can be summarized and key insights promoted to long-term storage.
    """

    def __init__(
        self,
        max_sessions: int = DEFAULT_MAX_SESSIONS,
        session_ttl_hours: int = DEFAULT_SESSION_TTL_HOURS,
        summary_threshold: int = DEFAULT_SUMMARY_THRESHOLD,
    ):
        self._max_sessions = max_sessions
        self._session_ttl = timedelta(hours=session_ttl_hours)
        self._summary_threshold = summary_threshold
        # session_key → SessionContext
        self._sessions: Dict[str, SessionContext] = {}
        # Track topic frequencies across all sessions for a user
        # user_id → {topic: count}
        self._topic_frequencies: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def get_or_create_session(
        self,
        session_key: str,
        user_id: str,
        repo_id: str,
    ) -> SessionContext:
        """Get an existing session or create a new one."""
        with self._lock:
            if session_key in self._sessions:
                session = self._sessions[session_key]
                session.updated_at = datetime.utcnow()
                return session

            # Evict oldest sessions if at capacity
            self._evict_sessions()

            session = SessionContext(
                session_id=str(uuid4()),
                user_id=user_id,
                repo_id=repo_id,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )
            self._sessions[session_key] = session
            return session

    def add_turn(
        self,
        session_key: str,
        user_id: str,
        repo_id: str,
        user_query: str,
        assistant_response: str,
        topics: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SessionContext:
        """Add a conversation turn to the session and track topics."""
        session = self.get_or_create_session(session_key, user_id, repo_id)

        turn = ConversationTurn(
            turn_id=str(uuid4()),
            user_query=user_query,
            assistant_response=assistant_response,
            timestamp=datetime.utcnow(),
            metadata=metadata or {},
        )

        with self._lock:
            session.turns.append(turn)
            session.updated_at = datetime.utcnow()

            # Track topics
            if topics:
                for topic in topics:
                    if topic not in session.topics and len(session.topics) < DEFAULT_MAX_TOPICS:
                        session.topics.append(topic)
                    self._topic_frequencies[user_id][topic] += 1

        return session

    def get_session(self, session_key: str) -> Optional[SessionContext]:
        with self._lock:
            return self._sessions.get(session_key)

    def get_session_turns(self, session_key: str) -> List[ConversationTurn]:
        with self._lock:
            session = self._sessions.get(session_key)
            return list(session.turns) if session else []

    def get_session_summary(self, session_key: str) -> str:
        """Get or generate summary for a session."""
        with self._lock:
            session = self._sessions.get(session_key)
            if not session:
                return ""
            if session.summary:
                return session.summary
            # Generate a basic extractive summary from conversation
            return self._generate_extractive_summary(session)

    def close_session(self, session_key: str) -> Optional[SessionContext]:
        """Close a session and return it for consolidation."""
        with self._lock:
            session = self._sessions.pop(session_key, None)
            if session and not session.summary:
                session.summary = self._generate_extractive_summary(session)
            return session

    # ------------------------------------------------------------------
    # Context retrieval for prompts
    # ------------------------------------------------------------------

    def get_context_for_query(
        self,
        session_key: str,
        current_query: str,
        max_turns: int = 5,
    ) -> Dict[str, Any]:
        """
        Build contextual information for the current query.

        Returns a dict with:
          - recent_turns: last N turns as formatted text
          - session_topics: topics discussed in this session
          - session_summary: summary of the session so far
          - related_topics: topics the user frequently asks about
        """
        with self._lock:
            session = self._sessions.get(session_key)
            if not session:
                return {
                    "recent_turns": "",
                    "session_topics": [],
                    "session_summary": "",
                    "related_topics": [],
                }

            # Recent turns
            recent = session.turns[-max_turns:] if session.turns else []
            recent_text_parts = []
            for t in recent:
                recent_text_parts.append(f"Q: {t.user_query[:200]}")
                resp_preview = t.assistant_response[:300] if t.assistant_response else ""
                recent_text_parts.append(f"A: {resp_preview}")
            recent_text = "\n".join(recent_text_parts)

            # Related topics from user history
            user_topics = self._topic_frequencies.get(session.user_id, {})
            related = sorted(user_topics.items(), key=lambda x: x[1], reverse=True)[:10]
            related_topics = [t for t, _ in related]

            return {
                "recent_turns": recent_text,
                "session_topics": list(session.topics),
                "session_summary": session.summary or self._generate_extractive_summary(session),
                "related_topics": related_topics,
            }

    # ------------------------------------------------------------------
    # Consolidation helpers
    # ------------------------------------------------------------------

    def get_sessions_for_consolidation(self) -> List[SessionContext]:
        """
        Return sessions that are ready for consolidation to long-term.

        Criteria:
          - Session has more turns than the summary threshold
          - Session is expired (inactive for longer than TTL)
        """
        now = datetime.utcnow()
        result = []
        with self._lock:
            for key, session in list(self._sessions.items()):
                is_stale = (now - session.updated_at) > self._session_ttl
                is_large = len(session.turns) >= self._summary_threshold
                if is_stale or is_large:
                    if not session.summary:
                        session.summary = self._generate_extractive_summary(session)
                    result.append(session)
        return result

    def get_frequent_topics(self, user_id: str, min_count: int = 3) -> List[Dict[str, Any]]:
        """Return topics the user asks about frequently."""
        with self._lock:
            user_topics = self._topic_frequencies.get(user_id, {})
            return [
                {"topic": topic, "count": count}
                for topic, count in sorted(user_topics.items(), key=lambda x: x[1], reverse=True)
                if count >= min_count
            ]

    def extract_insights_for_promotion(self, session: SessionContext) -> List[MemoryEntry]:
        """
        Extract promotable insights from a session.

        This creates MemoryEntry objects suitable for long-term storage.
        """
        entries: List[MemoryEntry] = []

        # Ensure summary is generated
        summary = session.summary or self._generate_extractive_summary(session)

        # 1. Session summary as a SUMMARY entry
        if summary:
            entries.append(MemoryEntry.create(
                user_id=session.user_id,
                repo_id=session.repo_id,
                memory_type=MemoryType.SUMMARY,
                key=f"session_summary_{session.session_id[:8]}",
                value={
                    "summary": summary,
                    "turn_count": len(session.turns),
                    "topics": session.topics,
                    "session_start": session.created_at.isoformat(),
                    "session_end": session.updated_at.isoformat(),
                },
                weight=min(2.0 + len(session.turns) * 0.1, 5.0),
                tier=MemoryTier.LONG_TERM,
                expiry_days=90,
            ))

        # 2. Frequently discussed topics as INSIGHT entries
        for topic in session.topics:
            freq = self._topic_frequencies.get(session.user_id, {}).get(topic, 1)
            if freq >= 2:
                entries.append(MemoryEntry.create(
                    user_id=session.user_id,
                    repo_id=session.repo_id,
                    memory_type=MemoryType.INSIGHT,
                    key=f"topic_interest_{topic[:50]}",
                    value={"topic": topic, "frequency": freq},
                    weight=min(1.0 + freq * 0.3, 5.0),
                    tier=MemoryTier.LONG_TERM,
                    expiry_days=180,
                ))

        return entries

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup_expired(self) -> List[SessionContext]:
        """Remove expired sessions. Returns them for optional consolidation."""
        now = datetime.utcnow()
        expired = []
        with self._lock:
            for key in list(self._sessions.keys()):
                session = self._sessions[key]
                if (now - session.updated_at) > self._session_ttl:
                    if not session.summary:
                        session.summary = self._generate_extractive_summary(session)
                    expired.append(self._sessions.pop(key))
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired episodic sessions")
        return expired

    def session_count(self) -> int:
        with self._lock:
            return len(self._sessions)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _evict_sessions(self) -> None:
        """Evict oldest sessions when at capacity. Caller must hold lock."""
        while len(self._sessions) >= self._max_sessions:
            # Find oldest by updated_at
            oldest_key = min(self._sessions, key=lambda k: self._sessions[k].updated_at)
            session = self._sessions.pop(oldest_key)
            logger.debug(f"Evicted episodic session {oldest_key} (last active: {session.updated_at})")

    @staticmethod
    def _generate_extractive_summary(session: SessionContext) -> str:
        """
        Generate a simple extractive summary from conversation turns.

        For production, this could call an LLM for abstractive summarization.
        Here we build a concise extractive summary from query highlights.
        """
        if not session.turns:
            return ""

        # Collect unique query topics
        queries = []
        for turn in session.turns:
            q = turn.user_query.strip()
            if q and len(q) > 5:
                queries.append(q[:150])

        if not queries:
            return ""

        # Build summary
        n_turns = len(session.turns)
        topic_str = ", ".join(session.topics[:5]) if session.topics else "general"

        summary_parts = [
            f"Session with {n_turns} exchange(s) about: {topic_str}.",
            "Questions discussed:",
        ]
        # Include up to 8 unique query previews
        seen = set()
        for q in queries:
            normalized = q.lower()[:80]
            if normalized not in seen:
                seen.add(normalized)
                summary_parts.append(f"  - {q}")
            if len(seen) >= 8:
                break

        return "\n".join(summary_parts)

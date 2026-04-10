"""
Short-term (Working) Memory — per-session conversation window.

Responsibilities:
  - Maintain a token-aware sliding window of recent dialog turns
  - LRU eviction when window exceeds capacity
  - Fast O(1) access to the most recent context
  - Thread-safe operations
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from api.memory.models import ConversationTurn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MAX_TURNS = 50          # Max dialog turns per session
DEFAULT_MAX_TOKENS = 32_000     # Total token budget for working memory
ESTIMATED_CHARS_PER_TOKEN = 4   # Rough heuristic for token estimation


def _estimate_tokens(text: str) -> int:
    """Quick token estimate without loading tiktoken."""
    return max(1, len(text) // ESTIMATED_CHARS_PER_TOKEN)


class ShortTermMemory:
    """
    Token-aware sliding-window working memory.

    Each session (user+repo pair) gets its own window. When the window
    exceeds the configured capacity (turns or tokens), the oldest turns
    are evicted first.
    """

    def __init__(
        self,
        max_turns: int = DEFAULT_MAX_TURNS,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        self._max_turns = max_turns
        self._max_tokens = max_tokens
        # session_key → OrderedDict[turn_id → ConversationTurn]
        self._sessions: Dict[str, OrderedDict[str, ConversationTurn]] = {}
        self._token_counts: Dict[str, int] = {}  # session_key → total tokens
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_turn(
        self,
        session_key: str,
        user_query: str,
        assistant_response: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ConversationTurn:
        """Append a dialog turn and evict oldest if over budget."""
        tokens = _estimate_tokens(user_query) + _estimate_tokens(assistant_response)
        turn = ConversationTurn(
            turn_id=str(uuid4()),
            user_query=user_query,
            assistant_response=assistant_response,
            timestamp=datetime.utcnow(),
            token_count=tokens,
            metadata=metadata or {},
        )

        with self._lock:
            if session_key not in self._sessions:
                self._sessions[session_key] = OrderedDict()
                self._token_counts[session_key] = 0

            window = self._sessions[session_key]
            window[turn.turn_id] = turn
            self._token_counts[session_key] += tokens

            # Evict oldest turns while over budget
            self._evict(session_key)

        return turn

    def get_window(self, session_key: str, last_n: Optional[int] = None) -> List[ConversationTurn]:
        """Return the most recent turns for a session."""
        with self._lock:
            window = self._sessions.get(session_key)
            if not window:
                return []
            turns = list(window.values())
            if last_n is not None:
                turns = turns[-last_n:]
            return turns

    def get_context_string(self, session_key: str, last_n: Optional[int] = None) -> str:
        """Format the working memory as a prompt-ready string."""
        turns = self.get_window(session_key, last_n)
        if not turns:
            return ""
        parts: List[str] = []
        for t in turns:
            parts.append(f"<user>{t.user_query}</user>")
            parts.append(f"<assistant>{t.assistant_response}</assistant>")
        return "\n".join(parts)

    def get_token_usage(self, session_key: str) -> int:
        with self._lock:
            return self._token_counts.get(session_key, 0)

    def clear_session(self, session_key: str) -> int:
        """Clear all turns for a session. Returns the number of turns removed."""
        with self._lock:
            window = self._sessions.pop(session_key, None)
            self._token_counts.pop(session_key, None)
            return len(window) if window else 0

    def get_all_turns(self, session_key: str) -> List[ConversationTurn]:
        """Return ALL turns (for episodic consolidation before clear)."""
        with self._lock:
            window = self._sessions.get(session_key)
            if not window:
                return []
            return list(window.values())

    def session_count(self) -> int:
        with self._lock:
            return len(self._sessions)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _evict(self, session_key: str) -> None:
        """Evict oldest turns until within capacity. Caller must hold lock."""
        window = self._sessions[session_key]

        # Evict by turn count
        while len(window) > self._max_turns:
            _, evicted = window.popitem(last=False)
            self._token_counts[session_key] -= evicted.token_count

        # Evict by token budget
        while self._token_counts[session_key] > self._max_tokens and len(window) > 1:
            _, evicted = window.popitem(last=False)
            self._token_counts[session_key] -= evicted.token_count

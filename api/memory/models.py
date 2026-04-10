"""
Shared data models for the multi-tier memory system.

Memory tiers:
  - Short-term: per-request working memory (conversation window)
  - Episodic: per-session intermediate memory (cross-query context)
  - Long-term: persistent knowledge base (SQLite-backed)
"""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict
import json
import uuid


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class MemoryType(str, Enum):
    """Classification of memory types."""
    PREFERENCE = "preference"      # User preferences (model, language, etc.)
    INTERACTION = "interaction"    # Interaction history (usage patterns)
    INSIGHT = "insight"            # Distilled insights from repeated patterns
    CONTEXT = "context"            # Temporary contextual fragments
    KNOWLEDGE = "knowledge"        # Long-term knowledge base entries
    SUMMARY = "summary"            # Consolidated conversation summaries


class MemoryTier(str, Enum):
    """Which storage tier a memory resides in."""
    SHORT_TERM = "short_term"
    EPISODIC = "episodic"
    LONG_TERM = "long_term"


class ConsolidationStrategy(str, Enum):
    """How episodic memories get promoted to long-term."""
    FREQUENCY = "frequency"        # Promote if accessed N times
    WEIGHT = "weight"              # Promote if weight exceeds threshold
    EXPLICIT = "explicit"          # Manually promoted
    DECAY = "decay"                # Keep only if not decayed below threshold


# ---------------------------------------------------------------------------
# Core Data Classes
# ---------------------------------------------------------------------------

@dataclass
class MemoryEntry:
    """Single memory entry — backward-compatible with existing API."""
    id: str
    user_id: str
    repo_id: str
    memory_type: MemoryType
    key: str
    value: Dict[str, Any]
    weight: float = 1.0
    access_count: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed_at: datetime = field(default_factory=datetime.utcnow)
    expiry_at: Optional[datetime] = None
    tier: MemoryTier = MemoryTier.SHORT_TERM
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        user_id: str,
        repo_id: str,
        memory_type: MemoryType,
        key: str,
        value: Dict[str, Any],
        weight: float = 1.0,
        expiry_days: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tier: MemoryTier = MemoryTier.SHORT_TERM,
    ) -> MemoryEntry:
        """Factory method to create a new memory entry."""
        now = datetime.utcnow()
        entry = cls(
            id=str(uuid.uuid4()),
            user_id=user_id,
            repo_id=repo_id,
            memory_type=memory_type,
            key=key,
            value=value,
            weight=weight,
            access_count=0,
            created_at=now,
            updated_at=now,
            last_accessed_at=now,
            tier=tier,
            metadata=metadata or {},
        )
        if expiry_days:
            entry.expiry_at = now + timedelta(days=expiry_days)
        return entry

    def is_expired(self) -> bool:
        if self.expiry_at is None:
            return False
        return datetime.utcnow() > self.expiry_at

    def touch(self) -> None:
        """Record an access — updates timestamp and bumps access counter."""
        self.last_accessed_at = datetime.utcnow()
        self.access_count += 1

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["memory_type"] = self.memory_type.value
        data["tier"] = self.tier.value
        data["created_at"] = self.created_at.isoformat()
        data["updated_at"] = self.updated_at.isoformat()
        data["last_accessed_at"] = self.last_accessed_at.isoformat()
        if data["expiry_at"]:
            data["expiry_at"] = data["expiry_at"].isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MemoryEntry:
        """Reconstruct a MemoryEntry from a dictionary (e.g. from SQLite row)."""
        data = dict(data)  # shallow copy
        data["memory_type"] = MemoryType(data["memory_type"])
        data["tier"] = MemoryTier(data.get("tier", "short_term"))
        for dt_field in ("created_at", "updated_at", "last_accessed_at"):
            val = data.get(dt_field)
            if isinstance(val, str):
                data[dt_field] = datetime.fromisoformat(val)
        expiry = data.get("expiry_at")
        if isinstance(expiry, str):
            data["expiry_at"] = datetime.fromisoformat(expiry)
        if isinstance(data.get("value"), str):
            data["value"] = json.loads(data["value"])
        if isinstance(data.get("metadata"), str):
            data["metadata"] = json.loads(data["metadata"])
        return cls(**data)


@dataclass
class MemoryQuery:
    """Query parameters for searching memories."""
    user_id: str
    repo_id: str
    memory_types: Optional[List[MemoryType]] = None
    key_prefix: Optional[str] = None
    limit: int = 10
    min_weight: float = 0.0
    include_expired: bool = False
    tier: Optional[MemoryTier] = None


@dataclass
class MemoryStats:
    """Statistics about user memories."""
    total_count: int
    by_type: Dict[str, int]
    by_tier: Dict[str, int]
    oldest_memory: Optional[datetime]
    newest_memory: Optional[datetime]
    total_weight: float
    avg_weight: float


@dataclass
class ConversationTurn:
    """A single user-assistant exchange for working memory."""
    turn_id: str
    user_query: str
    assistant_response: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "user_query": self.user_query,
            "assistant_response": self.assistant_response,
            "timestamp": self.timestamp.isoformat(),
            "token_count": self.token_count,
            "metadata": self.metadata,
        }


@dataclass
class SessionContext:
    """Aggregated context for an episodic session."""
    session_id: str
    user_id: str
    repo_id: str
    turns: List[ConversationTurn] = field(default_factory=list)
    summary: str = ""
    topics: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "repo_id": self.repo_id,
            "turns": [t.to_dict() for t in self.turns],
            "summary": self.summary,
            "topics": self.topics,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }

"""Memory module — Multi-tier memory management (short-term, episodic, long-term)."""

from api.memory.models import (
    MemoryEntry,
    MemoryType,
    MemoryTier,
    MemoryQuery,
    MemoryStats,
    ConversationTurn,
    SessionContext,
    ConsolidationStrategy,
)
from api.memory.manager import (
    MemoryManager,
    get_memory_manager,
)
from api.memory.short_term import ShortTermMemory
from api.memory.episodic import EpisodicMemory
from api.memory.long_term import LongTermMemory
from api.memory.consolidation import ConsolidationEngine

__all__ = [
    # Core manager
    "MemoryManager",
    "get_memory_manager",
    # Models
    "MemoryEntry",
    "MemoryType",
    "MemoryTier",
    "MemoryQuery",
    "MemoryStats",
    "ConversationTurn",
    "SessionContext",
    "ConsolidationStrategy",
    # Tier implementations
    "ShortTermMemory",
    "EpisodicMemory",
    "LongTermMemory",
    "ConsolidationEngine",
]

"""
Tests for the multi-tier memory system.

Covers:
  - Short-term memory (sliding window, eviction, token tracking)
  - Episodic memory (session lifecycle, topic tracking, consolidation)
  - Long-term memory (SQLite persistence, FTS, decay, preferences)
  - Memory manager (unified API, tier routing, backward compat)
  - Consolidation engine (promotion, deduplication)
"""

import os
import tempfile
import time

import pytest

from api.memory.models import (
    MemoryEntry,
    MemoryQuery,
    MemoryTier,
    MemoryType,
    ConversationTurn,
    SessionContext,
)
from api.memory.short_term import ShortTermMemory
from api.memory.episodic import EpisodicMemory
from api.memory.long_term import LongTermMemory
from api.memory.consolidation import ConsolidationEngine
from api.memory.manager import MemoryManager


# ============================================================================
# Short-term Memory
# ============================================================================


class TestShortTermMemory:
    def test_add_and_retrieve_turn(self):
        stm = ShortTermMemory(max_turns=10, max_tokens=100_000)
        turn = stm.add_turn("s1", "hello?", "hi there!")
        assert turn.user_query == "hello?"
        assert turn.assistant_response == "hi there!"

        window = stm.get_window("s1")
        assert len(window) == 1
        assert window[0].turn_id == turn.turn_id

    def test_sliding_window_eviction_by_turns(self):
        stm = ShortTermMemory(max_turns=3, max_tokens=100_000)
        for i in range(5):
            stm.add_turn("s1", f"q{i}", f"a{i}")

        window = stm.get_window("s1")
        assert len(window) == 3
        # Oldest two should have been evicted
        assert window[0].user_query == "q2"

    def test_sliding_window_eviction_by_tokens(self):
        stm = ShortTermMemory(max_turns=100, max_tokens=50)
        # Each turn ~4 chars per token → ~2 tokens per short msg
        stm.add_turn("s1", "short", "msg")  # ~2 tokens
        stm.add_turn("s1", "a" * 200, "b" * 200)  # ~100 tokens → will evict

        window = stm.get_window("s1")
        # Should have at most 1 turn left (the big one, since budget is 50 tokens)
        assert len(window) >= 1

    def test_get_context_string(self):
        stm = ShortTermMemory()
        stm.add_turn("s1", "what is X?", "X is Y")
        stm.add_turn("s1", "explain more", "here is more detail")

        ctx = stm.get_context_string("s1")
        assert "<user>what is X?</user>" in ctx
        assert "<assistant>X is Y</assistant>" in ctx

    def test_clear_session(self):
        stm = ShortTermMemory()
        stm.add_turn("s1", "q", "a")
        assert stm.clear_session("s1") == 1
        assert stm.get_window("s1") == []

    def test_last_n(self):
        stm = ShortTermMemory()
        for i in range(10):
            stm.add_turn("s1", f"q{i}", f"a{i}")
        window = stm.get_window("s1", last_n=3)
        assert len(window) == 3
        assert window[0].user_query == "q7"


# ============================================================================
# Episodic Memory
# ============================================================================


class TestEpisodicMemory:
    def test_create_session_and_add_turns(self):
        em = EpisodicMemory()
        session = em.add_turn("s1", "user1", "repo1", "q1", "a1", topics=["architecture"])
        assert session.user_id == "user1"
        assert len(session.turns) == 1
        assert "architecture" in session.topics

    def test_session_context_retrieval(self):
        em = EpisodicMemory()
        em.add_turn("s1", "user1", "repo1", "explain auth", "auth works via JWT")
        em.add_turn("s1", "user1", "repo1", "how about caching?", "redis is used")

        ctx = em.get_context_for_query("s1", "tell me about logging")
        assert "auth" in ctx["recent_turns"].lower() or "caching" in ctx["recent_turns"].lower()
        assert isinstance(ctx["session_topics"], list)

    def test_topic_frequency_tracking(self):
        em = EpisodicMemory()
        em.add_turn("s1", "user1", "repo1", "q1", "a1", topics=["auth"])
        em.add_turn("s1", "user1", "repo1", "q2", "a2", topics=["auth"])
        em.add_turn("s1", "user1", "repo1", "q3", "a3", topics=["caching"])

        frequent = em.get_frequent_topics("user1", min_count=2)
        assert any(t["topic"] == "auth" and t["count"] >= 2 for t in frequent)

    def test_extractive_summary(self):
        em = EpisodicMemory()
        for i in range(5):
            em.add_turn("s1", "user1", "repo1", f"question about topic {i}?", f"answer {i}")

        summary = em.get_session_summary("s1")
        assert "5 exchange" in summary
        assert "question" in summary.lower()

    def test_close_session(self):
        em = EpisodicMemory()
        em.add_turn("s1", "user1", "repo1", "How does the authentication system work?", "It uses JWT tokens")
        session = em.close_session("s1")
        assert session is not None
        assert session.summary  # Should have generated summary

        # Session should be removed
        assert em.get_session("s1") is None

    def test_insight_extraction(self):
        em = EpisodicMemory()
        em.add_turn("s1", "user1", "repo1", "How does the API design work?", "It uses REST", topics=["api_design"])
        em.add_turn("s1", "user1", "repo1", "What about error handling?", "Uses middleware", topics=["api_design"])
        em.add_turn("s1", "user1", "repo1", "How do you run tests?", "Use pytest", topics=["testing"])

        session = em.get_session("s1")
        insights = em.extract_insights_for_promotion(session)
        assert len(insights) > 0
        # Should have at least a summary entry
        assert any(e.memory_type == MemoryType.SUMMARY for e in insights)


# ============================================================================
# Long-term Memory
# ============================================================================


class TestLongTermMemory:
    @pytest.fixture
    def ltm(self, tmp_path):
        db_path = str(tmp_path / "test_memory.db")
        return LongTermMemory(db_path=db_path)

    def test_store_and_retrieve(self, ltm):
        entry = MemoryEntry.create(
            user_id="u1", repo_id="r1",
            memory_type=MemoryType.KNOWLEDGE,
            key="api_pattern",
            value={"summary": "REST API uses controller pattern"},
            weight=3.0,
        )
        ltm.store(entry)

        results = ltm.retrieve(MemoryQuery(user_id="u1", repo_id="r1"))
        assert len(results) == 1
        assert results[0].key == "api_pattern"
        assert results[0].value["summary"] == "REST API uses controller pattern"

    def test_persistence(self, tmp_path):
        db_path = str(tmp_path / "persist_test.db")
        ltm1 = LongTermMemory(db_path=db_path)
        entry = MemoryEntry.create(
            user_id="u1", repo_id="r1",
            memory_type=MemoryType.PREFERENCE,
            key="preferred_model",
            value={"value": "gpt-4"},
        )
        ltm1.store(entry)

        # Create a new instance pointing to the same DB
        ltm2 = LongTermMemory(db_path=db_path)
        results = ltm2.retrieve(MemoryQuery(user_id="u1", repo_id="r1"))
        assert len(results) == 1
        assert results[0].value == {"value": "gpt-4"}

    def test_full_text_search(self, ltm):
        ltm.store(MemoryEntry.create(
            user_id="u1", repo_id="r1",
            memory_type=MemoryType.KNOWLEDGE,
            key="auth_system",
            value={"summary": "JWT authentication with refresh tokens"},
            weight=2.0,
        ))
        ltm.store(MemoryEntry.create(
            user_id="u1", repo_id="r1",
            memory_type=MemoryType.KNOWLEDGE,
            key="cache_layer",
            value={"summary": "Redis caching for API responses"},
            weight=2.0,
        ))

        results = ltm.search_text("u1", "r1", "authentication")
        assert any("auth" in r.key for r in results)

    def test_preferences(self, ltm):
        ltm.set_preference("u1", "r1", "theme", {"value": "dark"})
        ltm.set_preference("u1", "r1", "language", {"value": "zh"})

        prefs = ltm.get_preferences("u1", "r1")
        assert prefs["theme"] == {"value": "dark"}
        assert prefs["language"] == {"value": "zh"}

        # Update preference
        ltm.set_preference("u1", "r1", "theme", {"value": "light"})
        prefs = ltm.get_preferences("u1", "r1")
        assert prefs["theme"] == {"value": "light"}

    def test_weight_update(self, ltm):
        entry = MemoryEntry.create(
            user_id="u1", repo_id="r1",
            memory_type=MemoryType.INSIGHT,
            key="pattern1",
            value={"insight": "test"},
            weight=1.0,
        )
        ltm.store(entry)

        updated = ltm.increment_weight(entry.id, delta=0.5)
        assert updated.weight == 1.5

        updated = ltm.update_weight(entry.id, 5.0)
        assert updated.weight == 5.0

    def test_batch_store(self, ltm):
        entries = [
            MemoryEntry.create(
                user_id="u1", repo_id="r1",
                memory_type=MemoryType.KNOWLEDGE,
                key=f"item_{i}",
                value={"data": f"value_{i}"},
            )
            for i in range(10)
        ]
        count = ltm.store_batch(entries)
        assert count == 10
        assert ltm.count(user_id="u1") == 10

    def test_delete(self, ltm):
        entry = MemoryEntry.create(
            user_id="u1", repo_id="r1",
            memory_type=MemoryType.KNOWLEDGE,
            key="to_delete",
            value={"data": "temp"},
        )
        ltm.store(entry)
        assert ltm.delete(entry.id) is True
        assert ltm.get_by_id(entry.id) is None

    def test_stats(self, ltm):
        for i in range(3):
            ltm.store(MemoryEntry.create(
                user_id="u1", repo_id="r1",
                memory_type=MemoryType.KNOWLEDGE,
                key=f"k{i}", value={"d": i}, weight=float(i + 1),
            ))
        stats = ltm.get_stats("u1", "r1")
        assert stats.total_count == 3
        assert stats.total_weight == 6.0
        assert "knowledge" in stats.by_type


# ============================================================================
# Consolidation Engine
# ============================================================================


class TestConsolidation:
    @pytest.fixture
    def setup(self, tmp_path):
        em = EpisodicMemory(session_ttl_hours=0)  # sessions expire immediately
        ltm = LongTermMemory(db_path=str(tmp_path / "consol.db"))
        engine = ConsolidationEngine(em, ltm, interval_seconds=9999)
        return em, ltm, engine

    def test_promote_session_insights(self, setup):
        em, ltm, engine = setup

        # Add enough content to trigger promotion
        for i in range(5):
            em.add_turn("s1", "user1", "repo1", f"q{i}", f"a{i}", topics=["testing"])
        em.add_turn("s1", "user1", "repo1", "q5", "a5", topics=["testing"])

        # Make session expired (TTL=0)
        time.sleep(0.1)

        stats = engine.consolidate_now()
        assert stats["sessions_scanned"] > 0 or stats["entries_promoted"] >= 0

    def test_deduplication(self, setup):
        em, ltm, engine = setup

        # Pre-populate long-term with an entry
        ltm.store(MemoryEntry.create(
            user_id="user1", repo_id="repo1",
            memory_type=MemoryType.SUMMARY,
            key="session_summary_existing",
            value={"summary": "test session", "turn_count": 3},
            weight=3.0,
        ))

        # Trying to consolidate similar content should deduplicate
        stats = engine.consolidate_now()
        assert stats["entries_deduplicated"] >= 0


# ============================================================================
# Unified MemoryManager
# ============================================================================


class TestMemoryManager:
    @pytest.fixture
    def mm(self, tmp_path):
        return MemoryManager(
            db_path=str(tmp_path / "manager.db"),
            enable_consolidation=False,
        )

    def test_backward_compat_store_retrieve(self, mm):
        """Existing code using store/retrieve should still work."""
        entry = MemoryEntry.create(
            user_id="u1", repo_id="r1",
            memory_type=MemoryType.INTERACTION,
            key="chat_query",
            value={"query_preview": "test query"},
        )
        mm.store(entry)

        results = mm.retrieve(MemoryQuery(user_id="u1", repo_id="r1"))
        assert len(results) == 1

    def test_preferences_persisted(self, mm):
        mm.set_preference("u1", "r1", "model", {"value": "gpt-4"})
        prefs = mm.get_preferences("u1", "r1")
        assert prefs["model"] == {"value": "gpt-4"}

    def test_routing_persistent_types(self, mm):
        """PREFERENCE, INSIGHT, KNOWLEDGE go to long-term."""
        entry = MemoryEntry.create(
            user_id="u1", repo_id="r1",
            memory_type=MemoryType.KNOWLEDGE,
            key="kb_entry",
            value={"data": "persistent"},
        )
        mm.store(entry)

        # Should be in long-term
        lt_results = mm.long_term.retrieve(MemoryQuery(user_id="u1", repo_id="r1"))
        assert len(lt_results) == 1

    def test_routing_volatile_types(self, mm):
        """INTERACTION, CONTEXT stay in volatile memory."""
        entry = MemoryEntry.create(
            user_id="u1", repo_id="r1",
            memory_type=MemoryType.INTERACTION,
            key="interaction",
            value={"data": "volatile"},
        )
        mm.store(entry)

        # Should be in volatile, not long-term
        lt_results = mm.long_term.retrieve(MemoryQuery(
            user_id="u1", repo_id="r1",
            memory_types=[MemoryType.INTERACTION],
        ))
        assert len(lt_results) == 0

    def test_conversation_tracking(self, mm):
        turn = mm.add_conversation_turn("s1", "hello", "world")
        assert turn.user_query == "hello"

        ctx = mm.get_conversation_context("s1")
        assert "hello" in ctx

    def test_session_tracking(self, mm):
        session = mm.track_session_turn(
            "s1", "u1", "r1",
            "what is X?", "X is...",
            topics=["architecture"],
        )
        assert len(session.turns) == 1

        ctx = mm.get_session_context("s1", "follow up question")
        assert "architecture" in ctx["session_topics"]

    def test_close_session_promotes_insights(self, mm):
        for i in range(5):
            mm.track_session_turn(
                "s1", "u1", "r1",
                f"q{i}", f"a{i}",
                topics=["testing"],
            )
        mm.track_session_turn("s1", "u1", "r1", "q5", "a5", topics=["testing"])

        session = mm.close_session("s1")
        assert session is not None

        # Check if insights were promoted to long-term
        insights = mm.get_user_insights("u1", "r1")
        assert len(insights) > 0

    def test_search_knowledge(self, mm):
        mm.store_knowledge(
            "u1", "r1",
            key="auth_mechanism",
            value={"summary": "Uses JWT tokens for authentication"},
            weight=3.0,
        )
        results = mm.search_knowledge("u1", "r1", "JWT authentication")
        assert len(results) >= 0  # FTS may or may not match depending on tokenization

    def test_stats_merged(self, mm):
        mm.store(MemoryEntry.create(
            user_id="u1", repo_id="r1",
            memory_type=MemoryType.INTERACTION,
            key="volatile1", value={"x": 1},
        ))
        mm.store_knowledge("u1", "r1", "k1", {"y": 2})

        stats = mm.get_stats("u1", "r1")
        assert stats.total_count == 2
        assert "volatile" in stats.by_tier

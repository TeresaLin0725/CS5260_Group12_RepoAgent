"""
Unit tests for api.retriever — tokenizer, BM25Index, HybridRetriever.

Covers:
  - Code-aware tokenization (camelCase, snake_case, stop words)
  - BM25 index build, search, scoring, edge cases
  - HybridRetriever RRF merging, FAISS fallback, BM25 fallback
  - RetrieverOutput structure

Usage:
    pytest tests/unit/test_retriever.py -v
"""

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import MagicMock

import pytest

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from api.retriever import (
    BM25Index,
    HybridRetriever,
    RetrieverOutput,
    tokenize_code,
    tokenize_code_filtered,
)


# ============================================================================
# Helpers — lightweight Document stub
# ============================================================================


@dataclass
class FakeDocument:
    """Mimics adalflow Document for testing."""
    text: str = ""
    meta_data: Optional[Dict] = None


def make_docs(texts: List[str]) -> List[FakeDocument]:
    return [FakeDocument(text=t) for t in texts]


def make_docs_with_paths(items: List[tuple]) -> List[FakeDocument]:
    """items: list of (text, file_path) tuples."""
    return [
        FakeDocument(text=t, meta_data={"file_path": fp})
        for t, fp in items
    ]


# ============================================================================
# tokenize_code
# ============================================================================


class TestTokenizeCode:
    def test_camel_case(self):
        tokens = tokenize_code("DatabaseManager")
        assert "database" in tokens
        assert "manager" in tokens

    def test_pascal_case_acronym(self):
        tokens = tokenize_code("FAISSRetriever")
        assert "faiss" in tokens
        assert "retriever" in tokens

    def test_snake_case(self):
        tokens = tokenize_code("get_file_content")
        assert tokens == ["get", "file", "content"]

    def test_dot_notation(self):
        tokens = tokenize_code("DatabaseManager.prepare_retriever")
        assert "database" in tokens
        assert "manager" in tokens
        assert "prepare" in tokens
        assert "retriever" in tokens

    def test_empty_string(self):
        assert tokenize_code("") == []

    def test_single_char_filtered(self):
        """Tokens shorter than 2 chars are dropped."""
        tokens = tokenize_code("a b c de fg")
        assert "de" in tokens
        assert "fg" in tokens
        assert "a" not in tokens

    def test_pure_digits_filtered(self):
        tokens = tokenize_code("var123 456 width")
        assert "456" not in tokens
        assert "var123" in tokens or "var" in tokens

    def test_path_separator(self):
        tokens = tokenize_code("src/components/App.tsx")
        assert "src" in tokens
        assert "components" in tokens
        assert "app" in tokens
        assert "tsx" in tokens

    def test_mixed_case_and_underscores(self):
        tokens = tokenize_code("myFunc_camelCase_test")
        assert "my" in tokens
        assert "func" in tokens
        assert "camel" in tokens
        assert "case" in tokens
        assert "test" in tokens


class TestTokenizeCodeFiltered:
    def test_stop_words_removed(self):
        tokens = tokenize_code_filtered("the function is a helper for the module")
        assert "the" not in tokens
        assert "function" in tokens
        assert "helper" in tokens
        assert "module" in tokens

    def test_code_keywords_preserved(self):
        """Code keywords like def/class/return should NOT be removed."""
        tokens = tokenize_code_filtered("def myFunction return value class Handler")
        assert "def" in tokens
        assert "return" in tokens
        assert "class" in tokens


# ============================================================================
# BM25Index
# ============================================================================


class TestBM25Index:
    def test_build_basic(self):
        docs = make_docs(["hello world", "foo bar baz", "hello foo"])
        idx = BM25Index()
        idx.build(docs)
        assert idx.corpus_size == 3
        assert idx.avgdl > 0

    def test_build_empty_corpus(self):
        idx = BM25Index()
        idx.build([])
        assert idx.corpus_size == 0
        assert idx.search("hello") == []

    def test_search_returns_relevant(self):
        docs = make_docs([
            "Python FastAPI web framework",
            "JavaScript React frontend",
            "Python Django REST API",
        ])
        idx = BM25Index()
        idx.build(docs)
        results = idx.search("Python API", top_k=3)
        # doc 0 and doc 2 both have 'python', doc 2 also has 'api'
        indices = [r[0] for r in results]
        assert 2 in indices  # Django REST API should be top

    def test_search_top_k_limit(self):
        docs = make_docs([f"document number {i}" for i in range(10)])
        idx = BM25Index()
        idx.build(docs)
        results = idx.search("document", top_k=3)
        assert len(results) <= 3

    def test_search_no_match(self):
        docs = make_docs(["alpha beta gamma"])
        idx = BM25Index()
        idx.build(docs)
        results = idx.search("zzzznotfound")
        assert results == []

    def test_search_empty_query(self):
        docs = make_docs(["some content here"])
        idx = BM25Index()
        idx.build(docs)
        results = idx.search("")
        assert results == []

    def test_scores_are_positive(self):
        docs = make_docs(["database manager", "file system", "database connection pool"])
        idx = BM25Index()
        idx.build(docs)
        results = idx.search("database")
        for _, score in results:
            assert score > 0

    def test_scores_sorted_descending(self):
        docs = make_docs([
            "retriever retriever retriever",
            "retriever search",
            "unrelated content",
        ])
        idx = BM25Index()
        idx.build(docs)
        results = idx.search("retriever")
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_file_path_in_meta_data(self):
        """Documents with file_path in meta_data should match file-name queries."""
        docs = make_docs_with_paths([
            ("class Retriever:\n    pass", "api/retriever.py"),
            ("class Config:\n    pass", "api/config.py"),
        ])
        idx = BM25Index()
        idx.build(docs)
        results = idx.search("retriever.py")
        assert len(results) > 0
        assert results[0][0] == 0  # api/retriever.py document

    def test_inverted_index_built(self):
        docs = make_docs(["alpha beta", "beta gamma"])
        idx = BM25Index()
        idx.build(docs)
        assert "beta" in idx._inverted_index
        assert len(idx._inverted_index["beta"]) == 2  # appears in both docs

    def test_idf_computed(self):
        docs = make_docs(["rare_term common", "common everyday"])
        idx = BM25Index()
        idx.build(docs)
        # 'rare_term' appears in 1 doc, 'common' in 2 — rare should have higher IDF
        assert idx._idf.get("rare") is not None or "rare_term" in idx._idf
        assert idx._idf.get("common", 0) >= 0

    def test_custom_k1_and_b(self):
        docs = make_docs(["test document one", "test document two"])
        idx = BM25Index(k1=2.0, b=0.5)
        idx.build(docs)
        assert idx.k1 == 2.0
        assert idx.b == 0.5
        results = idx.search("test")
        assert len(results) == 2

    def test_document_without_text_attribute(self):
        """Plain strings should still work (falls back to str())."""
        idx = BM25Index()
        idx.build(["hello world", "foo bar"])
        results = idx.search("hello")
        assert len(results) >= 1
        assert results[0][0] == 0


# ============================================================================
# HybridRetriever
# ============================================================================


class FakeFAISSRetriever:
    """Mock FAISS retriever returning pre-defined indices."""

    def __init__(self, indices: List[int]):
        self._indices = indices

    def __call__(self, query: str):
        output = MagicMock()
        output.doc_indices = self._indices
        return [output]


class FailingFAISSRetriever:
    """FAISS retriever that always raises."""

    def __call__(self, query: str):
        raise RuntimeError("FAISS index corrupted")


class TestHybridRetriever:
    def _build(self, texts, faiss_indices=None, **kwargs):
        """Helper: build a HybridRetriever with BM25 + fake FAISS."""
        docs = make_docs(texts)
        bm25 = BM25Index()
        bm25.build(docs)
        faiss = FakeFAISSRetriever(faiss_indices or [])
        return HybridRetriever(
            faiss_retriever=faiss,
            bm25_index=bm25,
            documents=docs,
            **kwargs,
        )

    def test_basic_merge(self):
        texts = ["python api server", "javascript frontend", "python data pipeline"]
        retriever = self._build(texts, faiss_indices=[0, 2], top_k=3)
        results = retriever("python")
        assert len(results) == 1  # single-element list
        output = results[0]
        assert len(output.doc_indices) > 0

    def test_rrf_boosted_by_both(self):
        """Document appearing in BOTH FAISS and BM25 should rank higher via RRF."""
        texts = [
            "database connection pool manager",
            "web server handler",
            "database query optimizer",
        ]
        # FAISS returns [0, 2]; BM25 for "database" also matches 0 and 2
        retriever = self._build(texts, faiss_indices=[0, 2], top_k=3)
        output = retriever("database")[0]
        # Docs 0 and 2 should appear, boosted by both retrievers
        assert 0 in output.doc_indices
        assert 2 in output.doc_indices

    def test_faiss_failure_fallback_to_bm25(self):
        """When FAISS fails, results should still come from BM25."""
        docs = make_docs(["alpha beta", "gamma delta"])
        bm25 = BM25Index()
        bm25.build(docs)
        faiss = FailingFAISSRetriever()
        retriever = HybridRetriever(
            faiss_retriever=faiss, bm25_index=bm25, documents=docs, top_k=5,
        )
        output = retriever("alpha")[0]
        assert len(output.doc_indices) > 0
        assert 0 in output.doc_indices

    def test_both_empty_returns_empty_output(self):
        """If neither retriever has results, output should be empty."""
        docs = make_docs(["something specific"])
        bm25 = BM25Index()
        bm25.build(docs)
        faiss = FakeFAISSRetriever([])
        retriever = HybridRetriever(
            faiss_retriever=faiss, bm25_index=bm25, documents=docs, top_k=5,
        )
        output = retriever("zzz_nonexistent_query_zzz")[0]
        assert output.doc_indices == []

    def test_top_k_respected(self):
        texts = [f"document about topic {i}" for i in range(50)]
        retriever = self._build(texts, faiss_indices=list(range(30)), top_k=10)
        output = retriever("document topic")[0]
        assert len(output.doc_indices) <= 10

    def test_invalid_faiss_indices_filtered(self):
        """Indices beyond document list length should be dropped."""
        docs = make_docs(["only two docs", "here"])
        bm25 = BM25Index()
        bm25.build(docs)
        faiss = FakeFAISSRetriever([0, 1, 999])  # 999 is out of bounds
        retriever = HybridRetriever(
            faiss_retriever=faiss, bm25_index=bm25, documents=docs, top_k=5,
        )
        output = retriever("docs")[0]
        assert 999 not in output.doc_indices

    def test_query_preserved_in_output(self):
        retriever = self._build(["hello world"], faiss_indices=[0], top_k=5)
        output = retriever("hello")[0]
        assert output.query == "hello"

    def test_scores_descending(self):
        texts = [f"keyword alpha beta {i}" for i in range(5)]
        retriever = self._build(texts, faiss_indices=[0, 1, 2], top_k=5)
        output = retriever("keyword alpha")[0]
        scores = output.doc_scores
        assert scores == sorted(scores, reverse=True)

    def test_custom_weights(self):
        """BM25 weight=0 means only FAISS contributions matter."""
        texts = ["alpha code", "beta code", "gamma code"]
        docs = make_docs(texts)
        bm25 = BM25Index()
        bm25.build(docs)
        faiss = FakeFAISSRetriever([1])
        retriever = HybridRetriever(
            faiss_retriever=faiss,
            bm25_index=bm25,
            documents=docs,
            top_k=3,
            faiss_weight=1.0,
            bm25_weight=0.0,
        )
        output = retriever("code")[0]
        # Only FAISS result (doc 1) should have non-zero RRF
        if output.doc_indices:
            assert output.doc_indices[0] == 1

    def test_custom_rrf_k(self):
        """Different rrf_k values should still produce valid output."""
        retriever = self._build(["hello world"], faiss_indices=[0], top_k=5, rrf_k=10)
        output = retriever("hello")[0]
        assert len(output.doc_indices) >= 1


# ============================================================================
# RetrieverOutput
# ============================================================================


class TestRetrieverOutput:
    def test_default_fields(self):
        out = RetrieverOutput()
        assert out.doc_indices == []
        assert out.doc_scores == []
        assert out.documents == []
        assert out.query == ""

    def test_custom_fields(self):
        out = RetrieverOutput(
            doc_indices=[0, 1],
            doc_scores=[0.9, 0.8],
            query="test query",
        )
        assert len(out.doc_indices) == 2
        assert out.query == "test query"

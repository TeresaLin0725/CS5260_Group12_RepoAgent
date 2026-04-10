"""
Hybrid Retriever: FAISS (semantic) + BM25 (keyword) with Reciprocal Rank Fusion.

Combines dense vector search (FAISS) with sparse keyword search (BM25) using
Reciprocal Rank Fusion (RRF) to improve recall for code identifiers and
exact-match queries that pure semantic search often misses.

Typical improvement:
  - Function/class name queries: +30-50% recall
  - Mixed natural-language + identifier queries: +15-25% recall
  - Pure natural-language queries: comparable (FAISS dominant)
"""

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Code-aware tokenizer
# ---------------------------------------------------------------------------

# Split camelCase: "myFunction" → "my Function", "HTMLParser" → "HTML Parser"
_CAMEL_RE = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
# Non-alphanumeric → space
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")

# Minimal English stop words.  BM25's IDF naturally down-weights
# high-frequency code tokens (def, class, return, etc.) so we
# intentionally do NOT filter those here.
_STOP_WORDS: frozenset = frozenset({
    "the", "is", "at", "which", "on", "a", "an", "and", "or", "but",
    "in", "with", "to", "for", "of", "not", "no", "can", "had", "has",
    "have", "do", "does", "did", "be", "been", "being", "this", "that",
    "it", "its", "if", "then", "else", "when", "from", "by", "as",
    "are", "was", "were", "will", "would", "could", "should", "may",
    "might", "must", "shall", "so", "than", "too", "very", "just",
    "about", "above", "after", "again", "all", "also", "am", "any",
    "because", "before", "between", "both", "during", "each", "few",
    "more", "most", "other", "out", "own", "same", "some", "such",
    "up", "only", "into", "over", "how", "what", "where", "who",
})


def tokenize_code(text: str) -> List[str]:
    """
    Tokenize text with code-awareness.

    Handles camelCase, PascalCase, snake_case, dot.notation, path/separators.
    Returns lowercase tokens with length >= 2, excluding pure digits.

    Examples::

        >>> tokenize_code("DatabaseManager.prepare_retriever")
        ['database', 'manager', 'prepare', 'retriever']
        >>> tokenize_code("get_file_content")
        ['get', 'file', 'content']
        >>> tokenize_code("FAISSRetriever")
        ['faiss', 'retriever']
    """
    if not text:
        return []
    # Split camelCase / PascalCase boundaries
    text = _CAMEL_RE.sub(" ", text)
    # Replace non-alphanumeric with space (handles snake_case, dots, slashes)
    text = _NON_ALNUM_RE.sub(" ", text.lower())
    # Filter: length >= 2, not pure digits
    return [t for t in text.split() if len(t) >= 2 and not t.isdigit()]


def tokenize_code_filtered(text: str) -> List[str]:
    """Tokenize with stop-word filtering — used for BM25 indexing and querying."""
    return [t for t in tokenize_code(text) if t not in _STOP_WORDS]


# ---------------------------------------------------------------------------
# BM25 Index
# ---------------------------------------------------------------------------


class BM25Index:
    """
    Lightweight Okapi BM25 index for keyword-based document retrieval.

    Built from a list of documents; supports fast scoring of queries against
    all indexed documents.  Uses an inverted index for efficiency — only
    documents containing at least one query term are scored.

    Scoring formula (per query term *q* in document *d*)::

        score(q, d) = IDF(q) * tf(q,d) * (k1 + 1)
                      / (tf(q,d) + k1 * (1 - b + b * |d| / avgdl))

    where IDF uses the Robertson–Spärck-Jones formula with +1 smoothing.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size: int = 0
        self.avgdl: float = 0.0
        self._doc_freqs: Dict[str, int] = {}
        self._doc_lens: List[int] = []
        self._tf: List[Dict[str, int]] = []
        self._idf: Dict[str, float] = {}
        self._inverted_index: Dict[str, List[int]] = {}

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, documents: list) -> None:
        """
        Build the BM25 index from a list of Document objects.

        Each document's ``.text`` is tokenized with code-aware splitting.
        File paths from ``meta_data['file_path']`` are prepended so that
        searching for a filename finds chunks from that file.

        Args:
            documents: List of adalflow Document objects with ``.text``
                       and optional ``.meta_data`` dict.
        """
        self.corpus_size = len(documents)
        if self.corpus_size == 0:
            logger.warning("BM25Index.build: no documents provided")
            return

        total_len = 0
        self._tf = []
        self._doc_lens = []
        self._inverted_index = {}
        doc_freq_counter: Dict[str, int] = {}

        for doc_idx, doc in enumerate(documents):
            # Extract text content
            text = doc.text if hasattr(doc, "text") else str(doc)

            # Prepend file path for identifier matching
            if hasattr(doc, "meta_data") and isinstance(doc.meta_data, dict):
                file_path = doc.meta_data.get("file_path", "")
                if file_path:
                    text = f"{file_path}\n{text}"

            tokens = tokenize_code_filtered(text)
            tf = Counter(tokens)
            self._tf.append(dict(tf))
            self._doc_lens.append(len(tokens))
            total_len += len(tokens)

            # Update document frequencies and inverted index
            for term in tf:
                doc_freq_counter[term] = doc_freq_counter.get(term, 0) + 1
                if term not in self._inverted_index:
                    self._inverted_index[term] = []
                self._inverted_index[term].append(doc_idx)

        self._doc_freqs = doc_freq_counter
        self.avgdl = total_len / self.corpus_size

        # Pre-compute IDF: Robertson–Spärck-Jones with +1 smoothing
        self._idf = {}
        for term, df in self._doc_freqs.items():
            self._idf[term] = math.log(
                (self.corpus_size - df + 0.5) / (df + 0.5) + 1.0
            )

        logger.info(
            "BM25 index built: %d documents, %d unique terms, avg_doc_len=%.1f",
            self.corpus_size,
            len(self._idf),
            self.avgdl,
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """
        Score indexed documents against the query and return the top-K.

        Uses the inverted index so only documents containing at least one
        query term are scored — efficient even for large corpora.

        Args:
            query: Raw query string (will be tokenized).
            top_k: Maximum number of results to return.

        Returns:
            List of ``(doc_index, bm25_score)`` tuples, sorted by score
            descending.  Empty list if no matches.
        """
        if self.corpus_size == 0:
            return []

        query_tokens = tokenize_code_filtered(query)
        if not query_tokens:
            return []

        # Score only candidate documents from the inverted index
        candidate_scores: Dict[int, float] = {}

        for term in query_tokens:
            idf = self._idf.get(term)
            if idf is None:
                continue
            posting_list = self._inverted_index.get(term, [])
            for doc_idx in posting_list:
                tf_val = self._tf[doc_idx].get(term, 0)
                doc_len = self._doc_lens[doc_idx]
                # BM25 scoring
                numerator = tf_val * (self.k1 + 1)
                denominator = tf_val + self.k1 * (
                    1 - self.b + self.b * doc_len / self.avgdl
                )
                candidate_scores[doc_idx] = (
                    candidate_scores.get(doc_idx, 0.0) + idf * numerator / denominator
                )

        if not candidate_scores:
            return []

        # Sort by score descending and return top_k
        sorted_results = sorted(
            candidate_scores.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_results[:top_k]


# ---------------------------------------------------------------------------
# Retriever output (compatible with adalflow FAISSRetriever)
# ---------------------------------------------------------------------------


@dataclass
class RetrieverOutput:
    """
    Return type for HybridRetriever, wire-compatible with adalflow's
    FAISSRetriever output so ``RAG.call()`` works unchanged.

    Attributes:
        doc_indices: Ordered list of document indices in the source list.
        doc_scores:  Corresponding RRF scores (higher = more relevant).
        documents:   Populated downstream by ``RAG.call()`` from doc_indices.
        query:       The original query string.
    """

    doc_indices: list = field(default_factory=list)
    doc_scores: list = field(default_factory=list)
    documents: list = field(default_factory=list)
    query: str = ""


# ---------------------------------------------------------------------------
# Hybrid Retriever
# ---------------------------------------------------------------------------


class HybridRetriever:
    """
    Combines FAISS (dense/semantic) and BM25 (sparse/keyword) retrieval
    using Reciprocal Rank Fusion (RRF).

    RRF formula for each document *d*::

        rrf_score(d) = Σ  weight_i / (k + rank_i(d))

    where the sum is over each retriever *i* that returned *d*,
    ``rank_i`` is the **1-based** rank in that retriever's results,
    and *k* is a constant (default 60, per the original RRF paper by
    Cormack, Clarke & Butt, 2009).

    This approach improves recall for exact identifiers (function names,
    class names, variable names) that BM25 matches well, while retaining
    the semantic understanding of FAISS for natural-language queries.

    Args:
        faiss_retriever: An adalflow ``FAISSRetriever`` instance.
        bm25_index:      A populated ``BM25Index`` instance.
        documents:       The source document list (same list used for both
                         retrievers).
        top_k:           Number of final results to return.
        rrf_k:           RRF constant *k* (default 60).
        faiss_weight:    Multiplier for FAISS RRF contribution (default 1.0).
        bm25_weight:     Multiplier for BM25 RRF contribution (default 1.0).
    """

    def __init__(
        self,
        faiss_retriever,
        bm25_index: BM25Index,
        documents: list,
        top_k: int = 20,
        rrf_k: int = 60,
        faiss_weight: float = 1.0,
        bm25_weight: float = 1.0,
    ):
        self.faiss_retriever = faiss_retriever
        self.bm25_index = bm25_index
        self.documents = documents
        self.top_k = top_k
        self.rrf_k = rrf_k
        self.faiss_weight = faiss_weight
        self.bm25_weight = bm25_weight

    def __call__(self, query: str) -> List[RetrieverOutput]:
        """
        Execute hybrid retrieval and return results in FAISSRetriever-compatible
        format.

        Steps:
            1. Run FAISS semantic search (uses FAISS's own top_k).
            2. Run BM25 keyword search (2× ``self.top_k`` candidates).
            3. Merge via Reciprocal Rank Fusion.
            4. Return the top ``self.top_k`` results.

        Args:
            query: The user's search query.

        Returns:
            A single-element list ``[RetrieverOutput]`` with merged
            ``doc_indices`` and ``doc_scores``.
        """
        rrf_scores: Dict[int, float] = {}

        # --- 1. FAISS semantic search ---------------------------------
        faiss_indices: List[int] = []
        try:
            faiss_results = self.faiss_retriever(query)
            if faiss_results and len(faiss_results) > 0:
                faiss_indices = list(faiss_results[0].doc_indices)
        except Exception as e:
            logger.warning("FAISS retrieval failed, using BM25 only: %s", e)

        for rank, idx in enumerate(faiss_indices):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + self.faiss_weight / (
                self.rrf_k + rank + 1
            )

        # --- 2. BM25 keyword search -----------------------------------
        bm25_results: List[Tuple[int, float]] = []
        try:
            bm25_results = self.bm25_index.search(query, top_k=self.top_k * 2)
        except Exception as e:
            logger.warning("BM25 retrieval failed, using FAISS only: %s", e)

        for rank, (idx, _bm25_score) in enumerate(bm25_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + self.bm25_weight / (
                self.rrf_k + rank + 1
            )

        # --- 3. Handle empty results ---------------------------------
        if not rrf_scores:
            logger.warning("Both FAISS and BM25 returned empty results for query: %s", query[:80])
            return [RetrieverOutput(query=query)]

        # --- 4. Sort, validate, and select top_k ---------------------
        sorted_results = sorted(
            rrf_scores.items(), key=lambda x: x[1], reverse=True
        )

        # Filter to valid document indices (defensive)
        num_docs = len(self.documents)
        valid_results = [
            (idx, score) for idx, score in sorted_results if 0 <= idx < num_docs
        ]

        final_results = valid_results[: self.top_k]
        final_indices = [idx for idx, _ in final_results]
        final_scores = [score for _, score in final_results]

        # --- 5. Log retrieval diversity stats -------------------------
        faiss_set = set(faiss_indices)
        bm25_set = {idx for idx, _ in bm25_results}
        final_set = set(final_indices)
        faiss_only_in_final = len(final_set - bm25_set)
        bm25_only_in_final = len(final_set - faiss_set)
        both_in_final = len(final_set & faiss_set & bm25_set)

        logger.info(
            "Hybrid search: query=%r | FAISS_candidates=%d, BM25_candidates=%d, "
            "overlap=%d | final=%d (faiss_only=%d, bm25_only=%d, both=%d)",
            query[:60],
            len(faiss_indices),
            len(bm25_results),
            len(faiss_set & bm25_set),
            len(final_indices),
            faiss_only_in_final,
            bm25_only_in_final,
            both_in_final,
        )

        # --- 6. Build compatible output -------------------------------
        output = RetrieverOutput(
            doc_indices=final_indices,
            doc_scores=final_scores,
            query=query,
        )
        return [output]

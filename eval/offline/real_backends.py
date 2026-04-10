"""
Real backend adapters for offline evaluation.

Provides factory functions that create real LLM / Intent Classifier / RAG
retriever callables from the project's production components, so the
benchmark runner can evaluate against actual model behaviour.

Usage:
    python -m eval.offline.run_benchmark --backend real --provider openai --model gpt-4o
    python -m eval.offline.run_benchmark --backend real --provider ollama --model qwen3:4b
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# LLM callable factory
# -----------------------------------------------------------------------

def create_real_llm(
    provider: str = "openai",
    model: Optional[str] = None,
) -> Callable[[str], Awaitable[str]]:
    """Create a real ``async (prompt: str) -> str`` LLM callable.

    Reuses the project's ``create_llm_callable`` from
    :pymod:`api.agent.llm_utils`, which already handles all providers
    (Google, OpenAI, Ollama, OpenRouter, Bedrock, Azure, DashScope).

    Parameters
    ----------
    provider : str
        One of "openai", "google", "ollama", "openrouter", "bedrock",
        "azure", "dashscope".
    model : str or None
        Model name; ``None`` uses the provider's default from
        ``generator.json``.

    Returns
    -------
    Callable[[str], Awaitable[str]]
        An async function that sends a prompt and returns the full
        response text.
    """
    from api.agent.llm_utils import create_llm_callable

    if provider == "ollama":
        from adalflow.components.model_client.ollama_client import OllamaClient

        model_name = model or "qwen3:8b"
        model_instance = OllamaClient()
        model_kwargs = {
            "model": model_name,
            "options": {
                "temperature": 0.7,
                "top_p": 0.8,
                "num_ctx": 4096,
            },
        }
    elif provider == "google":
        import google.generativeai as genai

        from api.config import get_model_config

        config = get_model_config(provider, model)
        model_kwargs = config["model_kwargs"]
        google_api_key = os.environ.get("GOOGLE_API_KEY")
        if google_api_key:
            genai.configure(api_key=google_api_key)
        model_instance = genai.GenerativeModel(model_kwargs["model"])
    else:
        from api.config import get_model_config

        config = get_model_config(provider, model)
        model_client_class = config["model_client"]
        model_kwargs = config["model_kwargs"]
        model_instance = model_client_class()

    llm_fn = create_llm_callable(
        provider=provider,
        model=model_instance,
        model_kwargs=model_kwargs,
    )

    logger.info(
        "Created real LLM callable: provider=%s model=%s",
        provider, model_kwargs.get("model"),
    )
    return llm_fn


# -----------------------------------------------------------------------
# Intent classifier factory
# -----------------------------------------------------------------------

def create_real_intent_classifier(
    mode: str = "llm",
    provider: str = "openai",
    model: Optional[str] = None,
    confidence_threshold: float = 0.6,
) -> Callable[[str], Awaitable[Any]]:
    """Create a real intent classifier callable.

    Parameters
    ----------
    mode : str
        "llm" for :class:`LLMIntentClassifier` (Plan A, needs LLM),
        "embedding" for :class:`EmbeddingIntentClassifier` (Plan B, needs embedder).
    provider : str
        LLM provider (used when ``mode="llm"``).
    model : str or None
        Model name (used when ``mode="llm"``).
    confidence_threshold : float
        Minimum confidence to accept a classification.

    Returns
    -------
    Callable[[str], Awaitable[IntentResult | None]]
        ``async (query: str) -> IntentResult``
    """
    if mode == "llm":
        from api.agent.intent_classifier import LLMIntentClassifier

        llm_fn = create_real_llm(provider=provider, model=model)
        classifier = LLMIntentClassifier(
            llm_fn=llm_fn,
            confidence_threshold=confidence_threshold,
        )
        logger.info(
            "Created real LLM intent classifier: provider=%s model=%s threshold=%.2f",
            provider, model, confidence_threshold,
        )
        return classifier.classify

    elif mode == "embedding":
        from api.agent.intent_classifier import EmbeddingIntentClassifier
        from api.tools.embedder import get_embedder

        embedder = get_embedder()

        async def embed_fn(texts: List[str]) -> List[List[float]]:
            """Adapter: adalflow Embedder -> List[List[float]]."""
            result = embedder(input=texts)
            return [e.embedding for e in result.data]

        classifier = EmbeddingIntentClassifier(
            embed_fn=embed_fn,
            confidence_threshold=confidence_threshold,
        )
        logger.info(
            "Created real embedding intent classifier: threshold=%.2f",
            confidence_threshold,
        )
        return classifier.classify

    else:
        raise ValueError(f"Unknown intent classifier mode: {mode!r}. Use 'llm' or 'embedding'.")


# -----------------------------------------------------------------------
# RAG retriever factory
# -----------------------------------------------------------------------

def create_real_retriever(
    repo_url: str,
    repo_type: str = "github",
    access_token: Optional[str] = None,
    provider: str = "openai",
    model: Optional[str] = None,
    top_k: int = 10,
) -> Callable[[str], Awaitable[List[str]]]:
    """Create a real RAG retriever callable.

    Instantiates a :class:`RAG` object, prepares the FAISS + BM25
    hybrid retriever for the given repository, and returns an async
    function that retrieves ranked file paths.

    Parameters
    ----------
    repo_url : str
        Repository URL (e.g. "https://github.com/user/repo").
    repo_type : str
        "github", "gitlab", or "bitbucket".
    access_token : str or None
        Access token for private repos.
    provider : str
        LLM provider (for RAG instance).
    model : str or None
        Model name.
    top_k : int
        Number of documents to retrieve per query.

    Returns
    -------
    Callable[[str], Awaitable[List[str]]]
        ``async (query: str) -> [file_path, ...]``
    """
    from api.rag import RAG

    rag = RAG(provider=provider, model=model)
    rag.prepare_retriever(
        repo_url_or_path=repo_url,
        type=repo_type,
        access_token=access_token,
    )
    logger.info(
        "Created real RAG retriever: repo=%s provider=%s",
        repo_url, provider,
    )

    async def retriever_fn(query: str) -> List[str]:
        """Retrieve ranked file paths for a query."""
        results = rag(query, language="en")
        files: List[str] = []
        if results and results[0].documents:
            for doc in results[0].documents[:top_k]:
                path = doc.meta_data.get("file_path", "")
                if path and path not in files:
                    files.append(path)
        return files

    return retriever_fn


# -----------------------------------------------------------------------
# Local BM25 retriever (no embedding API required)
# -----------------------------------------------------------------------

#: Directories to skip when indexing a local repo
_SKIP_DIRS = frozenset({
    ".git", "__pycache__", "node_modules", ".venv", "venv", ".tox",
    "dist", "build", ".eggs", ".mypy_cache", ".pytest_cache",
    "docs", "docs_src", "examples", "benchmarks", "scripts",
    ".github", "tests", "test",
})

#: Minimum file size to index (bytes) — skip empty __init__.py stubs
_MIN_FILE_BYTES = 200


def create_local_bm25_retriever(
    repo_dir: Path,
    top_k: int = 10,
    extensions: Tuple[str, ...] = (".py",),
) -> Callable[[str], Awaitable[List[str]]]:
    """Build a BM25 index from a local repository and return an async retriever.

    No embedding API or LLM is required — retrieval is purely keyword-based
    using :class:`api.retriever.BM25Index`.  Useful for offline evaluation
    without any external credentials.

    Parameters
    ----------
    repo_dir : Path
        Root directory of the cloned repository (e.g. ``.repos/pallets_flask``).
    top_k : int
        Number of top-ranked file paths to return per query.
    extensions : tuple of str
        File extensions to index (default: Python files only).

    Returns
    -------
    Callable[[str], Awaitable[List[str]]]
        ``async (query: str) -> [relative_file_path, ...]``

    Notes
    -----
    The returned paths are **relative to** ``repo_dir``, matching the
    ``golden_files`` format used in the RAG test fixtures.
    """
    from api.retriever import BM25Index

    repo_dir = Path(repo_dir)
    if not repo_dir.exists():
        raise FileNotFoundError(f"Repo directory not found: {repo_dir}")

    # ------------------------------------------------------------------
    # Collect source files
    # ------------------------------------------------------------------
    file_paths: List[Path] = []
    for root, dirs, files in os.walk(repo_dir):
        dirs[:] = [
            d for d in dirs
            if d not in _SKIP_DIRS and not d.startswith(".")
        ]
        for fname in files:
            if not any(fname.endswith(ext) for ext in extensions):
                continue
            fpath = Path(root) / fname
            try:
                if fpath.stat().st_size < _MIN_FILE_BYTES:
                    continue
            except OSError:
                continue
            file_paths.append(fpath)

    if not file_paths:
        raise RuntimeError(f"No source files found in {repo_dir}")

    # ------------------------------------------------------------------
    # Build Document objects and split with code-aware chunker
    # ------------------------------------------------------------------
    from adalflow.core.types import Document as AdalDocument
    from api.code_splitter import CodeAwareTextSplitter
    from api.config import configs as app_configs

    raw_documents: List[AdalDocument] = []
    for fpath in file_paths:
        rel = fpath.relative_to(repo_dir).as_posix()
        try:
            content = fpath.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        ext = fpath.suffix.lower()
        is_code = ext in (".py", ".js", ".ts", ".java", ".go", ".rs", ".jsx",
                          ".tsx", ".cpp", ".c", ".h", ".hpp", ".cs", ".swift", ".php")
        raw_documents.append(AdalDocument(
            text=content,
            meta_data={"file_path": rel, "type": ext.lstrip("."), "is_code": is_code},
        ))

    splitter_cfg = app_configs.get("text_splitter", {})
    splitter = CodeAwareTextSplitter(
        chunk_size=splitter_cfg.get("chunk_size", 200),
        chunk_overlap=splitter_cfg.get("chunk_overlap", 30),
    )
    split_docs = splitter(raw_documents)

    # Build parallel lookup: split_doc index → relative file path
    rel_paths: List[str] = [
        d.meta_data.get("file_path", "") if hasattr(d, "meta_data") and d.meta_data else ""
        for d in split_docs
    ]

    # ------------------------------------------------------------------
    # Build BM25 index (synchronous, done once at startup)
    # ------------------------------------------------------------------
    bm25 = BM25Index()
    bm25.build(split_docs)

    logger.info(
        "LocalBM25Retriever ready: repo=%s  files=%d  chunks=%d  top_k=%d",
        repo_dir.name, len(raw_documents), len(split_docs), top_k,
    )

    # ------------------------------------------------------------------
    # Async retriever function
    # ------------------------------------------------------------------
    async def retriever_fn(query: str) -> List[str]:
        results = bm25.search(query, top_k=top_k * 2)
        # Deduplicate: multiple chunks from the same file → keep best rank
        files: List[str] = []
        for idx, _score in results:
            fp = rel_paths[idx] if idx < len(rel_paths) else ""
            if fp and fp not in files:
                files.append(fp)
            if len(files) >= top_k:
                break
        return files

    return retriever_fn


def create_local_hybrid_retriever(
    repo_dir: Path,
    top_k: int = 10,
    extensions: Tuple[str, ...] = (".py",),
    embedder_type: str = "openai",
) -> Callable[[str], Awaitable[List[str]]]:
    """Build a FAISS + BM25 hybrid index from a local repository.

    Unlike :func:`create_local_bm25_retriever` which is keyword-only,
    this creates a full hybrid retriever identical to the production
    pipeline: FAISS semantic embeddings + BM25 keyword search merged
    via Reciprocal Rank Fusion (RRF).

    Requires an embedding API key (``OPENAI_API_KEY``, ``GOOGLE_API_KEY``,
    etc.) to compute embeddings.

    Parameters
    ----------
    repo_dir : Path
        Root directory of the cloned repository.
    top_k : int
        Number of top-ranked file paths to return per query.
    extensions : tuple of str
        File extensions to index (default: Python files only).
    embedder_type : str
        Embedder provider: "openai", "google", "bedrock" (default: "openai").

    Returns
    -------
    Callable[[str], Awaitable[List[str]]]
        ``async (query: str) -> [relative_file_path, ...]``
    """
    from api.retriever import BM25Index, HybridRetriever
    from api.code_splitter import CodeAwareTextSplitter
    from api.tools.embedder import get_embedder
    from adalflow.core.types import Document
    from adalflow.components.retriever.faiss_retriever import FAISSRetriever
    from adalflow.components.data_process import ToEmbeddings
    from api.config import configs

    repo_dir = Path(repo_dir)
    if not repo_dir.exists():
        raise FileNotFoundError(f"Repo directory not found: {repo_dir}")

    # ------------------------------------------------------------------
    # Collect source files
    # ------------------------------------------------------------------
    file_paths: List[Path] = []
    for root, dirs, files in os.walk(repo_dir):
        dirs[:] = [
            d for d in dirs
            if d not in _SKIP_DIRS and not d.startswith(".")
        ]
        for fname in files:
            if not any(fname.endswith(ext) for ext in extensions):
                continue
            fpath = Path(root) / fname
            try:
                if fpath.stat().st_size < _MIN_FILE_BYTES:
                    continue
            except OSError:
                continue
            file_paths.append(fpath)

    if not file_paths:
        raise RuntimeError(f"No source files found in {repo_dir}")

    # ------------------------------------------------------------------
    # Read files into Document objects
    # ------------------------------------------------------------------
    raw_documents: List[Document] = []
    for fpath in file_paths:
        rel = fpath.relative_to(repo_dir).as_posix()
        try:
            content = fpath.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        ext = fpath.suffix.lower()
        is_code = ext in (".py", ".js", ".ts", ".java", ".go", ".rs", ".jsx",
                          ".tsx", ".cpp", ".c", ".h", ".hpp", ".cs", ".swift", ".php")
        raw_documents.append(Document(
            text=content,
            meta_data={
                "file_path": rel,
                "type": ext.lstrip("."),
                "is_code": is_code,
            },
        ))

    # ------------------------------------------------------------------
    # Split with code-aware chunker
    # ------------------------------------------------------------------
    splitter_cfg = configs.get("text_splitter", {})
    splitter = CodeAwareTextSplitter(
        chunk_size=splitter_cfg.get("chunk_size", 200),
        chunk_overlap=splitter_cfg.get("chunk_overlap", 30),
    )
    split_docs = splitter(raw_documents)
    logger.info(
        "Hybrid retriever: %d files → %d chunks after code-aware splitting",
        len(raw_documents), len(split_docs),
    )

    # ------------------------------------------------------------------
    # Compute embeddings
    # ------------------------------------------------------------------
    embedder = get_embedder(embedder_type=embedder_type)
    embedder_transformer = ToEmbeddings(
        embedder=embedder,
        batch_size=configs.get("embedder", {}).get("batch_size", 100),
    )
    embedded_docs = embedder_transformer(split_docs)
    logger.info("Computed embeddings for %d chunks", len(embedded_docs))

    # Filter out docs with no/invalid vectors
    valid_docs = [d for d in embedded_docs if hasattr(d, "vector") and d.vector is not None]
    if not valid_docs:
        raise RuntimeError("No valid embeddings produced; check your API key and embedder config")
    logger.info("Valid embedded chunks: %d / %d", len(valid_docs), len(embedded_docs))

    # ------------------------------------------------------------------
    # Build FAISS retriever
    # ------------------------------------------------------------------
    faiss_retriever = FAISSRetriever(
        top_k=top_k,
        embedder=embedder,
        documents=valid_docs,
        document_map_func=lambda doc: doc.vector,
    )

    # ------------------------------------------------------------------
    # Build BM25 index
    # ------------------------------------------------------------------
    bm25 = BM25Index()
    bm25.build(valid_docs)

    # ------------------------------------------------------------------
    # Combine into HybridRetriever
    # ------------------------------------------------------------------
    hybrid = HybridRetriever(
        faiss_retriever=faiss_retriever,
        bm25_index=bm25,
        documents=valid_docs,
        top_k=top_k,
    )

    # Build file_path lookup from valid_docs
    doc_file_paths = [
        d.meta_data.get("file_path", "") if hasattr(d, "meta_data") and d.meta_data else ""
        for d in valid_docs
    ]

    logger.info(
        "LocalHybridRetriever ready: repo=%s  chunks=%d  top_k=%d",
        repo_dir.name, len(valid_docs), top_k,
    )

    # ------------------------------------------------------------------
    # Async retriever function
    # ------------------------------------------------------------------
    async def retriever_fn(query: str) -> List[str]:
        results = hybrid(query)
        files: List[str] = []
        if results and results[0].doc_indices:
            for idx in results[0].doc_indices:
                if 0 <= idx < len(doc_file_paths):
                    fp = doc_file_paths[idx]
                    if fp and fp not in files:
                        files.append(fp)
        return files[:top_k]

    return retriever_fn


# -----------------------------------------------------------------------
# ReAct tools factory
# -----------------------------------------------------------------------

def create_real_tools(
    repo_url: str,
    repo_type: str = "github",
    access_token: Optional[str] = None,
    provider: str = "openai",
    model: Optional[str] = None,
    language: str = "en",
) -> Dict[str, Callable[[str], Awaitable[str]]]:
    """Create real ReAct tools from a prepared RAG instance.

    Returns the same tool dictionary that production uses:
    ``rag_search``, ``read_file``, ``list_repo_files``, etc.

    Parameters
    ----------
    repo_url : str
        Repository URL.
    repo_type : str
        "github", "gitlab", or "bitbucket".
    access_token : str or None
        Access token for private repos.
    provider : str
        LLM / Embedder provider.
    model : str or None
        Model name.
    language : str
        Language code for retrieval.

    Returns
    -------
    Dict[str, Callable]
        Tool name -> async callable.
    """
    from api.rag import RAG
    from api.agent.tools.search_tools import build_react_tools

    rag = RAG(provider=provider, model=model)
    rag.prepare_retriever(
        repo_url_or_path=repo_url,
        type=repo_type,
        access_token=access_token,
    )
    tools = build_react_tools(
        rag_instance=rag,
        language=language,
        repo_url=repo_url,
        repo_type=repo_type,
        token=access_token,
    )
    logger.info(
        "Created real ReAct tools: %s (repo=%s)",
        list(tools.keys()), repo_url,
    )
    return tools


# -----------------------------------------------------------------------
# Convenience: all-in-one factory
# -----------------------------------------------------------------------

def create_all_real_backends(
    provider: str = "openai",
    model: Optional[str] = None,
    repo_url: Optional[str] = None,
    repo_type: str = "github",
    access_token: Optional[str] = None,
    intent_mode: str = "llm",
    language: str = "en",
) -> Dict[str, Any]:
    """Create all real backend components for a full benchmark run.

    Returns
    -------
    dict with keys:
        - ``llm_fn``: async LLM callable
        - ``judge_llm_fn``: async LLM callable for LLM-as-Judge
        - ``classifier_fn``: async intent classifier callable
        - ``retriever_fn``: async retriever callable (or None)
        - ``tools``: dict of ReAct tool callables (or None)
    """
    llm_fn = create_real_llm(provider=provider, model=model)

    # Use the same LLM as judge (could be a separate, stronger model)
    judge_llm_fn = llm_fn

    classifier_fn = create_real_intent_classifier(
        mode=intent_mode,
        provider=provider,
        model=model,
    )

    retriever_fn = None
    tools = None
    if repo_url:
        retriever_fn = create_real_retriever(
            repo_url=repo_url,
            repo_type=repo_type,
            access_token=access_token,
            provider=provider,
            model=model,
        )
        tools = create_real_tools(
            repo_url=repo_url,
            repo_type=repo_type,
            access_token=access_token,
            provider=provider,
            model=model,
            language=language,
        )

    return {
        "llm_fn": llm_fn,
        "judge_llm_fn": judge_llm_fn,
        "classifier_fn": classifier_fn,
        "retriever_fn": retriever_fn,
        "tools": tools,
    }

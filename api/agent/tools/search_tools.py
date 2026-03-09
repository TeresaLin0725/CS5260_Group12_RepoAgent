"""
Search and file-reading tools for the ReAct agent.

Provides factory functions that create async tool callables
bound to a specific RAG instance / repository context.
"""

import logging
from typing import Awaitable, Callable, Dict, Optional

logger = logging.getLogger(__name__)


def build_react_tools(
    rag_instance,
    language: str = "en",
    repo_url: Optional[str] = None,
    repo_type: Optional[str] = None,
    token: Optional[str] = None,
) -> Dict[str, Callable[[str], Awaitable[str]]]:
    """
    Build the tool dictionary for a ReAct run.

    Args:
        rag_instance: A prepared ``RAG`` object (retriever already loaded).
        language: Language code for retrieval.
        repo_url: Repository URL (for file reads).
        repo_type: Repository type (github/gitlab/bitbucket).
        token: Access token (for private repos).

    Returns:
        Dict mapping tool names to async callables ``(input: str) -> str``.
    """

    async def rag_search(query: str) -> str:
        """Semantic search over the codebase via RAG."""
        try:
            results = rag_instance(query, language=language)
            if results and results[0].documents:
                docs = results[0].documents
                parts = []
                for doc in docs[:6]:
                    path = doc.meta_data.get("file_path", "unknown")
                    parts.append(f"### {path}\n{doc.text}")
                return "\n\n---\n\n".join(parts)
            return "No relevant documents found."
        except Exception as exc:
            logger.error("rag_search tool error: %s", exc)
            return f"Search error: {exc}"

    tools: Dict[str, Callable[[str], Awaitable[str]]] = {
        "rag_search": rag_search,
    }

    # Only add read_file when we have the repo information it needs.
    if repo_url:

        async def read_file(file_path: str) -> str:
            """Read a file's content from the repository."""
            try:
                from api.data_pipeline import get_file_content

                content = get_file_content(
                    repo_url, file_path.strip(), repo_type or "github", token
                )
                if content:
                    return content
                return f"File '{file_path}' not found or empty."
            except Exception as exc:
                logger.error("read_file tool error for '%s': %s", file_path, exc)
                return f"File read error: {exc}"

        tools["read_file"] = read_file

    return tools

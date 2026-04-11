"""
Search and file-reading tools for the ReAct agent.

Provides factory functions that create async tool callables
bound to a specific RAG instance / repository context.
"""

import logging
import os
import re
from typing import Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


def _is_wrapped_in_quotes(value: str) -> bool:
    value = (value or "").strip()
    return len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}


def _strip_wrapped_quotes(value: str) -> str:
    cleaned = (value or "").strip()
    while _is_wrapped_in_quotes(cleaned):
        cleaned = cleaned[1:-1].strip()
    return cleaned


def _normalize_repo_browse_path(path: str) -> str:
    cleaned = _strip_wrapped_quotes(path).replace("\\", "/").strip()
    if cleaned in {"", ".", "./", "/", "*", "*.*"}:
        return ""
    return cleaned.lstrip("/")


def _normalize_search_pattern(pattern: str) -> str:
    cleaned = (pattern or "").strip()
    if not cleaned:
        return ""

    raw_parts = re.split(r"\s+OR\s+", cleaned, flags=re.IGNORECASE)
    if len(raw_parts) > 1 and all(_is_wrapped_in_quotes(part) for part in raw_parts):
        return "|".join(re.escape(_strip_wrapped_quotes(part)) for part in raw_parts)

    return _strip_wrapped_quotes(cleaned)


def build_react_tools(
    rag_instance,
    language: str = "en",
    repo_url: Optional[str] = None,
    repo_type: Optional[str] = None,
    token: Optional[str] = None,
    user_id: Optional[str] = None,
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
                cleaned = file_path.strip().strip("'\"").lstrip("/")

                # Prefer reading from the local clone when available
                if local_repo_dir:
                    full_path = os.path.normpath(os.path.join(local_repo_dir, cleaned))
                    if full_path.startswith(os.path.normpath(local_repo_dir)) and os.path.isfile(full_path):
                        with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read()
                        if len(content) > 50000:
                            content = content[:50000] + "\n... [truncated]"
                        return content if content else f"File '{cleaned}' is empty."

                # Fallback to API when local file is not available
                from api.data_pipeline import get_file_content

                content = get_file_content(
                    repo_url, cleaned, repo_type or "github", token
                )
                if content:
                    return content
                return f"File '{cleaned}' not found or empty."
            except Exception as exc:
                logger.error("read_file tool error for '%s': %s", file_path, exc)
                return f"File read error: {exc}"

        tools["read_file"] = read_file

    # ------------------------------------------------------------------
    # list_repo_files: browse the local repo directory structure
    # ------------------------------------------------------------------
    local_repo_dir = _get_local_repo_dir(rag_instance)
    if local_repo_dir:

        async def list_repo_files(path: str) -> str:
            """List files and directories under a path in the repository."""
            try:
                normalized_path = _normalize_repo_browse_path(path)
                target = os.path.normpath(os.path.join(local_repo_dir, normalized_path))
                # Safety: prevent path traversal outside repo
                if not target.startswith(os.path.normpath(local_repo_dir)):
                    return "Error: path is outside the repository."

                if not os.path.exists(target):
                    return f"Path '{path}' does not exist in the repository."

                if os.path.isfile(target):
                    rel = os.path.relpath(target, local_repo_dir).replace("\\", "/")
                    return f"(file) {rel}"

                entries = []
                for name in sorted(os.listdir(target)):
                    full = os.path.join(target, name)
                    rel = os.path.relpath(full, local_repo_dir).replace("\\", "/")
                    if os.path.isdir(full):
                        entries.append(f"  {rel}/")
                    else:
                        entries.append(f"  {rel}")

                if not entries:
                    return f"Directory '{path}' is empty."

                display_path = normalized_path or "/"
                header = f"Contents of {display_path}  ({len(entries)} items):\n"
                output = header + "\n".join(entries)
                if len(output) > 8000:
                    output = output[:8000] + "\n... [truncated]"
                return output
            except Exception as exc:
                logger.error("list_repo_files error for '%s': %s", path, exc)
                return f"List error: {exc}"

        tools["list_repo_files"] = list_repo_files

    # ------------------------------------------------------------------
    # code_grep: exact text / regex search across the codebase
    # ------------------------------------------------------------------
    if local_repo_dir:

        async def code_grep(pattern: str) -> str:
            """Search for an exact string or regex pattern across all files in the repository."""
            try:
                raw_pattern = pattern
                pattern = _normalize_search_pattern(pattern)
                if not pattern:
                    return "Error: empty search pattern."

                try:
                    compiled = re.compile(pattern, re.IGNORECASE)
                except re.error:
                    # Fall back to literal search if pattern is not valid regex
                    compiled = re.compile(re.escape(pattern), re.IGNORECASE)

                matches = []
                max_matches = 30
                _SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".tox", "dist", "build"}
                _BINARY_EXT = {".png", ".jpg", ".jpeg", ".gif", ".ico", ".woff", ".woff2", ".ttf", ".eot",
                               ".pdf", ".zip", ".tar", ".gz", ".exe", ".dll", ".so", ".pyc", ".pkl"}

                for dirpath, dirnames, filenames in os.walk(local_repo_dir):
                    # Prune skipped directories
                    dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
                    for fname in filenames:
                        if os.path.splitext(fname)[1].lower() in _BINARY_EXT:
                            continue
                        fpath = os.path.join(dirpath, fname)
                        rel = os.path.relpath(fpath, local_repo_dir).replace("\\", "/")
                        try:
                            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                                for lineno, line in enumerate(f, 1):
                                    if compiled.search(line):
                                        snippet = line.rstrip()[:120]
                                        matches.append(f"{rel}:{lineno}: {snippet}")
                                        if len(matches) >= max_matches:
                                            break
                        except (OSError, UnicodeDecodeError):
                            continue
                        if len(matches) >= max_matches:
                            break
                    if len(matches) >= max_matches:
                        break

                if not matches:
                    return f"No matches found for '{pattern}'."

                header = f"Found {len(matches)} match(es) for '{pattern}':\n"
                output = header + "\n".join(matches)
                if len(matches) >= max_matches:
                    output += f"\n... [limited to {max_matches} results]"
                return output
            except Exception as exc:
                logger.error("code_grep error for '%s': %s", raw_pattern, exc)
                return f"Grep error: {exc}"

        tools["code_grep"] = code_grep

    # ------------------------------------------------------------------
    # read_function: extract a specific function/class definition
    # ------------------------------------------------------------------
    if local_repo_dir:

        async def read_function(query: str) -> str:
            """Read a specific function or class definition from a file.

            Input format: 'file_path::target_name'
            Examples:
              - 'src/main.py::process_request'
              - 'api/rag.py::RAG'
              - 'utils/helpers.py::validate_input'

            Returns the complete definition (including decorators and docstring)
            so you don't need to read the entire file.
            """
            try:
                if "::" not in query:
                    return "Error: use format 'file_path::function_or_class_name' (e.g. 'src/main.py::my_func')"
                file_path, target = query.split("::", 1)
                file_path = _strip_wrapped_quotes(file_path).lstrip("/")
                target = _strip_wrapped_quotes(target)
                full_path = os.path.normpath(os.path.join(local_repo_dir, file_path))
                if not full_path.startswith(os.path.normpath(local_repo_dir)):
                    return "Error: path is outside the repository."
                if not os.path.isfile(full_path):
                    return f"File '{file_path}' not found."

                with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()

                # Find the target definition using regex patterns
                results = _extract_definition(lines, target)
                if results:
                    header = f"### {file_path}::{target} (lines {results['start']}-{results['end']})\n"
                    return header + results["text"]
                return f"Definition '{target}' not found in '{file_path}'. Try code_grep to locate it."
            except Exception as exc:
                logger.error("read_function error for '%s': %s", query, exc)
                return f"Read function error: {exc}"

        tools["read_function"] = read_function

    # ------------------------------------------------------------------
    # find_references: find all usages of a function/class/identifier
    # ------------------------------------------------------------------
    if local_repo_dir:

        async def find_references(identifier: str) -> str:
            """Find all files that import, call, or reference a specific function, class, or identifier.

            Input: an identifier name (e.g. 'ReActRunner', 'build_react_tools', 'handle_websocket_chat').
            Returns a list of files and lines where the identifier appears, grouped by usage type
            (import, call, definition).
            """
            try:
                identifier = _strip_wrapped_quotes(identifier)
                if not identifier:
                    return "Error: empty identifier."

                _SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv",
                              ".tox", "dist", "build", ".next"}
                _BINARY_EXT = {".png", ".jpg", ".jpeg", ".gif", ".ico", ".woff", ".woff2",
                               ".ttf", ".eot", ".pdf", ".zip", ".tar", ".gz", ".exe",
                               ".dll", ".so", ".pyc", ".pkl"}

                imports: List[str] = []
                calls: List[str] = []
                definitions: List[str] = []
                other: List[str] = []
                total = 0
                max_total = 40

                # Patterns
                import_pat = re.compile(
                    r"(import\s+.*" + re.escape(identifier) + r"|from\s+\S+\s+import\s+.*"
                    + re.escape(identifier) + r")", re.IGNORECASE
                )
                def_pat = re.compile(
                    r"(def\s+" + re.escape(identifier) + r"|class\s+" + re.escape(identifier) + r")",
                    re.IGNORECASE
                )
                call_pat = re.compile(re.escape(identifier) + r"\s*\(")
                mention_pat = re.compile(re.escape(identifier))

                for dirpath, dirnames, filenames in os.walk(local_repo_dir):
                    dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
                    for fname in filenames:
                        if os.path.splitext(fname)[1].lower() in _BINARY_EXT:
                            continue
                        fpath = os.path.join(dirpath, fname)
                        rel = os.path.relpath(fpath, local_repo_dir).replace("\\", "/")
                        try:
                            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                                for lineno, line in enumerate(f, 1):
                                    if not mention_pat.search(line):
                                        continue
                                    snippet = line.rstrip()[:120]
                                    entry = f"  {rel}:{lineno}: {snippet}"
                                    if def_pat.search(line):
                                        definitions.append(entry)
                                    elif import_pat.search(line):
                                        imports.append(entry)
                                    elif call_pat.search(line):
                                        calls.append(entry)
                                    else:
                                        other.append(entry)
                                    total += 1
                                    if total >= max_total:
                                        break
                        except (OSError, UnicodeDecodeError):
                            continue
                        if total >= max_total:
                            break
                    if total >= max_total:
                        break

                if total == 0:
                    return f"No references found for '{identifier}'."

                parts = [f"References for `{identifier}` ({total} found):"]
                if definitions:
                    parts.append(f"\n**Definitions** ({len(definitions)}):")
                    parts.extend(definitions[:5])
                if imports:
                    parts.append(f"\n**Imports** ({len(imports)}):")
                    parts.extend(imports[:10])
                if calls:
                    parts.append(f"\n**Calls** ({len(calls)}):")
                    parts.extend(calls[:15])
                if other:
                    parts.append(f"\n**Other mentions** ({len(other)}):")
                    parts.extend(other[:10])
                output = "\n".join(parts)
                if total >= max_total:
                    output += f"\n... [limited to {max_total} results]"
                return output
            except Exception as exc:
                logger.error("find_references error for '%s': %s", identifier, exc)
                return f"Find references error: {exc}"

        tools["find_references"] = find_references

    # ------------------------------------------------------------------
    # memory_search: search the persistent knowledge base
    # ------------------------------------------------------------------
    async def memory_search(query: str) -> str:
        """Search the long-term knowledge base for previously learned facts and insights."""
        try:
            from api.memory.manager import get_memory_manager

            mm = get_memory_manager()
            entries = mm.search_knowledge(
                user_id=user_id or "anonymous",
                repo_id=_infer_repo_id(repo_url),
                query_text=query.strip(),
                limit=8,
            )
            if not entries:
                return "No relevant knowledge found in memory."

            parts = []
            for entry in entries:
                key = entry.key or "unknown"
                value = entry.value if isinstance(entry.value, str) else str(entry.value)
                parts.append(f"### {key}\n{value[:500]}")
            return "\n\n---\n\n".join(parts)
        except Exception as exc:
            logger.error("memory_search error: %s", exc)
            return f"Memory search error: {exc}"

    tools["memory_search"] = memory_search

    return tools


def _extract_definition(
    lines: List[str], target: str
) -> Optional[Dict[str, object]]:
    """Extract a function or class definition from source lines.

    Handles Python (def/class with indentation), JavaScript/TypeScript
    (function/class with braces), and similar languages.
    Returns {"start": int, "end": int, "text": str} or None.
    """
    # Build patterns for common languages
    patterns = [
        # Python: def target( or class target( or class target:
        re.compile(r"^(\s*)(async\s+)?def\s+" + re.escape(target) + r"\s*\("),
        re.compile(r"^(\s*)class\s+" + re.escape(target) + r"[\s(:]"),
        # JS/TS: function target(, const target =, export function target(
        re.compile(r"^(\s*)(export\s+)?(async\s+)?function\s+" + re.escape(target) + r"\s*[(<]"),
        re.compile(r"^(\s*)(export\s+)?(const|let|var)\s+" + re.escape(target) + r"\s*="),
        re.compile(r"^(\s*)(export\s+)?class\s+" + re.escape(target) + r"[\s{<]"),
    ]

    for line_idx, line in enumerate(lines):
        matched_indent = None
        for pat in patterns:
            m = pat.match(line)
            if m:
                matched_indent = len(m.group(1))
                break
        if matched_indent is None:
            continue

        # Found the start — now find the end of the definition
        start = line_idx
        # Walk backwards to include decorators / comments directly above
        while start > 0:
            prev = lines[start - 1].rstrip()
            if prev.startswith("@") or prev.startswith("//") or prev.startswith("#"):
                start -= 1
            else:
                break

        # Walk forward to find the end (indentation-based for Python, brace-matching for JS/TS)
        end = line_idx + 1
        is_brace_lang = "{" in line

        if is_brace_lang:
            # Brace counting
            depth = 0
            for i in range(line_idx, len(lines)):
                depth += lines[i].count("{") - lines[i].count("}")
                end = i + 1
                if depth <= 0 and i > line_idx:
                    break
        else:
            # Indentation-based (Python): definition ends when a non-empty line
            # has indentation <= the definition line, after the first body line
            for i in range(line_idx + 1, len(lines)):
                stripped = lines[i].rstrip()
                if not stripped:
                    # blank line — continue
                    end = i + 1
                    continue
                current_indent = len(lines[i]) - len(lines[i].lstrip())
                if current_indent <= matched_indent:
                    break
                end = i + 1

        # Cap at a reasonable size (~150 lines)
        if end - start > 150:
            end = start + 150

        text = "".join(lines[start:end])
        return {"start": start + 1, "end": end, "text": text}

    return None


def _get_local_repo_dir(rag_instance) -> Optional[str]:
    """Extract the local repo directory from a RAG instance, if available."""
    try:
        db_mgr = getattr(rag_instance, "db_manager", None)
        if db_mgr:
            paths = getattr(db_mgr, "repo_paths", None)
            if paths and "save_repo_dir" in paths:
                d = paths["save_repo_dir"]
                if os.path.isdir(d):
                    return d
    except Exception:
        pass
    return None


def _infer_repo_id(repo_url: Optional[str]) -> str:
    """Derive a repo_id string from a URL, matching websocket_wiki convention."""
    if not repo_url:
        return ""
    return repo_url.replace("/", "_").replace(".", "_")[:50]

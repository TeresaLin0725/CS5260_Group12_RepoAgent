"""
Code-Grounded Test Case Generator (Multi-Repo).

Clones diverse open-source repositories, extracts code context, and uses an
LLM to generate high-quality QA test cases grounded in real code — minimizing
hallucination by providing the actual source as context.

Each generated test case is tagged with its ``repo_url`` so the eval runner
knows which repository to index before running that case.

Generates test cases for:
  - react_testcases.json  (ReAct QA evaluation)
  - rag_testcases.json    (RAG retrieval evaluation)

Usage:
    # Dry run — preview which files will be used:
    python -m eval.offline.generate_testcases --dry-run

    # Generate from default curated repo list:
    python -m eval.offline.generate_testcases --provider openai --model gpt-4o

    # Generate from specific repos:
    python -m eval.offline.generate_testcases --provider openai --model gpt-4o \\
        --repo https://github.com/pallets/flask \\
        --repo https://github.com/tiangolo/fastapi

    # Include the local project as well:
    python -m eval.offline.generate_testcases --provider ollama --model qwen3:8b --include-local

    # Overwrite existing fixtures:
    python -m eval.offline.generate_testcases --provider openai --model gpt-4o --overwrite
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is importable and .env is loaded
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
try:
    from dotenv import load_dotenv
    load_dotenv(_PROJECT_ROOT / ".env")
except ImportError:
    pass  # dotenv not installed, rely on env vars being set externally
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)

FIXTURES_DIR = Path(__file__).parent / "fixtures"
REPOS_CACHE_DIR = Path(__file__).parent / ".repos"

# -----------------------------------------------------------------------
# Default curated repo list — diverse languages, domains, sizes
# -----------------------------------------------------------------------

@dataclass
class RepoConfig:
    """A repository to generate test cases from."""
    url: str
    branch: str = "main"
    description: str = ""
    language: str = "python"          # primary language
    tags: List[str] = field(default_factory=list)

# Small-to-medium, well-documented, diverse open-source repos
DEFAULT_REPOS: List[RepoConfig] = [
    RepoConfig(
        url="https://github.com/pallets/flask",
        description="Lightweight Python web framework",
        language="python",
        tags=["web", "framework", "python"],
    ),
    RepoConfig(
        url="https://github.com/psf/requests",
        description="HTTP library for Python",
        language="python",
        tags=["http", "networking", "python"],
    ),
    RepoConfig(
        url="https://github.com/pydantic/pydantic",
        branch="main",
        description="Data validation using Python type annotations",
        language="python",
        tags=["validation", "typing", "python"],
    ),
    RepoConfig(
        url="https://github.com/encode/httpx",
        description="Async HTTP client for Python",
        language="python",
        tags=["http", "async", "python"],
    ),
    RepoConfig(
        url="https://github.com/tiangolo/fastapi",
        description="Modern Python web framework for APIs",
        language="python",
        tags=["web", "api", "async", "python"],
    ),
]

# -----------------------------------------------------------------------
# Language-specific file discovery
# -----------------------------------------------------------------------

# Extensions to scan per language
LANG_EXTENSIONS = {
    "python": [".py"],
    "typescript": [".ts", ".tsx"],
    "javascript": [".js", ".jsx"],
    "go": [".go"],
    "rust": [".rs"],
    "java": [".java"],
}

# Directories to always skip
SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", ".tox",
    "dist", "build", ".eggs", "*.egg-info", ".mypy_cache", ".pytest_cache",
    "vendor", "third_party", "migrations", "test", "tests", "docs",
    "docs_src", "examples", "benchmarks", "scripts", ".github",
    "v1", "compat",  # skip legacy/compat layers
}

# Minimum file size (bytes) to consider — skip tiny __init__.py etc.
MIN_FILE_SIZE = 500

# Maximum files to select per repo
MAX_FILES_PER_REPO = 12


def discover_source_files(
    repo_dir: Path,
    language: str = "python",
    max_files: int = MAX_FILES_PER_REPO,
) -> List[Path]:
    """Auto-discover the most important source files in a repository.

    Strategy:
    1. Walk the repo, skip excluded dirs
    2. Filter by language extension and minimum size
    3. Score each file by importance heuristics:
       - More classes/functions = more important
       - Deeper nesting = less important (prefer core modules)
       - Larger file = more important (up to a point)
       - Files named after the project or in src/ are preferred
    4. Return top-N files sorted by score
    """
    extensions = LANG_EXTENSIONS.get(language, [".py"])
    candidates: List[Tuple[Path, float]] = []

    for root, dirs, files in os.walk(repo_dir):
        # Prune excluded directories
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]

        rel_root = Path(root).relative_to(repo_dir)
        depth = len(rel_root.parts)

        for fname in files:
            fpath = Path(root) / fname
            if not any(fname.endswith(ext) for ext in extensions):
                continue
            if fpath.stat().st_size < MIN_FILE_SIZE:
                continue

            # Score the file
            score = _score_file(fpath, repo_dir, depth)
            candidates.append((fpath, score))

    # Sort by score descending, take top N
    candidates.sort(key=lambda x: x[1], reverse=True)
    return [c[0] for c in candidates[:max_files]]


def _score_file(fpath: Path, repo_dir: Path, depth: int) -> float:
    """Heuristic score for file importance."""
    score = 0.0
    rel = fpath.relative_to(repo_dir)
    name = fpath.stem

    # Penalize deep nesting
    score -= depth * 2

    # File size (log scale, cap benefit at ~10KB)
    size = fpath.stat().st_size
    score += min(size / 1000, 10)

    # Prefer src/ or lib/ directories
    parts_lower = [p.lower() for p in rel.parts]
    if "src" in parts_lower or "lib" in parts_lower or "core" in parts_lower:
        score += 5

    # Bonus for common important file names
    important_names = {"app", "main", "core", "base", "client", "server",
                       "api", "router", "middleware", "auth", "models",
                       "utils", "helpers", "config", "settings", "manager",
                       "handler", "service", "engine", "pipeline", "schema"}
    if name.lower() in important_names:
        score += 3

    # Penalize test/example files that leaked through
    if name.startswith("test_") or name.endswith("_test") or "conftest" in name:
        score -= 20

    # Penalize __init__.py
    if name == "__init__":
        score -= 10

    # Count classes and functions (quick regex, no full AST parse)
    try:
        content = fpath.read_text(encoding="utf-8", errors="replace")
        n_classes = len(re.findall(r"^\s*class\s+\w+", content, re.MULTILINE))
        n_functions = len(re.findall(r"^\s*(?:async\s+)?def\s+\w+", content, re.MULTILINE))
        score += n_classes * 2 + n_functions * 0.5
    except Exception:
        pass

    return score


# -----------------------------------------------------------------------
# Repo cloning
# -----------------------------------------------------------------------

def clone_repo(repo: RepoConfig, cache_dir: Path = REPOS_CACHE_DIR) -> Optional[Path]:
    """Clone a repo (shallow) into cache directory. Returns repo local path."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Derive a stable directory name from the URL
    repo_name = repo.url.rstrip("/").split("/")[-2] + "_" + repo.url.rstrip("/").split("/")[-1]
    repo_dir = cache_dir / repo_name

    if repo_dir.exists():
        logger.info("Using cached repo: %s", repo_dir)
        return repo_dir

    logger.info("Cloning %s (branch=%s) ...", repo.url, repo.branch)
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", repo.branch,
             repo.url, str(repo_dir)],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        logger.info("Cloned to %s", repo_dir)
        return repo_dir
    except subprocess.CalledProcessError as e:
        logger.error("Clone failed for %s: %s", repo.url, e.stderr[:500])
        # Try without --branch (some repos use 'master')
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", repo.url, str(repo_dir)],
                check=True,
                capture_output=True,
                text=True,
                timeout=120,
            )
            return repo_dir
        except subprocess.CalledProcessError as e2:
            logger.error("Clone retry failed: %s", e2.stderr[:500])
            return None
    except subprocess.TimeoutExpired:
        logger.error("Clone timed out for %s", repo.url)
        return None


# -----------------------------------------------------------------------
# Code extraction (language-agnostic with Python AST fast-path)
# -----------------------------------------------------------------------

@dataclass
class ExtractedCode:
    """Structured extraction from a source file."""
    file_path: str          # relative to repo root
    full_content: str
    classes: List[Dict[str, Any]]
    functions: List[Dict[str, Any]]
    imports: List[str]
    constants: List[str]
    line_count: int


def extract_code_structure(file_path: Path, repo_dir: Path) -> Optional[ExtractedCode]:
    """Parse a source file and extract its structural elements."""
    if not file_path.exists():
        return None

    content = file_path.read_text(encoding="utf-8", errors="replace")
    rel_path = str(file_path.relative_to(repo_dir)).replace("\\", "/")

    # For Python, use AST for precise extraction
    if file_path.suffix == ".py":
        return _extract_python(content, rel_path)

    # For other languages, use regex-based extraction
    return _extract_generic(content, rel_path)


def _extract_python(content: str, rel_path: str) -> Optional[ExtractedCode]:
    """Extract structure from Python source using AST."""
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return _extract_generic(content, rel_path)

    classes = []
    functions = []
    imports = []
    constants = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            cls_info = {
                "name": node.name,
                "docstring": ast.get_docstring(node) or "",
                "methods": [],
            }
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    sig = _get_function_signature(content, item)
                    cls_info["methods"].append({
                        "name": item.name,
                        "signature": sig,
                        "docstring": ast.get_docstring(item) or "",
                    })
            classes.append(cls_info)

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            sig = _get_function_signature(content, node)
            functions.append({
                "name": node.name,
                "signature": sig,
                "docstring": ast.get_docstring(node) or "",
            })

        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            seg = ast.get_source_segment(content, node)
            if seg:
                imports.append(seg)

        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    val_src = ast.get_source_segment(content, node.value) or ""
                    if len(val_src) < 200:
                        constants.append(f"{target.id} = {val_src}")

    return ExtractedCode(
        file_path=rel_path,
        full_content=content,
        classes=classes,
        functions=functions,
        imports=imports,
        constants=constants,
        line_count=len(content.splitlines()),
    )


def _extract_generic(content: str, rel_path: str) -> ExtractedCode:
    """Regex-based extraction for non-Python files."""
    classes = []
    functions = []

    # Match class-like definitions (Python, TS, Java, Go struct, Rust struct/impl)
    for m in re.finditer(r"^\s*(?:export\s+)?(?:class|struct|impl|interface)\s+(\w+)", content, re.MULTILINE):
        classes.append({"name": m.group(1), "docstring": "", "methods": []})

    # Match function definitions
    for m in re.finditer(r"^\s*(?:export\s+)?(?:async\s+)?(?:function|func|fn|def)\s+(\w+)\s*\(", content, re.MULTILINE):
        functions.append({"name": m.group(1), "signature": m.group(0).strip(), "docstring": ""})

    return ExtractedCode(
        file_path=rel_path,
        full_content=content,
        classes=classes,
        functions=functions,
        imports=[],
        constants=[],
        line_count=len(content.splitlines()),
    )


def _get_function_signature(source: str, node) -> str:
    """Extract the 'def func(args) -> ret:' line(s) from source."""
    lines = source.splitlines()
    start = node.lineno - 1
    sig_lines = []
    for i in range(start, min(start + 10, len(lines))):
        sig_lines.append(lines[i])
        if lines[i].rstrip().endswith(":"):
            break
    return "\n".join(sig_lines)


def format_code_context(extracted: ExtractedCode, max_chars: int = 12000) -> str:
    """Format extracted code into a structured context string for the LLM prompt."""
    parts = [f"# File: {extracted.file_path}  ({extracted.line_count} lines)\n"]

    if extracted.constants:
        parts.append("## Constants")
        for c in extracted.constants[:10]:
            parts.append(f"  {c}")
        parts.append("")

    if extracted.functions:
        parts.append("## Functions")
        for fn in extracted.functions:
            parts.append(f"### {fn.get('signature', fn.get('name', '?'))}")
            if fn.get("docstring"):
                parts.append(f'    """{fn["docstring"][:300]}"""')
            parts.append("")

    if extracted.classes:
        parts.append("## Classes")
        for cls in extracted.classes:
            parts.append(f"### class {cls['name']}")
            if cls.get("docstring"):
                parts.append(f'    """{cls["docstring"][:300]}"""')
            for m in cls.get("methods", []):
                parts.append(f"  #### {m.get('signature', m.get('name', '?'))}")
                if m.get("docstring"):
                    parts.append(f'      """{m["docstring"][:200]}"""')
            parts.append("")

    structured = "\n".join(parts)

    # If structured summary is small enough, also include trimmed source
    remaining = max_chars - len(structured)
    if remaining > 2000:
        source_trimmed = extracted.full_content[:remaining]
        return structured + "\n\n## Full Source (trimmed)\n```\n" + source_trimmed + "\n```"

    return structured


# -----------------------------------------------------------------------
# Prompt templates
# -----------------------------------------------------------------------

REACT_GENERATION_PROMPT = """\
You are a test case generator for an AI coding assistant's evaluation suite.

Your task: generate {n_cases} high-quality QA test cases based on the source code below.
These test cases evaluate a ReAct (Reasoning + Acting) agent that answers questions
about code repositories using tools like rag_search, read_file, code_grep, etc.

## Repository
- URL: {repo_url}
- Description: {repo_description}

## Source Code Context
{code_context}

## Available Agent Tools
- rag_search(query) -- semantic search over the codebase
- read_file(file_path) -- read a specific file
- list_repo_files(path) -- list directory contents
- code_grep(pattern) -- regex search across source files
- memory_search(query) -- search conversation memory

## Category Definitions
{category_definitions}

## Requirements
1. Generate exactly {n_cases} test cases covering these categories: {categories}
2. **golden_answer MUST be grounded in the source code above** -- cite specific class names,
   function names, parameters, algorithms, data structures, and logic flow
3. golden_answer should be 80-200 words with technical depth
4. Questions should be specific and non-trivial (not "what does X do?" but "how does X handle Y?")
5. Include both English and Chinese questions (roughly 60% English, 40% Chinese)
6. For Chinese questions, golden_answer should also be in Chinese
7. expected_tools should reflect which tools the agent would realistically need
8. golden_files should list the file paths (relative to repo root) needed to answer
9. golden_files MUST use repository-root-relative POSIX paths such as "src/flask/json/tag.py"
10. Do NOT repeat directory prefixes, do NOT prepend the current file path, and do NOT invent files
11. Only include files that are present in the provided code context or are clearly referenced by it

## Output Format
Return a JSON array. Each element must have exactly these fields:
```json
{{
  "category": "<one of: {category_list}>",
  "query": "<the question>",
  "language": "<en|zh>",
  "expected_tools": ["<tool_name>", ...],
  "expected_max_iterations": <1|2|3>,
  "golden_answer": "<detailed answer grounded in source code>",
  "golden_files": ["<file_path>", ...],
  "tags": {tags_json}
}}
```

Return ONLY the JSON array, no markdown fences, no explanation.
"""

RAG_GENERATION_PROMPT = """\
You are a test case generator for a RAG (Retrieval-Augmented Generation) evaluation suite.

Your task: generate {n_cases} retrieval test queries for the source code below.
Each query will be used to evaluate whether a hybrid retriever (FAISS + BM25)
correctly retrieves this file when given the query.

## Repository
- URL: {repo_url}
- Description: {repo_description}

## Source Code Context
{code_context}

## Requirements
1. Generate exactly {n_cases} test queries
2. Queries should be natural questions or keyword phrases a developer would type
3. Mix query types:
   - Natural language questions (e.g., "How does the routing engine match URLs?")
   - Technical keyword phrases (e.g., "URL rule converter dispatch")
   - Mixed (e.g., "Flask request context push pop")
4. Include both English and Chinese queries (roughly 60% English, 40% Chinese)
5. Queries should be specific enough that this file is clearly the best match

## Output Format
Return a JSON array. Each element must have exactly these fields:
```json
{{
  "query": "<retrieval query>",
  "golden_files": ["{file_path}"],
  "language": "<en|zh>",
  "tags": {tags_json}
}}
```

Return ONLY the JSON array, no markdown fences, no explanation.
"""


# -----------------------------------------------------------------------
# Available tools & categories
# -----------------------------------------------------------------------

AVAILABLE_TOOLS = ["rag_search", "read_file", "list_repo_files", "code_grep", "memory_search"]

CATEGORY_DEFINITIONS = {
    "single_tool": {
        "description": "Question answerable with ONE tool call (usually rag_search)",
        "expected_tools_hint": '["rag_search"]',
        "expected_max_iterations": 2,
    },
    "multi_tool": {
        "description": "Question requiring MULTIPLE tool calls in sequence (e.g. search then read file)",
        "expected_tools_hint": '["rag_search", "read_file"]',
        "expected_max_iterations": 3,
    },
    "code_analysis": {
        "description": "Deep question about architecture, design patterns, trade-offs, or algorithm details",
        "expected_tools_hint": '["rag_search", "read_file"]',
        "expected_max_iterations": 3,
    },
    "architecture": {
        "description": "Question about how components interact, system design, data flow",
        "expected_tools_hint": '["rag_search", "read_file"]',
        "expected_max_iterations": 3,
    },
}


# -----------------------------------------------------------------------
# LLM interaction
# -----------------------------------------------------------------------

async def call_llm_for_generation(
    llm_fn,
    prompt: str,
    max_retries: int = 2,
) -> Optional[List[Dict[str, Any]]]:
    """Call LLM and parse the JSON response, with retries."""
    for attempt in range(max_retries + 1):
        try:
            raw = await llm_fn(prompt)
            raw = (raw or "").strip()

            if not raw:
                logger.warning("LLM returned empty response (attempt %d)", attempt + 1)
                if attempt < max_retries:
                    logger.info("Retrying...")
                continue

            # Strip <think>...</think> blocks (common with reasoning models)
            raw = re.sub(r"<think>[\s\S]*?</think>", "", raw).strip()

            # Strip markdown fences
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\s*\n?", "", raw)
                raw = re.sub(r"\n?\s*```\s*$", "", raw)

            # Try to extract JSON array from the response even if surrounded by text
            if not raw.startswith("["):
                match = re.search(r"\[[\s\S]*\]", raw)
                if match:
                    raw = match.group(0)

            if not raw:
                logger.warning("LLM response empty after cleanup (attempt %d)", attempt + 1)
                if attempt < max_retries:
                    logger.info("Retrying...")
                continue

            logger.debug("LLM response (first 500 chars): %s", raw[:500])

            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return parsed
            logger.warning("LLM returned non-list JSON (attempt %d)", attempt + 1)
        except json.JSONDecodeError as e:
            logger.warning(
                "JSON parse error (attempt %d): %s\nResponse preview: %.300s",
                attempt + 1, e, raw,
            )
            if attempt < max_retries:
                logger.info("Retrying...")
        except Exception as e:
            logger.error("LLM call failed (attempt %d): %s", attempt + 1, e)
            if attempt < max_retries:
                logger.info("Retrying...")

    return None


# -----------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------

def validate_react_case(case: Dict[str, Any], repo_dir: Path) -> List[str]:
    """Validate a generated ReAct test case, returning a list of issues."""
    issues = []

    required_fields = ["category", "query", "language", "expected_tools",
                       "expected_max_iterations", "golden_answer", "golden_files", "tags"]
    for f in required_fields:
        if f not in case:
            issues.append(f"Missing field: {f}")

    if case.get("language") not in ("en", "zh"):
        issues.append(f"Invalid language: {case.get('language')}")

    if case.get("category") not in CATEGORY_DEFINITIONS:
        issues.append(f"Invalid category: {case.get('category')}")

    for tool in case.get("expected_tools", []):
        if tool not in AVAILABLE_TOOLS:
            issues.append(f"Unknown tool: {tool}")

    answer = case.get("golden_answer", "")
    if answer and len(answer) < 40:
        issues.append(f"golden_answer too short ({len(answer)} chars)")

    # Validate golden_files exist in the cloned repo
    for gf in case.get("golden_files", []):
        full = repo_dir / gf
        if not full.exists():
            issues.append(f"golden_file does not exist: {gf}")

    return issues


def validate_rag_case(case: Dict[str, Any], repo_dir: Path) -> List[str]:
    """Validate a generated RAG test case."""
    issues = []

    required_fields = ["query", "golden_files", "language", "tags"]
    for f in required_fields:
        if f not in case:
            issues.append(f"Missing field: {f}")

    if case.get("language") not in ("en", "zh"):
        issues.append(f"Invalid language: {case.get('language')}")

    for gf in case.get("golden_files", []):
        full = repo_dir / gf
        if not full.exists():
            issues.append(f"golden_file does not exist: {gf}")

    if not case.get("query", "").strip():
        issues.append("Empty query")

    return issues


def _normalize_golden_file_path(path_str: str, repo_dir: Path) -> str:
    """Normalize a model-produced repo-relative path and repair common prefix duplication."""
    if not isinstance(path_str, str):
        return path_str

    normalized = path_str.strip().replace("\\", "/")
    normalized = re.sub(r"^(?:\./|/)+", "", normalized)
    if not normalized:
        return normalized

    direct = repo_dir / normalized
    if direct.exists():
        return normalized

    parts = [part for part in normalized.split("/") if part and part != "."]
    if not parts:
        return normalized

    # Repair duplicated prefixes like src/flask/src/flask/json.py -> src/flask/json.py.
    for start_idx in range(1, len(parts)):
        candidate = "/".join(parts[start_idx:])
        if (repo_dir / candidate).exists():
            return candidate

    return normalized


def normalize_case_paths(case: Dict[str, Any], repo_dir: Path) -> Dict[str, Any]:
    """Normalize golden_files in-place and log any automatic repairs."""
    golden_files = case.get("golden_files")
    if not isinstance(golden_files, list):
        return case

    normalized_files: List[str] = []
    for original in golden_files:
        normalized = _normalize_golden_file_path(original, repo_dir)
        if normalized != original:
            logger.info("    Normalized golden_file: %s -> %s", original, normalized)
        if normalized and normalized not in normalized_files:
            normalized_files.append(normalized)

    case["golden_files"] = normalized_files
    return case


# -----------------------------------------------------------------------
# ID assignment
# -----------------------------------------------------------------------

def assign_ids(cases: List[Dict[str, Any]], prefix: str, start_id: int) -> List[Dict[str, Any]]:
    """Assign sequential IDs to generated cases."""
    for i, case in enumerate(cases):
        case["id"] = f"{prefix}_{start_id + i:03d}"
    return cases


def get_next_id(existing_cases: List[Dict[str, Any]], prefix: str) -> int:
    """Find the next available ID number from existing cases."""
    max_id = 0
    pattern = re.compile(rf"^{prefix}_(\d+)$")
    for case in existing_cases:
        m = pattern.match(case.get("id", ""))
        if m:
            max_id = max(max_id, int(m.group(1)))
    return max_id + 1


# -----------------------------------------------------------------------
# Per-repo generation
# -----------------------------------------------------------------------

async def generate_cases_for_repo(
    llm_fn,
    repo: RepoConfig,
    repo_dir: Path,
    target: str,
    react_per_file: int = 5,
    rag_per_file: int = 3,
    dry_run: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Generate ReAct and/or RAG test cases for a single repo.

    Returns (react_cases, rag_cases).
    """
    # Discover important source files
    source_files = discover_source_files(repo_dir, language=repo.language)

    if not source_files:
        logger.warning("No source files found in %s", repo.url)
        return [], []

    logger.info(
        "Repo %s: discovered %d important source files",
        repo.url, len(source_files),
    )
    for sf in source_files:
        rel = sf.relative_to(repo_dir)
        logger.info("  - %s (%d bytes)", rel, sf.stat().st_size)

    react_cases: List[Dict[str, Any]] = []
    rag_cases: List[Dict[str, Any]] = []

    categories = ["single_tool", "multi_tool", "code_analysis", "architecture"]

    for fpath in source_files:
        extracted = extract_code_structure(fpath, repo_dir)
        if extracted is None:
            continue

        rel_path = extracted.file_path
        file_tags = list(set(repo.tags + [Path(rel_path).stem]))

        # --- ReAct cases ---
        if target in ("all", "react"):
            code_context = format_code_context(extracted, max_chars=12000)

            cat_defs = []
            for cat in categories:
                defn = CATEGORY_DEFINITIONS.get(cat, {})
                cat_defs.append(
                    f"- **{cat}**: {defn.get('description', 'N/A')}  "
                    f"(typical tools: {defn.get('expected_tools_hint', '[]')})"
                )

            prompt = REACT_GENERATION_PROMPT.format(
                n_cases=react_per_file,
                repo_url=repo.url,
                repo_description=repo.description,
                code_context=code_context,
                category_definitions="\n".join(cat_defs),
                categories=", ".join(categories),
                category_list=", ".join(categories),
                tags_json=json.dumps(file_tags),
            )

            if dry_run:
                print(f"  [DRY RUN] ReAct | {rel_path} | {react_per_file} cases | prompt={len(prompt)} chars")
            else:
                logger.info("  Generating %d ReAct cases from %s ...", react_per_file, rel_path)
                cases = await call_llm_for_generation(llm_fn, prompt)
                if cases:
                    for c in cases:
                        c["repo_url"] = repo.url
                        normalize_case_paths(c, repo_dir)
                    valid = [c for c in cases if not validate_react_case(c, repo_dir)]
                    # Log dropped cases
                    for c in cases:
                        issues = validate_react_case(c, repo_dir)
                        if issues:
                            logger.warning("    Drop: %s | %s", c.get("query", "?")[:40], "; ".join(issues))
                    valid = [c for c in cases if not validate_react_case(c, repo_dir)]
                    react_cases.extend(valid)
                    logger.info("    %s: %d/%d valid", rel_path, len(valid), len(cases))

        # --- RAG cases ---
        if target in ("all", "rag"):
            code_context = format_code_context(extracted, max_chars=8000)

            prompt = RAG_GENERATION_PROMPT.format(
                n_cases=rag_per_file,
                repo_url=repo.url,
                repo_description=repo.description,
                code_context=code_context,
                file_path=rel_path,
                tags_json=json.dumps(file_tags),
            )

            if dry_run:
                print(f"  [DRY RUN] RAG   | {rel_path} | {rag_per_file} cases | prompt={len(prompt)} chars")
            else:
                logger.info("  Generating %d RAG cases from %s ...", rag_per_file, rel_path)
                cases = await call_llm_for_generation(llm_fn, prompt)
                if cases:
                    for c in cases:
                        c["repo_url"] = repo.url
                        normalize_case_paths(c, repo_dir)
                    valid = [c for c in cases if not validate_rag_case(c, repo_dir)]
                    rag_cases.extend(valid)
                    logger.info("    %s: %d/%d valid", rel_path, len(valid), len(cases) if cases else 0)

    return react_cases, rag_cases


# -----------------------------------------------------------------------
# File I/O — merge or overwrite fixtures
# -----------------------------------------------------------------------

def load_existing_fixture(filename: str) -> List[Dict[str, Any]]:
    """Load existing fixture file, return empty list if not found."""
    path = FIXTURES_DIR / filename
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_fixture(cases: List[Dict[str, Any]], filename: str) -> Path:
    """Save cases to fixture file."""
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIXTURES_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2, ensure_ascii=False)
    logger.info("Saved %d cases to %s", len(cases), path)
    return path


def merge_cases(
    existing: List[Dict[str, Any]],
    new_cases: List[Dict[str, Any]],
    prefix: str,
    overwrite: bool = False,
) -> List[Dict[str, Any]]:
    """Merge new cases into existing, or overwrite entirely."""
    if overwrite:
        return assign_ids(new_cases, prefix, start_id=1)

    start_id = get_next_id(existing, prefix)
    new_with_ids = assign_ids(new_cases, prefix, start_id)

    # Deduplicate by query text
    existing_queries = {c.get("query", "").strip().lower() for c in existing}
    deduped = [c for c in new_with_ids if c.get("query", "").strip().lower() not in existing_queries]

    if len(deduped) < len(new_with_ids):
        logger.info(
            "Deduplication removed %d duplicate queries",
            len(new_with_ids) - len(deduped),
        )

    return existing + deduped


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

async def run(args: argparse.Namespace):
    """Main generation pipeline."""

    # Determine which repos to use
    repos: List[RepoConfig] = []

    if args.repo:
        for url in args.repo:
            repos.append(RepoConfig(
                url=url.rstrip("/"),
                description="User-specified repository",
                language=args.language,
            ))

    if args.include_local:
        repos.append(RepoConfig(
            url="local",
            description="Local project (deepwiki-open-chatbox)",
            language="python",
            tags=["local"],
        ))

    if not repos:
        repos = DEFAULT_REPOS
        logger.info("Using default curated repo list (%d repos)", len(repos))

    # Create LLM callable
    if args.dry_run:
        llm_fn = None
    else:
        from eval.offline.real_backends import create_real_llm
        llm_fn = create_real_llm(provider=args.provider, model=args.model)

    target = args.target

    all_react: List[Dict[str, Any]] = []
    all_rag: List[Dict[str, Any]] = []

    for repo in repos:
        print(f"\n{'='*60}")
        print(f"  Repo: {repo.url}")
        print(f"  Description: {repo.description}")
        print(f"{'='*60}")

        # Get repo directory
        if repo.url == "local":
            repo_dir = _PROJECT_ROOT
        else:
            repo_dir = clone_repo(repo)
            if repo_dir is None:
                logger.error("Skipping %s (clone failed)", repo.url)
                continue

        react_cases, rag_cases = await generate_cases_for_repo(
            llm_fn=llm_fn,
            repo=repo,
            repo_dir=repo_dir,
            target=target,
            react_per_file=args.per_file,
            rag_per_file=args.rag_per_file,
            dry_run=args.dry_run,
        )

        all_react.extend(react_cases)
        all_rag.extend(rag_cases)

        if not args.dry_run:
            print(f"  -> {len(react_cases)} ReAct + {len(rag_cases)} RAG cases")

    # Save results
    if args.dry_run:
        print(f"\n[DRY RUN] Would generate from {len(repos)} repos")
        return

    if all_react:
        existing = load_existing_fixture("react_testcases.json")
        merged = merge_cases(existing, all_react, "react", overwrite=args.overwrite)
        save_fixture(merged, "react_testcases.json")
        print(f"\nReAct: {len(all_react)} new cases, {len(merged)} total")

    if all_rag:
        existing = load_existing_fixture("rag_testcases.json")
        merged = merge_cases(existing, all_rag, "rag", overwrite=args.overwrite)
        save_fixture(merged, "rag_testcases.json")
        print(f"RAG:   {len(all_rag)} new cases, {len(merged)} total")

    if not all_react and not all_rag:
        print("\nNo valid cases generated.")

    # Summary
    print(f"\nRepos processed: {len(repos)}")
    print(f"Cache directory: {REPOS_CACHE_DIR}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate code-grounded test cases from multiple repositories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run with default repos:
  python -m eval.offline.generate_testcases --dry-run

  # Generate from default repos using OpenAI:
  python -m eval.offline.generate_testcases --provider openai --model gpt-4o

  # Generate from specific repos:
  python -m eval.offline.generate_testcases --provider openai --model gpt-4o \\
      --repo https://github.com/pallets/flask \\
      --repo https://github.com/psf/requests

  # Include local project too:
  python -m eval.offline.generate_testcases --provider ollama --model qwen3:8b --include-local

  # Only ReAct, 3 cases per file:
  python -m eval.offline.generate_testcases --target react --per-file 3

  # Clean repo cache:
  python -m eval.offline.generate_testcases --clean-cache
""",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "google", "ollama", "openrouter", "bedrock", "azure", "dashscope"],
        default="openai",
        help="LLM provider (default: openai)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (default: provider's default)",
    )
    parser.add_argument(
        "--repo",
        type=str,
        action="append",
        help="Repository URL (can be repeated). If omitted, uses curated default list.",
    )
    parser.add_argument(
        "--include-local",
        action="store_true",
        help="Also generate cases from the local project (deepwiki-open-chatbox)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="python",
        choices=list(LANG_EXTENSIONS.keys()),
        help="Primary language of --repo repos (default: python)",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="all",
        choices=["react", "rag", "all"],
        help="Which test types to generate (default: all)",
    )
    parser.add_argument(
        "--per-file",
        type=int,
        default=5,
        help="ReAct cases per source file (default: 5)",
    )
    parser.add_argument(
        "--rag-per-file",
        type=int,
        default=3,
        help="RAG cases per source file (default: 3)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing fixtures instead of merging",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview: show discovered files and prompt sizes without calling LLM",
    )
    parser.add_argument(
        "--clean-cache",
        action="store_true",
        help="Delete cached repos and exit",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.clean_cache:
        if REPOS_CACHE_DIR.exists():
            shutil.rmtree(REPOS_CACHE_DIR)
            print(f"Cleaned cache: {REPOS_CACHE_DIR}")
        else:
            print("No cache to clean.")
        return

    asyncio.run(run(args))


if __name__ == "__main__":
    main()

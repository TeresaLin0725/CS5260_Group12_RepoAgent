"""
Code-aware text splitter.

Splits source code by semantic boundaries (functions, classes, top-level
blocks) instead of naïve word-count windows, so each chunk is a coherent
semantic unit.  Falls back to the standard word-based splitter for
non-code documents or when AST parsing fails.

Supported languages:
    - Python  (via ``ast`` stdlib)
    - Generic (regex-based heuristic for JS/TS/Java/Go/Rust/C/C++)

Design goals
------------
1. **Each chunk ≈ one logical unit** (function, class, or top-level block).
2. **Oversized units are sub-split** with overlap so nothing is lost.
3. **All metadata is preserved**: ``file_path``, ``is_code``, etc.
4. **Drop-in compatible** with adalflow's ``TextSplitter`` interface —
   operates on ``Document`` lists and returns ``Document`` lists.
"""

from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import adalflow as adal
from adalflow.core.types import Document

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Target max words per chunk.  Individual functions/classes may exceed this
#: if they cannot be split further; the sub-splitter handles overflow.
DEFAULT_CHUNK_TARGET = 200

#: Overlap words when sub-splitting oversized units.
DEFAULT_CHUNK_OVERLAP = 30

#: Languages where we use AST-based splitting
_AST_LANGUAGES = frozenset({".py"})

#: Languages where we use regex-based heuristic splitting
_REGEX_LANGUAGES = frozenset({
    ".js", ".ts", ".jsx", ".tsx",
    ".java", ".go", ".rs",
    ".c", ".cpp", ".h", ".hpp",
    ".cs", ".swift", ".php",
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _word_count(text: str) -> int:
    return len(text.split())


def _sub_split_text(text: str, target: int, overlap: int) -> List[str]:
    """Split *text* into ≤ *target*-word windows with *overlap* word overlap."""
    words = text.split()
    if len(words) <= target:
        return [text]
    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(start + target, len(words))
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start += target - overlap
    return chunks


# ---------------------------------------------------------------------------
# Python AST splitter
# ---------------------------------------------------------------------------

@dataclass
class _CodeBlock:
    """A contiguous range of source lines representing one semantic unit."""
    name: str          # human-readable label, e.g. "class Foo" / "def bar"
    start_line: int    # 0-based inclusive
    end_line: int      # 0-based exclusive
    text: str


def _extract_python_blocks(source: str) -> List[_CodeBlock]:
    """Parse Python source and return one ``_CodeBlock`` per top-level node.

    Groups consecutive non-class/function lines (imports, constants,
    module-level assignments) into a single "header" block.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        logger.debug("Python AST parse failed; falling back to regex splitter")
        return []

    lines = source.splitlines(keepends=True)
    if not lines:
        return []

    blocks: List[_CodeBlock] = []
    # Track which lines are covered by named nodes
    covered = [False] * len(lines)

    for node in ast.iter_child_nodes(tree):
        if not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
            continue
        start = node.lineno - 1         # ast is 1-based
        end = node.end_lineno           # end_lineno is inclusive → use as exclusive

        # Include decorators
        if hasattr(node, "decorator_list") and node.decorator_list:
            first_dec = node.decorator_list[0]
            if hasattr(first_dec, "lineno"):
                start = min(start, first_dec.lineno - 1)

        # Clamp
        start = max(0, start)
        end = min(end, len(lines))

        if isinstance(node, ast.ClassDef):
            label = f"class {node.name}"
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            label = f"def {node.name}"
        else:
            label = f"block_L{start + 1}"

        text = "".join(lines[start:end])
        blocks.append(_CodeBlock(name=label, start_line=start, end_line=end, text=text))
        for i in range(start, end):
            covered[i] = i < len(covered) and True  # avoid index error

    # Gather uncovered lines into a "header" block
    header_lines: List[str] = []
    header_start: Optional[int] = None
    for i, line in enumerate(lines):
        if not covered[i]:
            if header_start is None:
                header_start = i
            header_lines.append(line)
        else:
            if header_lines:
                text = "".join(header_lines)
                if text.strip():
                    blocks.append(
                        _CodeBlock(
                            name="header",
                            start_line=header_start,
                            end_line=i,
                            text=text,
                        )
                    )
                header_lines = []
                header_start = None

    # Remaining tail
    if header_lines:
        text = "".join(header_lines)
        if text.strip():
            blocks.append(
                _CodeBlock(
                    name="header",
                    start_line=header_start,
                    end_line=len(lines),
                    text=text,
                )
            )

    # Sort by source order
    blocks.sort(key=lambda b: b.start_line)
    return blocks


# ---------------------------------------------------------------------------
# Regex-based splitter for non-Python languages
# ---------------------------------------------------------------------------

#: Patterns that indicate the start of a new top-level definition.
#: They must appear at the beginning of a line (after optional whitespace).
_GENERIC_BOUNDARY_RE = re.compile(
    r"^(?:"
    # C/C++/Java/Go/Rust/JS/TS function/method/class
    r"(?:export\s+)?(?:default\s+)?(?:async\s+)?(?:pub(?:\(crate\))?\s+)?(?:static\s+)?(?:fn|func|function|def|class|interface|struct|enum|impl|type|const\s+\w+\s*=\s*(?:function|\(.*\)\s*=>))"
    r"|"
    # Go function
    r"func\s+(?:\(.*?\)\s+)?\w+"
    r")",
    re.MULTILINE,
)


def _extract_regex_blocks(source: str) -> List[_CodeBlock]:
    """Use regex heuristics to split source into top-level definition blocks."""
    boundaries: List[int] = []
    for m in _GENERIC_BOUNDARY_RE.finditer(source):
        boundaries.append(m.start())

    if not boundaries:
        return []

    blocks: List[_CodeBlock] = []

    # Header: everything before first boundary
    if boundaries[0] > 0:
        header = source[: boundaries[0]]
        if header.strip():
            blocks.append(_CodeBlock(name="header", start_line=0, end_line=0, text=header))

    for i, start in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(source)
        text = source[start:end]
        # Extract name from first line
        first_line = text.split("\n", 1)[0].strip()[:60]
        blocks.append(_CodeBlock(name=first_line, start_line=0, end_line=0, text=text))

    return blocks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class CodeAwareTextSplitter(adal.core.component.DataComponent):
    """Split documents using code-structure awareness.

    For code files (identified by ``meta_data['is_code']`` or file extension),
    uses AST / regex to split by function/class boundaries.  Each resulting
    chunk preserves full metadata from the parent document.

    For non-code documents, falls back to the standard word-based splitter
    (same parameters as adalflow ``TextSplitter``).

    Extends ``adal.DataComponent`` so it can be used inside ``adal.Sequential``.

    Parameters
    ----------
    chunk_size : int
        Target maximum words per chunk (default 200).
    chunk_overlap : int
        Overlap words when sub-splitting oversized units (default 30).
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_TARGET,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ------------------------------------------------------------------ core

    def call(self, documents: List[Document]) -> List[Document]:
        """Split a list of Documents, returning a new (usually longer) list.

        This is the method invoked by ``adal.Sequential`` / ``Component.__call__``.
        """
        result: List[Document] = []
        for doc in documents:
            result.extend(self._split_one(doc))
        return result

    def __call__(self, documents: List[Document]) -> List[Document]:
        """Direct invocation also works."""
        return self.call(documents)

    def _split_one(self, doc: Document) -> List[Document]:
        meta = doc.meta_data or {}
        file_path: str = meta.get("file_path", "")
        is_code: bool = meta.get("is_code", False)

        ext = ""
        if file_path:
            dot = file_path.rfind(".")
            if dot != -1:
                ext = file_path[dot:].lower()

        # Decide strategy
        blocks: List[_CodeBlock] = []
        if is_code or ext in _AST_LANGUAGES | _REGEX_LANGUAGES:
            if ext in _AST_LANGUAGES:
                blocks = _extract_python_blocks(doc.text)
            if not blocks and ext in _REGEX_LANGUAGES:
                blocks = _extract_regex_blocks(doc.text)
            if not blocks:
                # Regex fallback for Python too (e.g. syntax error)
                blocks = _extract_regex_blocks(doc.text)

        if blocks:
            return self._blocks_to_docs(blocks, doc)

        # Fallback: word-based split
        return self._word_split(doc)

    # ------------------------------------------------------------------ blocks → docs

    def _blocks_to_docs(self, blocks: List[_CodeBlock], parent: Document) -> List[Document]:
        """Convert code blocks into Document chunks, sub-splitting oversized ones."""
        meta = parent.meta_data or {}
        file_path = meta.get("file_path", "")
        chunks: List[Document] = []

        # Merge tiny consecutive blocks (< 20 words) into one
        merged = self._merge_tiny_blocks(blocks)

        for i, blk in enumerate(merged):
            wc = _word_count(blk.text)
            if wc <= self.chunk_size:
                texts = [blk.text]
            else:
                texts = _sub_split_text(blk.text, self.chunk_size, self.chunk_overlap)

            for j, text in enumerate(texts):
                if not text.strip():
                    continue
                chunk_meta = dict(meta)
                chunk_meta["chunk_name"] = blk.name
                chunk_meta["chunk_index"] = i
                if len(texts) > 1:
                    chunk_meta["sub_chunk"] = j
                chunks.append(Document(text=text, meta_data=chunk_meta))

        # If nothing was produced, return original doc as single chunk
        if not chunks:
            chunks.append(parent)

        return chunks

    def _merge_tiny_blocks(
        self, blocks: List[_CodeBlock], tiny_threshold: int = 20
    ) -> List[_CodeBlock]:
        """Merge consecutive tiny header/unnamed blocks.

        Only merge when *both* the accumulator is tiny *and* the next
        block is also a header/unnamed block (not a ``def`` or ``class``).
        Named definitions are never absorbed into an adjacent block.
        """
        if not blocks:
            return blocks

        def _is_named(blk: _CodeBlock) -> bool:
            """True for function/class definitions that should stay standalone."""
            n = blk.name
            return n.startswith("def ") or n.startswith("class ") or n.startswith("async def ")

        merged: List[_CodeBlock] = []
        acc = blocks[0]
        for blk in blocks[1:]:
            # Only merge if accumulator is tiny AND incoming is not a named def/class
            if _word_count(acc.text) < tiny_threshold and not _is_named(blk):
                acc = _CodeBlock(
                    name=acc.name,
                    start_line=acc.start_line,
                    end_line=blk.end_line,
                    text=acc.text + blk.text,
                )
            else:
                merged.append(acc)
                acc = blk
        merged.append(acc)
        return merged

    # ------------------------------------------------------------------ word fallback

    def _word_split(self, doc: Document) -> List[Document]:
        """Simple word-count windowed split (same logic as adalflow TextSplitter)."""
        text = doc.text
        if not text or not text.strip():
            return [doc]

        words = text.split()
        if len(words) <= self.chunk_size:
            return [doc]

        meta = doc.meta_data or {}
        chunks: List[Document] = []
        start = 0
        idx = 0
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_text = " ".join(words[start:end])
            chunk_meta = dict(meta)
            chunk_meta["chunk_index"] = idx
            chunks.append(Document(text=chunk_text, meta_data=chunk_meta))
            if end >= len(words):
                break
            start += self.chunk_size - self.chunk_overlap
            idx += 1

        return chunks if chunks else [doc]

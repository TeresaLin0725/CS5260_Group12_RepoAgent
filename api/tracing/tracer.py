"""
AgentTracer — Langfuse **v4** integration for online observability.

Uses the Langfuse v4 Python SDK (OpenTelemetry-based) to record Traces,
Spans, and Generations via manual observations (``start_observation`` /
``start_as_current_observation``).

When ``LANGFUSE_ENABLED`` is falsy (or the SDK is missing), every public
method degrades to a no-op so the rest of the application is unaffected.

Environment variables (read from os.environ / .env):
    LANGFUSE_PUBLIC_KEY   – Langfuse public key
    LANGFUSE_SECRET_KEY   – Langfuse secret key
    LANGFUSE_HOST         – Langfuse API host (also reads LANGFUSE_BASE_URL)
    LANGFUSE_ENABLED      – set to "true" to activate tracing
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy Langfuse import — allows the module to load even when langfuse is not
# installed (all methods become no-ops).
# ---------------------------------------------------------------------------
_langfuse_available = False
try:
    from langfuse import Langfuse  # type: ignore[import-untyped]

    _langfuse_available = True
except ImportError:
    Langfuse = None  # type: ignore[assignment,misc]


def _is_enabled() -> bool:
    return (
        _langfuse_available
        and os.environ.get("LANGFUSE_ENABLED", "false").lower() in ("true", "1", "t")
    )


class _NoOpObj:
    """Lightweight stand-in that silently swallows any attribute access / call."""

    trace_id = None

    def __getattr__(self, _name: str):
        return _noop

    def __call__(self, *_a: Any, **_kw: Any):
        return self


_noop = _NoOpObj()


# ---------------------------------------------------------------------------
# _ObsWrapper — adapts a Langfuse v4 observation to a v2-like interface so
# that calling code (react.py, simple_chat.py, websocket_wiki.py) can keep
# using  parent.span(...)  /  parent.generation(...)  unchanged.
# ---------------------------------------------------------------------------

class _ObsWrapper:
    """Thin adapter around a Langfuse v4 observation object."""

    def __init__(self, obs: Any, client: Any) -> None:
        self._obs = obs
        self._client = client

    # -- identity ----------------------------------------------------------
    @property
    def trace_id(self) -> Optional[str]:
        try:
            return self._obs.trace_id
        except Exception:
            return None

    @property
    def id(self) -> Optional[str]:
        try:
            return self._obs.id
        except Exception:
            return None

    # -- child creation (v2-compatible) ------------------------------------
    def span(
        self,
        *,
        name: str,
        input: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
        **_kw: Any,
    ) -> "_ObsWrapper":
        child = self._obs.start_observation(name=name, as_type="span")
        upd: Dict[str, Any] = {}
        if input is not None:
            upd["input"] = input
        if metadata:
            upd["metadata"] = metadata
        if upd:
            child.update(**upd)
        return _ObsWrapper(child, self._client)

    def generation(
        self,
        *,
        name: str,
        model: Optional[str] = None,
        input: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
        **_kw: Any,
    ) -> "_ObsWrapper":
        child = self._obs.start_observation(name=name, as_type="generation")
        upd: Dict[str, Any] = {}
        if model:
            upd["model"] = model
        if input is not None:
            upd["input"] = input
        if metadata:
            upd["metadata"] = metadata
        if upd:
            child.update(**upd)
        return _ObsWrapper(child, self._client)

    # -- update / end ------------------------------------------------------
    def update(self, **kwargs: Any) -> "_ObsWrapper":
        # Map v2 ``usage`` key to v4 ``usage_details``
        if "usage" in kwargs:
            kwargs["usage_details"] = kwargs.pop("usage")
        try:
            self._obs.update(**kwargs)
        except Exception:
            pass
        return self

    def end(
        self,
        *,
        output: Any = None,
        usage: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **_kw: Any,
    ) -> None:
        upd: Dict[str, Any] = {}
        if output is not None:
            upd["output"] = output
        if usage:
            upd["usage_details"] = usage
        if metadata:
            upd["metadata"] = metadata
        try:
            if upd:
                self._obs.update(**upd)
            self._obs.end()
        except Exception:
            pass


class AgentTracer:
    """
    Thin wrapper around the Langfuse **v4** SDK.

    Usage::

        tracer = get_tracer()
        trace = tracer.start_trace(
            name="agent_request",
            user_id="u123",
            session_id="s456",
            metadata={"repo_id": "owner/repo", "language": "en"},
            tags=["react", "websocket"],
        )

        with tracer.span(trace, "intent_scheduling", input={"query": q}) as sp:
            result = scheduler.schedule(q)
            sp.end(output={"handled": result.handled})

        with tracer.generation(trace, "react_iteration_0", model="gemini") as gen:
            resp = await llm_fn(prompt)
            gen.end(output=resp, usage={"input_tokens": 100, "output_tokens": 50})

        tracer.score(trace, "iteration_count", 2)
        tracer.end_trace(trace)
    """

    def __init__(self) -> None:
        self._enabled = _is_enabled()
        self._client: Optional[Any] = None

        if self._enabled:
            try:
                host = (
                    os.environ.get("LANGFUSE_HOST")
                    or os.environ.get("LANGFUSE_BASE_URL")
                    or "https://cloud.langfuse.com"
                )
                self._client = Langfuse(
                    public_key=os.environ.get("LANGFUSE_PUBLIC_KEY"),
                    secret_key=os.environ.get("LANGFUSE_SECRET_KEY"),
                    base_url=host,
                )
                logger.info("Langfuse tracing enabled (host=%s)", host)
            except Exception as exc:
                logger.warning("Failed to initialise Langfuse client: %s — tracing disabled", exc)
                self._enabled = False
                self._client = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ------------------------------------------------------------------
    # Trace lifecycle
    # ------------------------------------------------------------------
    def start_trace(
        self,
        name: str = "agent_request",
        *,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        input: Optional[Any] = None,
    ) -> Any:
        """Create a new Langfuse trace.  Returns a trace handle (or _NoOpObj)."""
        if not self._enabled:
            return _noop
        try:
            root = self._client.start_observation(name=name, as_type="span")

            # v4 manual observations don't support first-class user_id /
            # session_id / tags — store them in metadata so they are visible
            # in the Langfuse dashboard.
            meta: Dict[str, Any] = dict(metadata or {})
            if user_id:
                meta["user_id"] = user_id
            if session_id:
                meta["session_id"] = session_id
            if tags:
                meta["tags"] = tags

            upd: Dict[str, Any] = {}
            if meta:
                upd["metadata"] = meta
            if input is not None:
                upd["input"] = input
            if upd:
                root.update(**upd)

            return _ObsWrapper(root, self._client)
        except Exception as exc:
            logger.debug("Langfuse start_trace error: %s", exc)
            return _noop

    def end_trace(self, trace: Any, *, output: Optional[Any] = None) -> None:
        if not self._enabled or isinstance(trace, _NoOpObj):
            return
        try:
            trace.end(output=output)
            self._client.flush()
        except Exception as exc:
            logger.debug("Langfuse end_trace error: %s", exc)

    # ------------------------------------------------------------------
    # Span helpers
    # ------------------------------------------------------------------
    @contextmanager
    def span(
        self,
        parent: Any,
        name: str,
        *,
        input: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Context manager that creates a nested span under *parent*."""
        if not self._enabled or isinstance(parent, _NoOpObj):
            yield _SpanCtx(_noop, 0)
            return

        start = time.time()
        try:
            sp = parent.span(name=name, input=input, metadata=metadata or {})
        except Exception as exc:
            logger.debug("Langfuse span creation error: %s", exc)
            yield _SpanCtx(_noop, start)
            return

        ctx = _SpanCtx(sp, start)
        try:
            yield ctx
        except Exception:
            ctx.end(metadata={"error": True})
            raise
        else:
            if not ctx._ended:
                ctx.end()

    # ------------------------------------------------------------------
    # Generation helpers
    # ------------------------------------------------------------------
    @contextmanager
    def generation(
        self,
        parent: Any,
        name: str,
        *,
        model: Optional[str] = None,
        input: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Context manager for an LLM generation event."""
        if not self._enabled or isinstance(parent, _NoOpObj):
            yield _GenCtx(_noop, 0)
            return

        start = time.time()
        try:
            gen = parent.generation(
                name=name,
                model=model,
                input=input,
                metadata=metadata or {},
            )
        except Exception as exc:
            logger.debug("Langfuse generation creation error: %s", exc)
            yield _GenCtx(_noop, start)
            return

        ctx = _GenCtx(gen, start)
        try:
            yield ctx
        except Exception:
            ctx.end(metadata={"error": True})
            raise
        else:
            if not ctx._ended:
                ctx.end()

    # ------------------------------------------------------------------
    # Scoring  (v4: client.create_score)
    # ------------------------------------------------------------------
    def score(
        self,
        trace: Any,
        name: str,
        value: float,
        *,
        comment: Optional[str] = None,
    ) -> None:
        """Attach a numeric score to a trace (non-blocking)."""
        if not self._enabled or isinstance(trace, _NoOpObj):
            return
        try:
            trace_id = getattr(trace, "trace_id", None)
            if trace_id and self._client:
                self._client.create_score(
                    trace_id=trace_id,
                    name=name,
                    value=value,
                    comment=comment,
                )
        except Exception as exc:
            logger.debug("Langfuse score error: %s", exc)

    def score_by_trace_id(
        self,
        trace_id: str,
        name: str,
        value: float,
        *,
        comment: Optional[str] = None,
    ) -> None:
        """Score an existing trace by its ID (useful for async / user feedback)."""
        if not self._enabled:
            return
        try:
            self._client.create_score(
                trace_id=trace_id,
                name=name,
                value=value,
                comment=comment,
            )
        except Exception as exc:
            logger.debug("Langfuse score_by_trace_id error: %s", exc)

    # ------------------------------------------------------------------
    # Convenience: user feedback
    # ------------------------------------------------------------------
    def record_user_feedback(
        self,
        trace_id: str,
        thumbs: int,
        *,
        rating: Optional[int] = None,
        comment: Optional[str] = None,
    ) -> None:
        """
        Record user feedback as Langfuse scores.

        Args:
            trace_id: The trace to attach feedback to.
            thumbs: 1 (positive) or -1 (negative).
            rating: Optional 1-5 star rating.
            comment: Free-text comment.
        """
        self.score_by_trace_id(trace_id, "user_thumbs", float(thumbs), comment=comment)
        if rating is not None:
            self.score_by_trace_id(trace_id, "user_rating", float(rating), comment=comment)

    # ------------------------------------------------------------------
    # Flush (for graceful shutdown)
    # ------------------------------------------------------------------
    def flush(self) -> None:
        if self._enabled and self._client:
            try:
                self._client.flush()
            except Exception:
                pass

    def shutdown(self) -> None:
        if self._enabled and self._client:
            try:
                self._client.shutdown()
            except Exception:
                pass


class _SpanCtx:
    """Helper returned by ``AgentTracer.span()`` context manager."""

    def __init__(self, span_obj: Any, start_time: float) -> None:
        self._span = span_obj
        self._start = start_time
        self._ended = False

    def end(
        self,
        *,
        output: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self._ended:
            return
        self._ended = True
        latency_ms = round((time.time() - self._start) * 1000, 1)
        meta = {"latency_ms": latency_ms}
        if metadata:
            meta.update(metadata)
        try:
            self._span.end(output=output, metadata=meta)
        except Exception:
            pass

    @property
    def span(self) -> Any:
        """Access the underlying Langfuse span object (for nesting)."""
        return self._span


class _GenCtx:
    """Helper returned by ``AgentTracer.generation()`` context manager."""

    def __init__(self, gen_obj: Any, start_time: float) -> None:
        self._gen = gen_obj
        self._start = start_time
        self._ended = False

    def end(
        self,
        *,
        output: Optional[Any] = None,
        usage: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self._ended:
            return
        self._ended = True
        latency_ms = round((time.time() - self._start) * 1000, 1)
        meta = {"latency_ms": latency_ms}
        if metadata:
            meta.update(metadata)
        try:
            self._gen.end(output=output, usage=usage, metadata=meta)
        except Exception:
            pass

    @property
    def generation(self) -> Any:
        """Access the underlying Langfuse generation object."""
        return self._gen


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_tracer: Optional[AgentTracer] = None


def get_tracer() -> AgentTracer:
    """Return the module-level AgentTracer singleton (lazy init)."""
    global _tracer
    if _tracer is None:
        _tracer = AgentTracer()
    return _tracer

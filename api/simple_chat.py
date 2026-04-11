"""
HTTP chat endpoint — thin adapter around the shared chat pipeline.

The WebSocket handler in ``websocket_wiki.py`` is the primary path.
This endpoint exists for clients that cannot use WebSockets.
"""

import logging
import time
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from api.data_pipeline import count_tokens

from api.prompts import (
    AGENT_CHAT_SYSTEM_PROMPT,
    DEEP_RESEARCH_BASE_PROMPT,
)
from api.rag import RAG
from api.agent.scheduler import AgentScheduler
from api.agent.react import ReActRunner
from api.agent.deep_research import DeepResearchOrchestrator, ResearchEventType
from api.agent.tools.search_tools import build_react_tools
from api.agent.llm_utils import create_llm_callable
from api.tracing import get_tracer

from api.chat_shared import (
    ChatMessage,
    ChatCompletionRequest,
    ThinkBlockFilter,
    infer_language_code,
    get_language_name,
    build_one_shot_deep_research_prompt,
    build_mcp_tools,
    parse_filter_params,
    retrieve_rag_context,
    fetch_file_content,
    build_conversation_history,
    assemble_prompt,
    build_memory_context,
    detect_modes,
    create_embed_fn,
)
from api.provider_factory import (
    create_model_and_kwargs,
    build_api_kwargs,
    stream_provider_response,
)

# Memory
from api.memory.manager import get_memory_manager
from api.memory.models import MemoryType, MemoryEntry

from api.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

agent_scheduler = AgentScheduler.default()
_memory_manager = get_memory_manager()

# FastAPI app (routes are re-registered on the main app in api.py)
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@app.post("/chat/completions/stream")
async def chat_completions_stream(request: ChatCompletionRequest):
    """Stream a chat completion response over HTTP (SSE-style)."""

    _start_time = time.time()
    user_id = request.user_id or "anonymous"
    repo_id = request.repo_url.replace("/", "_").replace(".", "_")[:50]
    session_key = f"{user_id}:{repo_id}"

    # Token-size guard
    input_too_large = False
    if request.messages:
        last_content = getattr(request.messages[-1], "content", "") or ""
        tokens = count_tokens(last_content, request.provider == "ollama")
        if tokens > 8000:
            logger.warning("Request %d tokens > 8000", tokens)
            input_too_large = True

    # Prepare RAG
    try:
        request_rag = RAG(provider=request.provider, model=request.model)
        ex_dirs, ex_files, in_dirs, in_files = parse_filter_params(
            request.excluded_dirs, request.excluded_files,
            request.included_dirs, request.included_files,
        )
        request_rag.prepare_retriever(
            request.repo_url, request.type, request.token,
            ex_dirs, ex_files, in_dirs, in_files,
        )
    except ValueError as e:
        detail = ("No valid document embeddings found."
                  if "No valid documents" in str(e) else str(e))
        raise HTTPException(status_code=500, detail=detail)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error preparing retriever: {e}")

    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided")
    if request.messages[-1].role != "user":
        raise HTTPException(status_code=400, detail="Last message must be from the user")

    # Conversation history into RAG memory
    for i in range(0, len(request.messages) - 1, 2):
        if i + 1 < len(request.messages):
            u, a = request.messages[i], request.messages[i + 1]
            if u.role == "user" and a.role == "assistant":
                request_rag.memory.add_dialog_turn(
                    user_query=u.content, assistant_response=a.content,
                )

    is_deep_research, is_agent_mode = detect_modes(request.messages)
    query = request.messages[-1].content

    # ==================================================================
    # Memory: load preferences & apply
    # ==================================================================
    prefs = _memory_manager.get_preferences(user_id, repo_id)
    if prefs.get("preferred_model", {}).get("value") and not request.model:
        request.model = prefs["preferred_model"]["value"]
    if prefs.get("preferred_provider", {}).get("value") and request.provider == "openai":
        request.provider = prefs["preferred_provider"]["value"]
    if prefs.get("preferred_language", {}).get("value") and not request.language:
        request.language = prefs["preferred_language"]["value"]

    _memory_manager.set_preference(user_id, repo_id, "preferred_model", {"value": request.model})
    _memory_manager.set_preference(user_id, repo_id, "preferred_provider", {"value": request.provider})
    _memory_manager.store(MemoryEntry.create(
        user_id=user_id, repo_id=repo_id,
        memory_type=MemoryType.INTERACTION, key="chat_query",
        value={"query_preview": query[:100], "timestamp": datetime.utcnow().isoformat(),
               "provider": request.provider, "model": request.model},
    ))

    # ==================================================================
    # Memory: episodic + long-term knowledge
    # ==================================================================
    episodic_ctx = _memory_manager.get_session_context(
        session_key, current_query=query, max_turns=3,
    )
    knowledge_entries = _memory_manager.search_knowledge(
        user_id, repo_id, query[:200], limit=3,
    )
    knowledge_context = ""
    if knowledge_entries:
        parts = []
        for ke in knowledge_entries:
            val = ke.value
            summary = val.get("summary", val.get("value", str(val))) if isinstance(val, dict) else str(val)
            parts.append(f"- {ke.key}: {summary[:300]}")
        knowledge_context = "\n".join(parts)

    memory_context_str = build_memory_context(episodic_ctx, knowledge_context)

    # Model (created early so llm_fn is available for tier-3 intent)
    model, model_kwargs = create_model_and_kwargs(request.provider, request.model)
    llm_fn = create_llm_callable(
        provider=request.provider, model=model, model_kwargs=model_kwargs,
    )

    # Scheduler — 3-tier intent classification (all modes)
    scheduled = None
    if not is_deep_research:
        embed_fn = create_embed_fn(request_rag)
        _enriched_scheduler = agent_scheduler.with_classifiers(
            embed_fn=embed_fn, llm_fn=llm_fn,
        )
        scheduled = await _enriched_scheduler.schedule_with_intent(
            query=query, language=request.language or "en",
        )

    # RAG context
    context_text = ""
    if not input_too_large:
        context_text = retrieve_rag_context(
            request_rag, query, request.language, request.filePath,
        )

    # Repo metadata
    repo_url = request.repo_url
    repo_name = repo_url.split("/")[-1] if "/" in repo_url else repo_url
    repo_type = request.type
    language_code = infer_language_code(
        query, request.language or "en",
    )
    language_name = get_language_name(language_code)

    # Tools
    rag_tools = build_react_tools(
        request_rag, language=language_code,
        repo_url=repo_url, repo_type=repo_type, token=request.token,
        user_id=user_id,
    )
    mcp_tools = build_mcp_tools(rag_tools)

    # ── Deep Research ─────────────────────────────────────────────
    if is_deep_research:
        tracer = get_tracer()
        trace = tracer.start_trace(
            name="deep_research", tags=["deep_research", "http"],
            metadata={"repo_id": repo_url, "provider": request.provider},
            input={"query": query},
        )

        dr_base_prompt = DEEP_RESEARCH_BASE_PROMPT.format(
            repo_type=repo_type, repo_url=repo_url,
            repo_name=repo_name, language_name=language_name,
        )
        # Inject memory into the DR base prompt
        if memory_context_str:
            dr_base_prompt += f"\n\n<memory_context>\n{memory_context_str}\n</memory_context>"

        orchestrator = DeepResearchOrchestrator(tools=mcp_tools, max_iterations=5)

        _start = time.time()

        async def _dr_stream():
            response_parts = []
            try:
                async for event in orchestrator.run(
                    query=query, base_system_prompt=dr_base_prompt,
                    initial_context=context_text, llm_fn=llm_fn,
                    language=language_code, trace=trace,
                ):
                    yield f"[RESEARCH_EVENT]{event.to_sse()}\n"
                    if event.event_type == ResearchEventType.CONCLUSION:
                        response_parts.append(event.data)
                        yield f"\n{event.data}"
                    elif event.event_type == ResearchEventType.ERROR:
                        yield f"\n⚠️ {event.data}\n"
            except Exception as exc:
                logger.error("DR stream error: %s", exc)
                yield f"\nError during deep research: {exc}"
                tracer.end_trace(trace, output={"error": str(exc)})
                return

            tracer.score(trace, "latency_ms", round((time.time() - _start) * 1000, 1))
            tracer.end_trace(trace, output={"mode": "deep_research"})

            # Persist memory
            _persist_memory(session_key, user_id, repo_id, query, "".join(response_parts))

        return StreamingResponse(_dr_stream(), media_type="text/event-stream")

    # ── Agent / Normal — ReAct (unified: normal = agent) ──────────────────
    export_hint = agent_scheduler.build_export_hint(query)
    system_prompt = AGENT_CHAT_SYSTEM_PROMPT.format(
        repo_type=repo_type, repo_url=repo_url,
        repo_name=repo_name, language_name=language_name,
        export_hint=export_hint,
    )

    tracer = get_tracer()
    trace = tracer.start_trace(
        name="agent_request", tags=["react" if is_agent_mode else "chat", "http"],
        metadata={"repo_id": repo_url, "provider": request.provider},
        input={"query": query},
    )
    runner = ReActRunner(tools=mcp_tools, max_iterations=4)
    _start = time.time()

    async def _react_stream():
        parts = []
        try:
            async for chunk in runner.run(
                query=query, system_prompt=system_prompt,
                initial_context=context_text, llm_fn=llm_fn,
                language=language_code, trace=trace,
            ):
                parts.append(chunk)
                yield chunk
        except Exception as e_react:
            logger.error("ReAct error: %s, falling back to direct LLM", e_react)
            try:
                file_content = fetch_file_content(
                    repo_url, request.filePath, request.type, request.token,
                )
                conversation_history = build_conversation_history(request_rag)
                prompt = assemble_prompt(
                    system_prompt=system_prompt,
                    conversation_history=conversation_history,
                    context_text=context_text,
                    query=query,
                    file_path=request.filePath,
                    file_content=file_content,
                    memory_context=memory_context_str,
                    provider=request.provider,
                )
                api_kw = build_api_kwargs(request.provider, model, model_kwargs, prompt)
                think_filter = ThinkBlockFilter()
                async for text in stream_provider_response(
                    request.provider, model, api_kw, prompt, think_filter,
                ):
                    parts.append(text)
                    yield text
            except Exception as fb_exc:
                logger.error("Direct LLM fallback failed: %s", fb_exc)
                yield f"\nError: {fb_exc}"
                tracer.end_trace(trace, output={"error": str(fb_exc)})
                return

        # Stage-2 export action
        full_resp = "".join(parts)
        stage2 = agent_scheduler.infer_second_stage_action(
            query=query, assistant_response=full_resp,
        )
        if stage2:
            parts.append(stage2)
            yield f"\n{stage2}"

        tracer.score(trace, "latency_ms", round((time.time() - _start) * 1000, 1))
        tracer.score(trace, "forced_termination", 0)
        tracer.end_trace(trace, output={"response_length": len("".join(parts))})

        # Persist memory
        _persist_memory(session_key, user_id, repo_id, query, "".join(parts))

    return StreamingResponse(_react_stream(), media_type="text/event-stream")


def _persist_memory(
    session_key: str, user_id: str, repo_id: str,
    query: str, full_response: str,
) -> None:
    """Persist conversation turn into memory (non-critical)."""
    try:
        if full_response.strip():
            topics = [w for w in query.split() if len(w) > 4][:5]
            _memory_manager.add_conversation_turn(
                session_key=session_key,
                user_query=query,
                assistant_response=full_response[:2000],
            )
            _memory_manager.track_session_turn(
                session_key=session_key, user_id=user_id, repo_id=repo_id,
                user_query=query, assistant_response=full_response[:2000],
                topics=topics,
            )
    except Exception as mem_err:
        logger.debug("Non-critical: memory persist failed: %s", mem_err)

"""
WebSocket chat handler — primary entry point for all chat modes.

Architecture: WebSocket is the main transport. The HTTP endpoint in
``simple_chat.py`` is a thin adapter that delegates here.

Modes:
  * **Normal** — ReAct loop (multi-step tool use), with scheduler-driven
    fast-path for simple queries or direct export invocations.
  * **Agent** — Same as Normal but with AGENT_CHAT_SYSTEM_PROMPT that
    includes export tool instructions.
  * **Deep Research** — Multi-phase orchestrator (decompose → investigate
    → gap-fill → synthesize).
"""

import logging
import time
from datetime import datetime
from typing import List, Optional

from fastapi import WebSocket, WebSocketDisconnect

from api.config import configs
from api.data_pipeline import count_tokens
from api.rag import RAG

from api.prompts import (
    AGENT_CHAT_SYSTEM_PROMPT,
    DEEP_RESEARCH_BASE_PROMPT,
)
from api.agent.scheduler import AgentScheduler
from api.agent.react import ReActRunner
from api.agent.deep_research import DeepResearchOrchestrator, ResearchEventType
from api.agent.tools.search_tools import build_react_tools
from api.agent.llm_utils import create_llm_callable

# Shared utilities (single source of truth)
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
    provider_error_hint,
)

# Memory & observability
from api.memory.manager import get_memory_manager
from api.memory.models import MemoryType, MemoryEntry
from api.monitoring.performance import get_performance_monitor
from api.tracing import get_tracer

from api.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

agent_scheduler = AgentScheduler.default()
_memory_manager = get_memory_manager()
_performance_monitor = get_performance_monitor()


# ---------------------------------------------------------------------------
# Main WebSocket handler
# ---------------------------------------------------------------------------

async def handle_websocket_chat(websocket: WebSocket):
    """Handle WebSocket connection for all chat modes with memory integration."""
    await websocket.accept()

    # Defaults (referenced in the finally block)
    _response_chunks: list = []
    session_key = ""
    query = ""
    user_id = "anonymous"
    repo_id = ""
    _langfuse_trace = None
    _ws_start_time = time.time()
    _tracer = get_tracer()

    try:
        # Parse incoming request
        request_data = await websocket.receive_json()
        request = ChatCompletionRequest(**request_data)

        user_id = request.user_id or "anonymous"
        repo_id = request.repo_url.replace("/", "_").replace(".", "_")[:50]
        session_key = f"{user_id}:{repo_id}"

        logger.info("WebSocket request user=%s repo=%s", user_id, repo_id)

        # ==================================================================
        # Langfuse trace
        # ==================================================================
        _langfuse_trace = _tracer.start_trace(
            name="agent_request",
            user_id=user_id,
            session_id=session_key,
            metadata={
                "repo_id": repo_id,
                "repo_url": request.repo_url,
                "language": request.language,
                "provider": request.provider,
                "model": request.model,
            },
            tags=["websocket"],
            input={"messages_count": len(request.messages) if request.messages else 0},
        )

        # ==================================================================
        # Token-size guard
        # ==================================================================
        input_too_large = False
        if request.messages:
            last_content = getattr(request.messages[-1], "content", "") or ""
            tokens = count_tokens(last_content, request.provider == "ollama")
            if tokens > 8000:
                logger.warning("Request %d tokens > 8000", tokens)
                input_too_large = True

        # ==================================================================
        # Prepare RAG
        # ==================================================================
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
            err = "No valid document embeddings found." if "No valid documents" in str(e) else str(e)
            await websocket.send_text(f"Error: {err}")
            await websocket.close()
            return
        except Exception as e:
            await websocket.send_text(f"Error preparing retriever: {e}")
            await websocket.close()
            return

        # Validate messages
        if not request.messages:
            await websocket.send_text("Error: No messages provided")
            await websocket.close()
            return
        if request.messages[-1].role != "user":
            await websocket.send_text("Error: Last message must be from the user")
            await websocket.close()
            return

        # Build conversation memory from previous turns
        for i in range(0, len(request.messages) - 1, 2):
            if i + 1 < len(request.messages):
                u, a = request.messages[i], request.messages[i + 1]
                if u.role == "user" and a.role == "assistant":
                    request_rag.memory.add_dialog_turn(
                        user_query=u.content, assistant_response=a.content,
                    )

        # Detect mode tags
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
        with _tracer.span(_langfuse_trace, "memory_operations",
                          metadata={"user_id": user_id, "repo_id": repo_id}) as _mem_ctx:
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
            _mem_ctx.end(output={
                "episodic_turns": len(episodic_ctx.get("recent_turns", [])) if isinstance(episodic_ctx, dict) else 0,
                "knowledge_entries": len(knowledge_entries) if knowledge_entries else 0,
            })

        memory_context_str = build_memory_context(episodic_ctx, knowledge_context)

        # Response accumulator
        _response_chunks = []

        async def _send(text: str) -> None:
            await websocket.send_text(text)
            _response_chunks.append(text)

        # ==================================================================
        # Create model + kwargs via unified factory (moved before scheduler
        # so that llm_fn is available for tier-3 intent classification)
        # ==================================================================
        model, model_kwargs = create_model_and_kwargs(request.provider, request.model)
        llm_fn = create_llm_callable(
            provider=request.provider, model=model, model_kwargs=model_kwargs,
        )

        # ==================================================================
        # Scheduler — 3-tier intent classification (all modes)
        # ==================================================================
        scheduled = None
        if not is_deep_research:
            embed_fn = create_embed_fn(request_rag)
            with _tracer.span(_langfuse_trace, "intent_scheduling", input={"query": query}) as _sc:
                _enriched_scheduler = agent_scheduler.with_classifiers(
                    embed_fn=embed_fn, llm_fn=llm_fn,
                )
                scheduled = await _enriched_scheduler.schedule_with_intent(
                    query=query, language=request.language or "en",
                )
                for ev in scheduled.events:
                    logger.info("schedule: %s %s tool=%s", ev.event_type.value, ev.message, ev.tool_name)
                _sc.end(output={
                    "handled": scheduled.handled,
                    "events": len(scheduled.events),
                    "tool": next((e.tool_name for e in scheduled.events if e.tool_name), None),
                    "intent": scheduled.intent_result.intent if scheduled.intent_result else None,
                })

            # Ambiguity → ask user to clarify (no action tag)
            if scheduled.handled and scheduled.content and "[ACTION:" not in scheduled.content:
                await websocket.send_text(scheduled.content)
                await websocket.close()
                return

            # Direct export invocation (scheduler resolved a single tool)
            if scheduled.handled and scheduled.content and "[ACTION:" in scheduled.content:
                await _send(scheduled.content)
                await websocket.close()
                return

        # ==================================================================
        # RAG context retrieval
        # ==================================================================
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
            query, request.language or configs["lang_config"]["default"],
        )
        language_name = get_language_name(language_code)

        # Build tools (shared by ReAct + DR)
        rag_tools = build_react_tools(
            request_rag,
            language=language_code,
            repo_url=repo_url,
            repo_type=repo_type,
            token=request.token,
            user_id=user_id,
        )
        mcp_tools = build_mcp_tools(rag_tools)

        # ==============================================================
        # PATH: Deep Research
        # ==============================================================
        if is_deep_research:
            logger.info("Deep Research orchestrator mode")
            try:
                dr_base_prompt = DEEP_RESEARCH_BASE_PROMPT.format(
                    repo_type=repo_type, repo_url=repo_url,
                    repo_name=repo_name, language_name=language_name,
                )
                # Inject memory into the DR base prompt
                if memory_context_str:
                    dr_base_prompt += f"\n\n<memory_context>\n{memory_context_str}\n</memory_context>"

                orchestrator = DeepResearchOrchestrator(tools=mcp_tools, max_iterations=5)

                async for event in orchestrator.run(
                    query=query,
                    base_system_prompt=dr_base_prompt,
                    initial_context=context_text,
                    llm_fn=llm_fn,
                    language=language_code,
                    trace=_langfuse_trace,
                ):
                    await _send(f"[RESEARCH_EVENT]{event.to_sse()}\n")
                    if event.event_type == ResearchEventType.CONCLUSION:
                        await _send(f"\n{event.data}")

                await websocket.close()
                return
            except Exception as dr_exc:
                logger.error("DR orchestrator failed: %s, falling back to one-shot", dr_exc)
                # Fall through to one-shot LLM below

        # ==============================================================
        # PATH: Normal / Agent — ReAct loop (unified: normal = agent)
        # ==============================================================
        if is_deep_research:
            # DR orchestrator failed; use one-shot prompt
            system_prompt = build_one_shot_deep_research_prompt(
                repo_type=repo_type, repo_url=repo_url,
                repo_name=repo_name, language_name=language_name,
            )
        else:
            system_prompt = AGENT_CHAT_SYSTEM_PROMPT.format(
                repo_type=repo_type, repo_url=repo_url,
                repo_name=repo_name, language_name=language_name,
            )

        # ── ReAct loop ──────────────────────────────────────────────
        runner = ReActRunner(tools=mcp_tools, max_iterations=4)

        try:
            async for chunk in runner.run(
                query=query,
                system_prompt=system_prompt,
                initial_context=context_text,
                llm_fn=llm_fn,
                language=language_code,
                trace=_langfuse_trace,
            ):
                await _send(chunk)

            # Stage-2: check if LLM response recommends an export
            if not is_deep_research:
                full_resp = "".join(_response_chunks)
                stage2 = agent_scheduler.infer_second_stage_action(
                    query=query, assistant_response=full_resp,
                )
                if stage2:
                    await _send(f"\n{stage2}")

        except Exception as e_react:
            logger.error("ReAct failed: %s, falling back to direct LLM", e_react)
            # Fallback: build full prompt and stream directly
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
                    await _send(text)
            except Exception as fb_exc:
                logger.error("Direct LLM fallback also failed: %s", fb_exc)
                err_msg = ("抱歉，处理您的请求时遇到问题。请重试。"
                           if language_code.startswith("zh")
                           else "Sorry, there was an issue. Please try again.")
                await _send(err_msg)

        await websocket.close()

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error("WebSocket handler error: %s", e)
        try:
            await websocket.send_text(f"Error: {e}")
            await websocket.close()
        except Exception:
            pass
    finally:
        # Persist conversation turn into memory
        try:
            if _response_chunks and query:
                full_response = "".join(_response_chunks)
                if full_response.strip():
                    _topics = [w for w in query.split() if len(w) > 4][:5]
                    _memory_manager.add_conversation_turn(
                        session_key=session_key,
                        user_query=query,
                        assistant_response=full_response[:2000],
                    )
                    _memory_manager.track_session_turn(
                        session_key=session_key, user_id=user_id, repo_id=repo_id,
                        user_query=query, assistant_response=full_response[:2000],
                        topics=_topics,
                    )
        except Exception as mem_err:
            logger.debug("Non-critical: memory persist failed: %s", mem_err)

        # End Langfuse trace
        try:
            if _langfuse_trace is not None:
                _latency = round((time.time() - _ws_start_time) * 1000, 1)
                _tracer.score(_langfuse_trace, "latency_ms", _latency)
                _tracer.score(_langfuse_trace, "forced_termination", 0)
                _tracer.end_trace(
                    _langfuse_trace,
                    output={"response_length": len("".join(_response_chunks))},
                )
        except Exception:
            pass

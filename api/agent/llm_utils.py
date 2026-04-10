"""
Provider-agnostic LLM callable factory.

Creates an ``async (prompt: str) -> str`` function that works with any
supported provider so the ReAct runner can call the LLM without
knowing provider-specific details.
"""

import asyncio
import logging
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)


async def _collect_streaming_response(provider: str, response) -> str:
    """Consume a provider-specific streaming response into a plain string."""
    parts = []

    if provider == "ollama":
        async for chunk in response:
            # Chat API: content is in chunk.message.content
            # Generate API: content is in chunk.response
            text = ""
            msg = getattr(chunk, "message", None)
            if msg is not None:
                text = getattr(msg, "content", None) or ""
            if not text:
                text = getattr(chunk, "response", None) or getattr(chunk, "text", None) or ""
            if text and not text.startswith("model=") and not text.startswith("created_at="):
                parts.append(text.replace("<think>", "").replace("</think>", ""))

    elif provider == "openrouter":
        async for chunk in response:
            parts.append(chunk if isinstance(chunk, str) else str(chunk))

    elif provider in ("openai", "azure"):
        async for chunk in response:
            choices = getattr(chunk, "choices", [])
            if choices:
                delta = getattr(choices[0], "delta", None)
                if delta:
                    text = getattr(delta, "content", None)
                    if text:
                        parts.append(text)

    elif provider == "bedrock":
        if isinstance(response, str):
            parts.append(response)
        else:
            parts.append(str(response))

    elif provider == "dashscope":
        async for text in response:
            if text:
                parts.append(text)

    else:
        # Google or unknown — sync iterator
        for chunk in response:
            if hasattr(chunk, "text"):
                parts.append(chunk.text)

    return "".join(parts)


def create_llm_callable(
    provider: str,
    model,
    model_kwargs: dict,
) -> Callable[[str], Awaitable[str]]:
    """
    Return an ``async (prompt: str) -> str`` callable that invokes the
    configured LLM and collects the full response text.

    For Google GenAI the call is offloaded to a thread because the SDK
    is synchronous.
    """

    async def llm_fn(prompt: str) -> str:
        from adalflow.core.types import ModelType

        # Google uses a different API surface
        if provider == "google":
            import google.generativeai as genai

            def _sync_call():
                return model.generate_content(prompt)

            resp = await asyncio.to_thread(_sync_call)
            return resp.text

        # All other providers share the adalflow interface
        call_kwargs = {**model_kwargs, "stream": True}
        api_kwargs = model.convert_inputs_to_api_kwargs(
            input=prompt,
            model_kwargs=call_kwargs,
            model_type=ModelType.LLM,
        )
        response = await model.acall(api_kwargs=api_kwargs, model_type=ModelType.LLM)
        return await _collect_streaming_response(provider, response)

    return llm_fn

"""
Unified provider factory — single source of truth for creating LLM model
instances, building ``model_kwargs``, and extracting text from streaming
chunks across all supported providers.

Every call‑site (websocket_wiki, simple_chat, deep_research, etc.) should
use these helpers instead of duplicating per‑provider switch‑cases.
"""

import logging
from typing import Any, AsyncIterator, Optional, Tuple

import google.generativeai as genai
from adalflow.components.model_client.ollama_client import OllamaClient
from adalflow.core.types import ModelType

from api.config import (
    get_model_config,
    OPENROUTER_API_KEY,
    OPENAI_API_KEY,
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
)
from api.bedrock_client import BedrockClient
from api.openai_client import OpenAIClient
from api.openrouter_client import OpenRouterClient
from api.azureai_client import AzureAIClient
from api.dashscope_client import DashscopeClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def create_model_and_kwargs(
    provider: str,
    model_name: Optional[str],
) -> Tuple[Any, dict]:
    """Return ``(model_client, model_kwargs)`` for the given provider.

    ``model_kwargs`` is ready-to-use with ``convert_inputs_to_api_kwargs``
    for adalflow-based providers, or empty for Google GenAI.
    """
    model_config = get_model_config(provider, model_name)["model_kwargs"]

    if provider == "ollama":
        model = OllamaClient()
        model_kwargs = {
            "model": model_config["model"],
            "stream": True,
            "options": {
                "temperature": model_config["temperature"],
                "top_p": model_config["top_p"],
                "num_ctx": model_config["num_ctx"],
            },
        }
    elif provider == "openrouter":
        if not OPENROUTER_API_KEY:
            logger.warning("OPENROUTER_API_KEY not configured")
        model = OpenRouterClient()
        model_kwargs = {
            "model": model_name,
            "stream": True,
            "temperature": model_config["temperature"],
        }
        if "top_p" in model_config:
            model_kwargs["top_p"] = model_config["top_p"]
    elif provider == "openai":
        if not OPENAI_API_KEY:
            logger.warning("OPENAI_API_KEY not configured")
        model = OpenAIClient()
        model_kwargs = {
            "model": model_name,
            "stream": True,
            "temperature": model_config["temperature"],
        }
        if "top_p" in model_config:
            model_kwargs["top_p"] = model_config["top_p"]
    elif provider == "bedrock":
        if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
            logger.warning("AWS credentials not configured")
        model = BedrockClient()
        model_kwargs = {"model": model_name}
        for k in ("temperature", "top_p"):
            if k in model_config:
                model_kwargs[k] = model_config[k]
    elif provider == "azure":
        model = AzureAIClient()
        model_kwargs = {
            "model": model_name,
            "stream": True,
            "temperature": model_config["temperature"],
            "top_p": model_config["top_p"],
        }
    elif provider == "dashscope":
        model = DashscopeClient()
        model_kwargs = {
            "model": model_name,
            "stream": True,
            "temperature": model_config["temperature"],
            "top_p": model_config["top_p"],
        }
    else:
        # Google GenAI (default)
        model = genai.GenerativeModel(
            model_name=model_config["model"],
            generation_config={
                "temperature": model_config["temperature"],
                "top_p": model_config["top_p"],
                "top_k": model_config["top_k"],
            },
        )
        model_kwargs = {}

    return model, model_kwargs


def build_api_kwargs(provider: str, model: Any, model_kwargs: dict, prompt: str) -> dict:
    """Build ``api_kwargs`` for an adalflow ``acall``.

    For Google GenAI the prompt is baked into ``generate_content`` directly,
    so we return an empty dict.
    """
    if provider == "google" or not model_kwargs:
        return {}

    if provider == "ollama":
        prompt = prompt + " /no_think"

    return model.convert_inputs_to_api_kwargs(
        input=prompt,
        model_kwargs=model_kwargs,
        model_type=ModelType.LLM,
    )


async def stream_provider_response(
    provider: str,
    model: Any,
    api_kwargs: dict,
    prompt: str,
    think_filter: Any = None,
) -> AsyncIterator[str]:
    """Yield text chunks from a streaming LLM call.

    ``think_filter`` should be a ``_ThinkBlockFilter`` instance (or None to
    skip filtering).  The caller is responsible for constructing
    ``api_kwargs`` via :func:`build_api_kwargs`.
    """

    def _apply_filter(text: str) -> str:
        if think_filter is None:
            return text
        return think_filter.feed(text)

    if provider == "google" or not api_kwargs:
        # Google GenAI — synchronous iterator
        response = model.generate_content(prompt, stream=True)
        for chunk in response:
            if hasattr(chunk, "text"):
                clean = _apply_filter(chunk.text)
                if clean:
                    yield clean
        return

    response = await model.acall(api_kwargs=api_kwargs, model_type=ModelType.LLM)

    if provider == "ollama":
        async for chunk in response:
            text = None
            if isinstance(chunk, dict):
                text = chunk.get("message", {}).get("content") if isinstance(chunk.get("message"), dict) else chunk.get("message")
            else:
                message = getattr(chunk, "message", None)
                if message is not None:
                    text = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
            if not text:
                text = getattr(chunk, "response", None) or getattr(chunk, "text", None)
            if not text and hasattr(chunk, "__dict__"):
                message = chunk.__dict__.get("message")
                if isinstance(message, dict):
                    text = message.get("content")
            if isinstance(text, str) and text and not text.startswith("model=") and not text.startswith("created_at="):
                clean = _apply_filter(text)
                if clean:
                    yield clean

    elif provider == "openrouter":
        async for chunk in response:
            text = chunk if isinstance(chunk, str) else str(chunk)
            clean = _apply_filter(text)
            if clean:
                yield clean

    elif provider in ("openai", "azure"):
        async for chunk in response:
            choices = getattr(chunk, "choices", [])
            if choices:
                delta = getattr(choices[0], "delta", None)
                if delta:
                    text = getattr(delta, "content", None)
                    if text is not None:
                        clean = _apply_filter(text)
                        if clean:
                            yield clean

    elif provider == "bedrock":
        if isinstance(response, str):
            clean = _apply_filter(response)
            if clean:
                yield clean
        else:
            clean = _apply_filter(str(response))
            if clean:
                yield clean

    elif provider == "dashscope":
        async for text in response:
            if text:
                clean = _apply_filter(text)
                if clean:
                    yield clean
    else:
        # Unknown provider fallback
        async for chunk in response:
            text = str(chunk)
            clean = _apply_filter(text)
            if clean:
                yield clean


def provider_error_hint(provider: str) -> str:
    """Return a user-facing hint for a provider configuration error."""
    hints = {
        "ollama": "Check that Ollama is running locally.",
        "openrouter": "Check OPENROUTER_API_KEY environment variable.",
        "openai": "Check OPENAI_API_KEY environment variable.",
        "bedrock": "Check AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables.",
        "azure": "Check AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, and AZURE_OPENAI_VERSION environment variables.",
        "dashscope": "Check DASHSCOPE_API_KEY environment variable.",
        "google": "Check GOOGLE_API_KEY environment variable.",
    }
    return hints.get(provider, f"Check credentials for provider '{provider}'.")

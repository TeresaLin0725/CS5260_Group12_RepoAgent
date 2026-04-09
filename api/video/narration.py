"""Phase 2b: LLM narration script generation and parsing."""

import json
import logging
import re
import time
from typing import TYPE_CHECKING, List, Optional

from api.video.constants import NARRATION_MAX_CHARS
from api.video.text_utils import _truncate_narration

if TYPE_CHECKING:
    from api.content_analyzer import AnalyzedContent

logger = logging.getLogger(__name__)


def _analysis_to_prompt_json(analyzed: "AnalyzedContent") -> str:
    """Serialize structured analysis into compact JSON for the prompt."""
    payload = {
        "repo_name": analyzed.repo_name,
        "repo_type_hint": analyzed.repo_type_hint,
        "project_overview": analyzed.project_overview,
        "architecture": analyzed.architecture,
        "tech_stack": {
            "languages": analyzed.tech_stack.languages,
            "frameworks": analyzed.tech_stack.frameworks,
            "key_libraries": analyzed.tech_stack.key_libraries,
            "infrastructure": analyzed.tech_stack.infrastructure,
        },
        "key_modules": [{"name": m.name, "responsibility": m.responsibility} for m in analyzed.key_modules],
        "data_flow": analyzed.data_flow,
        "api_points": analyzed.api_points,
        "target_users": analyzed.target_users,
        "module_progression": [
            {
                "name": m.name,
                "stage": m.stage,
                "role": m.role,
                "solves": m.solves,
                "position": m.position,
            }
            for m in analyzed.module_progression
        ],
    }
    if analyzed.deployment_info:
        payload["deployment_info"] = analyzed.deployment_info
    if analyzed.component_hierarchy:
        payload["component_hierarchy"] = analyzed.component_hierarchy
    if analyzed.data_schemas:
        payload["data_schemas"] = analyzed.data_schemas

    return json.dumps(payload, ensure_ascii=False, indent=2)


async def generate_narration_script(
    analyzed: "AnalyzedContent",
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> List[dict]:
    """Call the LLM with VIDEO_NARRATION_PROMPT to produce a narration script."""
    from api.content_analyzer import _get_language_name
    from api.prompts import VIDEO_NARRATION_PROMPT
    from api.video.storyline import _fallback_narration_script

    start = time.perf_counter()
    language_name = _get_language_name(analyzed.language)
    analysis_json = _analysis_to_prompt_json(analyzed)
    prompt = VIDEO_NARRATION_PROMPT.format(
        language_name=language_name,
        analysis_json=analysis_json,
    )
    logger.info("Timing - video prompt assembly completed in %.2fs", time.perf_counter() - start)

    llm_start = time.perf_counter()
    raw_text = await _call_llm_raw(
        prompt=prompt,
        provider=provider or "openai",
        model=model,
    )
    logger.info("Timing - narration LLM call completed in %.2fs", time.perf_counter() - llm_start)

    parse_start = time.perf_counter()
    scenes = _parse_scene_array(raw_text)
    logger.info("Timing - narration parsing completed in %.2fs", time.perf_counter() - parse_start)
    if not scenes:
        fallback_start = time.perf_counter()
        scenes = _fallback_narration_script(analyzed)
        logger.info("Timing - narration fallback script built in %.2fs", time.perf_counter() - fallback_start)

    logger.info("Narration script generated - %d scenes (total %.2fs)", len(scenes), time.perf_counter() - start)
    return scenes


# TODO: Extract to shared api/llm_dispatch.py (duplicated in content_analyzer.py)
async def _call_llm_raw(prompt: str, provider: str, model: Optional[str]) -> str:
    """Dispatch a non-streaming LLM call and return the raw text response."""
    from adalflow.core.types import ModelType
    from api.config import get_model_config
    from api.content_analyzer import _extract_text_from_response

    config = get_model_config(provider, model)
    model_kwargs = config["model_kwargs"]

    if provider == "ollama":
        from adalflow.components.model_client.ollama_client import OllamaClient

        client = OllamaClient()
        kwargs = {
            "model": model_kwargs["model"],
            "stream": False,
            "options": {k: v for k, v in model_kwargs.items() if k != "model"},
        }
        api_kwargs = client.convert_inputs_to_api_kwargs(
            input=prompt, model_kwargs=kwargs, model_type=ModelType.LLM,
        )
        response = await client.acall(api_kwargs=api_kwargs, model_type=ModelType.LLM)
    elif provider == "openrouter":
        from api.openrouter_client import OpenRouterClient

        client = OpenRouterClient()
        kwargs = {
            "model": model_kwargs["model"],
            "stream": False,
            "temperature": model_kwargs.get("temperature", 0.5),
        }
        api_kwargs = client.convert_inputs_to_api_kwargs(
            input=prompt, model_kwargs=kwargs, model_type=ModelType.LLM,
        )
        response = await client.acall(api_kwargs=api_kwargs, model_type=ModelType.LLM)
    elif provider == "openai":
        from api.openai_client import OpenAIClient

        client = OpenAIClient()
        kwargs = {
            "model": model_kwargs["model"],
            "stream": False,
            "temperature": model_kwargs.get("temperature", 0.5),
        }
        api_kwargs = client.convert_inputs_to_api_kwargs(
            input=prompt, model_kwargs=kwargs, model_type=ModelType.LLM,
        )
        response = await client.acall(api_kwargs=api_kwargs, model_type=ModelType.LLM)
    elif provider == "bedrock":
        from api.bedrock_client import BedrockClient

        client = BedrockClient()
        kwargs = {
            "model": model_kwargs["model"],
            "temperature": model_kwargs.get("temperature", 0.5),
            "top_p": model_kwargs.get("top_p", 0.9),
        }
        api_kwargs = client.convert_inputs_to_api_kwargs(
            input=prompt, model_kwargs=kwargs, model_type=ModelType.LLM,
        )
        response = await client.acall(api_kwargs=api_kwargs, model_type=ModelType.LLM)
    elif provider == "azure":
        from api.azureai_client import AzureAIClient

        client = AzureAIClient()
        kwargs = {
            "model": model_kwargs["model"],
            "stream": False,
            "temperature": model_kwargs.get("temperature", 0.5),
            "top_p": model_kwargs.get("top_p", 0.9),
        }
        api_kwargs = client.convert_inputs_to_api_kwargs(
            input=prompt, model_kwargs=kwargs, model_type=ModelType.LLM,
        )
        response = await client.acall(api_kwargs=api_kwargs, model_type=ModelType.LLM)
    elif provider == "dashscope":
        from api.dashscope_client import DashscopeClient

        client = DashscopeClient()
        kwargs = {
            "model": model_kwargs["model"],
            "stream": False,
            "temperature": model_kwargs.get("temperature", 0.5),
            "top_p": model_kwargs.get("top_p", 0.9),
        }
        api_kwargs = client.convert_inputs_to_api_kwargs(
            input=prompt, model_kwargs=kwargs, model_type=ModelType.LLM,
        )
        response = await client.acall(api_kwargs=api_kwargs, model_type=ModelType.LLM)
    else:
        import google.generativeai as genai

        gen_model = genai.GenerativeModel(
            model_name=model_kwargs.get("model", "gemini-2.0-flash"),
            generation_config={
                "temperature": model_kwargs.get("temperature", 0.5),
                "top_p": model_kwargs.get("top_p", 0.9),
                "top_k": model_kwargs.get("top_k", 40),
            },
        )
        response = await gen_model.generate_content_async(prompt)
        return response.text if hasattr(response, "text") else str(response)

    return _extract_text_from_response(response, provider)


def _parse_scene_array(raw_text: str) -> List[dict]:
    """Parse a JSON array of scene objects from raw LLM text."""
    cleaned = re.sub(r"```(?:json)?\s*", "", raw_text)
    cleaned = re.sub(r"```\s*$", "", cleaned)

    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    match = re.search(r"\[", cleaned)
    if match:
        depth = 0
        start = match.start()
        for i in range(start, len(cleaned)):
            if cleaned[i] == "[":
                depth += 1
            elif cleaned[i] == "]":
                depth -= 1
                if depth == 0:
                    try:
                        data = json.loads(cleaned[start : i + 1])
                        if isinstance(data, list):
                            return data
                    except json.JSONDecodeError:
                        break

    return []

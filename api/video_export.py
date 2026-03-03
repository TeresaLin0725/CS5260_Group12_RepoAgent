"""
Video Export Module — Video Renderer

Converts structured AnalyzedContent into a video presentation.

Phase 2b-Video: Uses an LLM call (VIDEO_NARRATION_PROMPT) to rewrite the
    dry structured analysis into a conversational narration script with
    scene titles and duration hints.
Phase 3: Renders the narration script into an MP4 video.

NOTE: The Phase 3 renderer (actual video composition) is a stub that raises
NotImplementedError. To activate it, install moviepy/ffmpeg and implement
the frame composition logic below.

The Phase 2b narration-script generation is fully functional — it produces
a JSON array of scenes that can be consumed by any video pipeline.
"""

import json
import logging
from typing import Any, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from api.content_analyzer import AnalyzedContent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase 2b-Video: structured AnalyzedContent → narration script (LLM call)
# ---------------------------------------------------------------------------

def _analysis_to_prompt_json(analyzed: "AnalyzedContent") -> str:
    """Serialize the structured analysis into a compact JSON string for the prompt."""
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
    """
    Phase 2b-Video: call the LLM with VIDEO_NARRATION_PROMPT to produce a
    narration script.

    Returns a list of scene dicts:
        [{"title": str, "narration": str, "duration_seconds": int}, ...]
    """
    from api.content_analyzer import _get_language_name, _extract_json_from_llm

    language_name = _get_language_name(analyzed.language)
    analysis_json = _analysis_to_prompt_json(analyzed)

    from api.prompts import VIDEO_NARRATION_PROMPT

    prompt = VIDEO_NARRATION_PROMPT.format(
        language_name=language_name,
        analysis_json=analysis_json,
    )

    # Use the same LLM dispatch as content_analyzer
    from api.content_analyzer import _run_llm_structured_analysis

    # We reuse the generic LLM caller — the prompt is different but the
    # plumbing (provider dispatch) is identical.
    raw_text = await _call_llm_raw(
        prompt=prompt,
        provider=provider or "openai",
        model=model,
    )

    # Parse JSON array
    scenes = _parse_scene_array(raw_text)
    if not scenes:
        # Fallback: build a basic script from the structured data
        scenes = _fallback_narration_script(analyzed)

    logger.info("Narration script generated — %d scenes", len(scenes))
    return scenes


async def _call_llm_raw(prompt: str, provider: str, model: Optional[str]) -> str:
    """Dispatch a non-streaming LLM call and return the raw text response."""
    from api.config import get_model_config
    from adalflow.core.types import ModelType

    config = get_model_config(provider, model)
    model_kwargs = config["model_kwargs"]

    if provider == "ollama":
        from adalflow.components.model_client.ollama_client import OllamaClient
        client = OllamaClient()
        kwargs = {"model": model_kwargs["model"], "stream": False,
                  "options": {k: v for k, v in model_kwargs.items() if k != "model"}}
        api_kwargs = client.convert_inputs_to_api_kwargs(input=prompt, model_kwargs=kwargs, model_type=ModelType.LLM)
        response = await client.acall(api_kwargs=api_kwargs, model_type=ModelType.LLM)
    elif provider == "openrouter":
        from api.openrouter_client import OpenRouterClient
        client = OpenRouterClient()
        kwargs = {"model": model_kwargs["model"], "stream": False, "temperature": model_kwargs.get("temperature", 0.5)}
        api_kwargs = client.convert_inputs_to_api_kwargs(input=prompt, model_kwargs=kwargs, model_type=ModelType.LLM)
        response = await client.acall(api_kwargs=api_kwargs, model_type=ModelType.LLM)
    elif provider == "openai":
        from api.openai_client import OpenAIClient
        client = OpenAIClient()
        kwargs = {"model": model_kwargs["model"], "stream": False, "temperature": model_kwargs.get("temperature", 0.5)}
        api_kwargs = client.convert_inputs_to_api_kwargs(input=prompt, model_kwargs=kwargs, model_type=ModelType.LLM)
        response = await client.acall(api_kwargs=api_kwargs, model_type=ModelType.LLM)
    elif provider == "bedrock":
        from api.bedrock_client import BedrockClient
        client = BedrockClient()
        kwargs = {"model": model_kwargs["model"], "temperature": model_kwargs.get("temperature", 0.5),
                  "top_p": model_kwargs.get("top_p", 0.9)}
        api_kwargs = client.convert_inputs_to_api_kwargs(input=prompt, model_kwargs=kwargs, model_type=ModelType.LLM)
        response = await client.acall(api_kwargs=api_kwargs, model_type=ModelType.LLM)
    elif provider == "azure":
        from api.azureai_client import AzureAIClient
        client = AzureAIClient()
        kwargs = {"model": model_kwargs["model"], "stream": False,
                  "temperature": model_kwargs.get("temperature", 0.5), "top_p": model_kwargs.get("top_p", 0.9)}
        api_kwargs = client.convert_inputs_to_api_kwargs(input=prompt, model_kwargs=kwargs, model_type=ModelType.LLM)
        response = await client.acall(api_kwargs=api_kwargs, model_type=ModelType.LLM)
    elif provider == "dashscope":
        from api.dashscope_client import DashscopeClient
        client = DashscopeClient()
        kwargs = {"model": model_kwargs["model"], "stream": False,
                  "temperature": model_kwargs.get("temperature", 0.5), "top_p": model_kwargs.get("top_p", 0.9)}
        api_kwargs = client.convert_inputs_to_api_kwargs(input=prompt, model_kwargs=kwargs, model_type=ModelType.LLM)
        response = await client.acall(api_kwargs=api_kwargs, model_type=ModelType.LLM)
    else:
        # Google Generative AI
        import google.generativeai as genai
        gen_model = genai.GenerativeModel(
            model_name=model_kwargs.get("model", "gemini-2.0-flash"),
            generation_config={"temperature": model_kwargs.get("temperature", 0.5),
                               "top_p": model_kwargs.get("top_p", 0.9),
                               "top_k": model_kwargs.get("top_k", 40)},
        )
        response = await gen_model.generate_content_async(prompt)
        return response.text if hasattr(response, "text") else str(response)

    from api.content_analyzer import _extract_text_from_response
    return _extract_text_from_response(response, provider)


def _parse_scene_array(raw_text: str) -> List[dict]:
    """Parse a JSON array of scene objects from raw LLM text."""
    import re
    cleaned = re.sub(r"```(?:json)?\s*", "", raw_text)
    cleaned = re.sub(r"```\s*$", "", cleaned)

    # Try the whole text
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # Find first [ ... ] block
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


def _fallback_narration_script(analyzed: "AnalyzedContent") -> List[dict]:
    """Build a simple narration script directly from the structured data."""
    scenes: list[dict] = []

    scenes.append({
        "title": analyzed.repo_name,
        "narration": f"Welcome! Let's walk through {analyzed.repo_name}. {analyzed.project_overview[:200] if analyzed.project_overview else ''}",
        "duration_seconds": 15,
    })

    if analyzed.architecture:
        narration = "Here's the high-level architecture. " + " ".join(analyzed.architecture[:3])
        scenes.append({"title": "Architecture", "narration": narration[:400], "duration_seconds": 20})

    if analyzed.tech_stack and (analyzed.tech_stack.languages or analyzed.tech_stack.frameworks):
        items = analyzed.tech_stack.languages + analyzed.tech_stack.frameworks
        narration = "The project uses: " + ", ".join(items[:6]) + "."
        scenes.append({"title": "Tech Stack", "narration": narration, "duration_seconds": 15})

    if analyzed.key_modules:
        mods = ", ".join(m.name for m in analyzed.key_modules[:5])
        narration = f"The key modules are: {mods}."
        scenes.append({"title": "Key Modules", "narration": narration, "duration_seconds": 15})

    if analyzed.data_flow:
        narration = "Let's trace the data flow. " + " Then, ".join(analyzed.data_flow[:4]) + "."
        scenes.append({"title": "Data Flow", "narration": narration[:400], "duration_seconds": 20})

    scenes.append({
        "title": "Thank You",
        "narration": f"That's a quick overview of {analyzed.repo_name}. Thanks for watching!",
        "duration_seconds": 10,
    })

    return scenes


# ---------------------------------------------------------------------------
# Phase 3: Video render (stub — requires moviepy / ffmpeg)
# ---------------------------------------------------------------------------

def render_video_from_analyzed(analyzed: "AnalyzedContent") -> bytes:
    """
    Phase 2b-Video + Phase 3: AnalyzedContent → MP4 bytes.

    Currently raises NotImplementedError because actual video composition
    requires moviepy + ffmpeg.  The narration script generation (Phase 2b)
    is fully functional via ``generate_narration_script()``.
    """
    logger.info("Video export requested for %s — stub renderer", analyzed.repo_name)

    # TODO: Implement actual video rendering once moviepy/ffmpeg are available.
    # The flow would be:
    #   1. scenes = await generate_narration_script(analyzed)
    #   2. For each scene, render a text-card image (Pillow / cairo)
    #   3. Optionally run TTS on scene["narration"]
    #   4. Compose images + audio into an MP4 with moviepy

    raise NotImplementedError(
        "Video export is not yet implemented. "
        "Install moviepy/ffmpeg and implement the render_video_from_analyzed function. "
        "The narration script generator (generate_narration_script) is ready to use."
    )


def render_video(summary_text: str, repo_name: str) -> bytes:
    """
    Backward-compatible entry point.

    Wraps the legacy summary_text in a minimal AnalyzedContent and
    delegates to the structured renderer.
    """
    from api.content_analyzer import AnalyzedContent

    analyzed = AnalyzedContent(repo_name=repo_name)
    return render_video_from_analyzed(analyzed)


"""
Content Analyzer — Phase 1 (extraction) + Phase 2a (structured understanding)

Produces a structured `AnalyzedContent` object (JSON-backed) from
repo embeddings via a single LLM call.  Each export format (PDF, PPT,
Video) then applies its own Phase 2b adapter on top of the structured data.

Public API
----------
- AnalyzedContent          – Pydantic model (structured semantic JSON)
- RepoAnalysisRequest      – Input model for repo-embedding-based exports
- analyze_repo_content()   – Repo embeddings → AnalyzedContent
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, computed_field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class ModuleInfo(BaseModel):
    """A single key module / component."""
    name: str = ""
    responsibility: str = ""


class TechStack(BaseModel):
    """Structured tech-stack breakdown."""
    languages: List[str] = Field(default_factory=list)
    frameworks: List[str] = Field(default_factory=list)
    key_libraries: List[str] = Field(default_factory=list)
    infrastructure: List[str] = Field(default_factory=list)


class ModuleProgression(BaseModel):
    """Lightweight module progression entry for video-oriented storytelling."""
    name: str = ""
    stage: str = ""  # core | expansion
    role: str = ""
    solves: str = ""
    position: str = ""


# ---------------------------------------------------------------------------
# Fallback: clean raw JSON text for display
# ---------------------------------------------------------------------------

# Map JSON keys to human-readable section headers
_JSON_KEY_TO_HEADER = {
    "project_overview": "Project Overview:",
    "architecture": "Architecture & Design:",
    "tech_stack": "Tech Stack & Dependencies:",
    "key_modules": "Key Modules & Components:",
    "data_flow": "Data Flow & Processing:",
    "api_points": "API & Integration Points:",
    "target_users": "Target Users & Use Cases:",
    "deployment_info": "Deployment & Infrastructure:",
    "component_hierarchy": "Component Hierarchy:",
    "data_schemas": "Data Schemas & Models:",
}

# Keys to skip (not displayed as sections)
_JSON_KEYS_SKIP = {"repo_type_hint", "repo_name", "project_name"}


def _clean_raw_json_for_display(raw: str, repo_name: str = "") -> str:
    """
    When JSON parsing completely fails, attempt to extract readable content
    from the raw LLM output that looks like JSON.
    Strips brackets, braces, quotes, and JSON key names to produce clean text.
    """
    # First, try one more time to parse with aggressive repair
    try:
        repaired = _repair_json_string(raw)
        # Strip markdown fences
        repaired = re.sub(r"```(?:json)?\s*", "", repaired)
        repaired = re.sub(r"```\s*$", "", repaired)
        data = json.loads(repaired)
        if isinstance(data, dict) and data:
            return _dict_to_readable_text(data, repo_name)
    except Exception:
        pass

    # Try extracting { ... } block and repair
    match = re.search(r"\{", raw)
    if match:
        depth = 0
        start = match.start()
        end = len(raw) - 1
        for i in range(start, len(raw)):
            if raw[i] == "{":
                depth += 1
            elif raw[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        candidate = raw[start:end + 1]
        try:
            data = json.loads(_repair_json_string(candidate))
            if isinstance(data, dict) and data:
                return _dict_to_readable_text(data, repo_name)
        except Exception:
            pass

    # Ultimate fallback: regex-strip JSON syntax characters
    text = raw
    # Remove markdown fences
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    # Remove JSON key-value patterns: "key": -> Section header
    for jk, header in _JSON_KEY_TO_HEADER.items():
        text = re.sub(rf'"\s*{jk}\s*"\s*:\s*', f"\n{header}\n", text, flags=re.IGNORECASE)
    # Remove skipped keys
    for jk in _JSON_KEYS_SKIP:
        text = re.sub(rf'"\s*{jk}\s*"\s*:\s*"[^"]*"\s*,?\s*', "", text, flags=re.IGNORECASE)
    # Remove remaining JSON structural syntax
    text = re.sub(r'^\s*\{\s*', '', text)
    text = re.sub(r'\s*\}\s*$', '', text)
    text = text.replace("[", "").replace("]", "")
    text = text.replace("{", "").replace("}", "")
    # Clean up stray quotes around values
    text = re.sub(r'(?<!\w)"([^"]{3,})"(?!\w)', r'\1', text)
    # Remove remaining "name": / "responsibility": inner keys
    text = re.sub(r'"\s*name\s*"\s*:\s*"([^"]*)"', r'\1', text, flags=re.IGNORECASE)
    text = re.sub(r'"\s*responsibility\s*"\s*:\s*"([^"]*)"', r': \1', text, flags=re.IGNORECASE)
    text = re.sub(r'"\s*languages?\s*"\s*:\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'"\s*frameworks?\s*"\s*:\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'"\s*key_libraries\s*"\s*:\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'"\s*infrastructure\s*"\s*:\s*', '', text, flags=re.IGNORECASE)
    # Clean trailing/leading commas from lines
    text = re.sub(r',\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*,\s*', '', text, flags=re.MULTILINE)
    # Collapse multiple blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Add project name header
    result = f"Project Name: {repo_name}\n\n" + text.strip()
    return result


def _dict_to_readable_text(data: dict, repo_name: str = "") -> str:
    """Convert a parsed JSON dict into clean, section-structured text."""
    lines: list[str] = []
    name = repo_name or data.get("repo_name", "") or data.get("project_name", "")
    lines.append(f"Project Name: {name}")
    lines.append("")

    # project_overview
    overview = data.get("project_overview", "")
    if overview and isinstance(overview, str):
        lines.append("Project Overview:")
        lines.append(overview)
        lines.append("")

    # architecture
    arch = data.get("architecture", [])
    if arch:
        lines.append("Architecture & Design:")
        if isinstance(arch, list):
            for item in arch:
                lines.append(f"- {item}" if isinstance(item, str) else f"- {item}")
        elif isinstance(arch, str):
            lines.append(arch)
        lines.append("")

    # tech_stack
    ts = data.get("tech_stack", {})
    if isinstance(ts, dict):
        items = []
        for key in ("languages", "frameworks", "key_libraries", "infrastructure"):
            val = ts.get(key, [])
            if isinstance(val, list):
                items.extend(val)
            elif isinstance(val, str) and val:
                items.append(val)
        if items:
            lines.append("Tech Stack & Dependencies:")
            for item in items:
                lines.append(f"- {item}")
            lines.append("")

    # key_modules
    mods = data.get("key_modules", [])
    if mods and isinstance(mods, list):
        lines.append("Key Modules & Components:")
        for m in mods:
            if isinstance(m, dict):
                n = m.get("name", "")
                r = m.get("responsibility", "")
                lines.append(f"- {n}: {r}" if n else f"- {r}")
            elif isinstance(m, str):
                lines.append(f"- {m}")
        lines.append("")

    # data_flow
    df = data.get("data_flow", [])
    if df:
        lines.append("Data Flow & Processing:")
        if isinstance(df, list):
            for step in df:
                lines.append(f"- {step}" if isinstance(step, str) else f"- {step}")
        elif isinstance(df, str):
            lines.append(df)
        lines.append("")

    # api_points
    apis = data.get("api_points", [])
    if apis:
        lines.append("API & Integration Points:")
        if isinstance(apis, list):
            for pt in apis:
                lines.append(f"- {pt}" if isinstance(pt, str) else f"- {pt}")
        elif isinstance(apis, str):
            lines.append(apis)
        lines.append("")

    # target_users
    tu = data.get("target_users", "")
    if tu and isinstance(tu, str):
        lines.append("Target Users & Use Cases:")
        lines.append(tu)
        lines.append("")

    # deployment_info
    dep = data.get("deployment_info", "")
    if dep and isinstance(dep, str):
        lines.append("Deployment & Infrastructure:")
        lines.append(dep)
        lines.append("")

    # component_hierarchy
    ch = data.get("component_hierarchy", "")
    if ch and isinstance(ch, str):
        lines.append("Component Hierarchy:")
        lines.append(ch)
        lines.append("")

    # data_schemas
    ds = data.get("data_schemas", "")
    if ds and isinstance(ds, str):
        lines.append("Data Schemas & Models:")
        lines.append(ds)
        lines.append("")

    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# AnalyzedContent — the single structured artefact from Phase 2a
# ---------------------------------------------------------------------------

class AnalyzedContent(BaseModel):
    """
    Structured semantic understanding of a repository.

    Produced by Phase 2a (one LLM call that outputs JSON).
    Consumed by Phase 2b adapters (PDF / PPT / Video) which each decide
    how to render this data in format-specific ways.
    """

    repo_name: str = ""
    repo_url: str = ""
    language: str = "en"

    # Repo type inferred by the LLM
    repo_type_hint: str = "generic"  # library | webapp | microservice | data_pipeline | cli_tool | generic

    # Core structured sections
    project_overview: str = ""
    architecture: List[str] = Field(default_factory=list)
    tech_stack: TechStack = Field(default_factory=TechStack)
    key_modules: List[ModuleInfo] = Field(default_factory=list)
    data_flow: List[str] = Field(default_factory=list)
    api_points: List[str] = Field(default_factory=list)
    target_users: str = ""

    # Video-oriented module storyline support
    module_progression: List[ModuleProgression] = Field(default_factory=list)

    # Optional deep-dive fields (populated based on repo_type_hint)
    deployment_info: Optional[str] = None
    component_hierarchy: Optional[str] = None
    data_schemas: Optional[str] = None

    # Fallback: raw LLM response text, used when JSON parsing fails
    raw_llm_text: str = Field(default="", exclude=True)

    # ------------------------------------------------------------------
    # Backward-compatible computed property: plain-text summary
    # ------------------------------------------------------------------
    @computed_field  # type: ignore[misc]
    @property
    def summary_text(self) -> str:
        """
        Auto-generate a legacy plain-text summary from the structured fields.
        This keeps existing ``render_pdf(analyzed.summary_text, …)`` calls
        working during the migration period.

        If all structured fields are empty (e.g. JSON parsing failed),
        falls back to the raw LLM response text so the PDF is never blank.
        """
        has_content = bool(
            self.project_overview
            or self.architecture
            or self.key_modules
            or self.data_flow
            or self.api_points
            or self.target_users
        )

        if not has_content and self.raw_llm_text:
            # Fallback: clean JSON artifacts from raw text, then display
            return _clean_raw_json_for_display(self.raw_llm_text, self.repo_name)

        lines: list[str] = []

        lines.append(f"Project Name: {self.repo_name}")
        lines.append("")

        if self.project_overview:
            lines.append("Project Overview:")
            lines.append(self.project_overview)
            lines.append("")

        if self.architecture:
            lines.append("Architecture & Design:")
            for item in self.architecture:
                lines.append(f"- {item}")
            lines.append("")

        if self.tech_stack and (
            self.tech_stack.languages
            or self.tech_stack.frameworks
            or self.tech_stack.key_libraries
            or self.tech_stack.infrastructure
        ):
            lines.append("Tech Stack & Dependencies:")
            for lang in self.tech_stack.languages:
                lines.append(f"- {lang}")
            for fw in self.tech_stack.frameworks:
                lines.append(f"- {fw}")
            for lib in self.tech_stack.key_libraries:
                lines.append(f"- {lib}")
            for infra in self.tech_stack.infrastructure:
                lines.append(f"- {infra}")
            lines.append("")

        if self.key_modules:
            lines.append("Key Modules & Components:")
            for mod in self.key_modules:
                lines.append(f"- {mod.name}: {mod.responsibility}")
            lines.append("")

        if self.data_flow:
            lines.append("Data Flow & Processing:")
            for step in self.data_flow:
                lines.append(f"- {step}")
            lines.append("")

        if self.api_points:
            lines.append("API & Integration Points:")
            for pt in self.api_points:
                lines.append(f"- {pt}")
            lines.append("")

        if self.target_users:
            lines.append("Target Users & Use Cases:")
            lines.append(self.target_users)
            lines.append("")

        if self.module_progression:
            lines.append("Module Progression:")
            for mod in self.module_progression:
                lines.append(f"- {mod.name} [{mod.stage}]: role={mod.role}; solves={mod.solves}; position={mod.position}")
            lines.append("")

        # Append optional deep-dive sections when present
        if self.deployment_info:
            lines.append("Deployment & Infrastructure:")
            lines.append(self.deployment_info)
            lines.append("")

        if self.component_hierarchy:
            lines.append("Component Hierarchy:")
            lines.append(self.component_hierarchy)
            lines.append("")

        if self.data_schemas:
            lines.append("Data Schemas & Models:")
            lines.append(self.data_schemas)
            lines.append("")

        return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class RepoAnalysisRequest(BaseModel):
    repo_url: str = ""
    repo_name: str = ""
    provider: str = "openai"
    model: Optional[str] = None
    language: str = "en"
    repo_type: str = "github"
    access_token: Optional[str] = None
    excluded_dirs: Optional[str] = None
    excluded_files: Optional[str] = None
    included_dirs: Optional[str] = None
    included_files: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_language_name(code: str) -> str:
    """Map ISO language code to human-readable name."""
    _MAP = {"en": "English", "zh": "Chinese", "ja": "Japanese", "ko": "Korean"}
    return _MAP.get(code, "English")


def _repair_json_string(s: str) -> str:
    """Fix common LLM JSON errors before parsing."""
    # 1) Remove trailing commas before } or ]
    s = re.sub(r',\s*([}\]])', r'\1', s)
    # 2) Fix missing commas between array string elements:
    #    "..."\n"..." -> "...",\n"..."
    s = re.sub(r'"\s*\n\s*"', '",\n"', s)
    # 3) Fix missing commas between } and { in arrays
    s = re.sub(r'}\s*\n\s*\{', '},\n{', s)
    # 4) Fix missing commas between } and "
    s = re.sub(r'}\s*\n\s*"', '},\n"', s)
    # 5) Fix missing commas between "..." and { in arrays
    s = re.sub(r'"\s*\n\s*\{', '",\n{', s)
    return s


def _extract_json_from_llm(raw: str) -> dict:
    """
    Best-effort extraction of a JSON object from LLM output.
    Handles markdown fences, leading commentary, trailing text,
    and common LLM JSON syntax errors (missing commas, etc.).
    """
    # Strip markdown fences
    cleaned = re.sub(r"```(?:json)?\s*", "", raw)
    cleaned = re.sub(r"```\s*$", "", cleaned)

    # Try parsing the entire cleaned text first
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try with JSON repair
    try:
        repaired = _repair_json_string(cleaned)
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    # Try to find the first { ... } block
    match = re.search(r"\{", cleaned)
    if match:
        depth = 0
        start = match.start()
        for i in range(start, len(cleaned)):
            if cleaned[i] == "{":
                depth += 1
            elif cleaned[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = cleaned[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        pass
                    # Try repair on the extracted block
                    try:
                        return json.loads(_repair_json_string(candidate))
                    except json.JSONDecodeError:
                        pass
                    break

    logger.warning(
        "Failed to parse JSON from LLM output (length=%d); first 300 chars: %s",
        len(raw), raw[:300],
    )
    return {}


def _build_analyzed_content(
    raw_json: dict,
    repo_name: str,
    repo_url: str,
    language: str,
    raw_llm_text: str = "",
) -> AnalyzedContent:
    """Construct an AnalyzedContent from the raw JSON dict returned by LLM."""

    # Parse key_modules
    key_modules: list[ModuleInfo] = []
    for m in raw_json.get("key_modules", []):
        if isinstance(m, dict):
            key_modules.append(ModuleInfo(**m))
        elif isinstance(m, str):
            key_modules.append(ModuleInfo(name=m, responsibility=""))

    # Parse module_progression
    module_progression: list[ModuleProgression] = []
    for m in raw_json.get("module_progression", []):
        if isinstance(m, dict):
            module_progression.append(ModuleProgression(**m))

    # Parse tech_stack
    ts_raw = raw_json.get("tech_stack", {})
    if isinstance(ts_raw, dict):
        tech_stack = TechStack(
            languages=ts_raw.get("languages", []),
            frameworks=ts_raw.get("frameworks", []),
            key_libraries=ts_raw.get("key_libraries", []),
            infrastructure=ts_raw.get("infrastructure", []),
        )
    else:
        tech_stack = TechStack()

    analyzed = AnalyzedContent(
        repo_name=repo_name or raw_json.get("repo_name", ""),
        repo_url=repo_url,
        language=language,
        repo_type_hint=raw_json.get("repo_type_hint", "generic"),
        project_overview=raw_json.get("project_overview", ""),
        architecture=raw_json.get("architecture", []),
        tech_stack=tech_stack,
        key_modules=key_modules,
        data_flow=raw_json.get("data_flow", []),
        module_progression=module_progression,
        api_points=raw_json.get("api_points", []),
        target_users=raw_json.get("target_users", ""),
        deployment_info=raw_json.get("deployment_info"),
        component_hierarchy=raw_json.get("component_hierarchy"),
        data_schemas=raw_json.get("data_schemas"),
        raw_llm_text=raw_llm_text,
    )

    if not raw_json:
        logger.warning("_build_analyzed_content received empty JSON; raw_llm_text fallback active (len=%d)", len(raw_llm_text))

    return analyzed


# ---------------------------------------------------------------------------
# Phase 1: Content extraction (pure Python, no LLM)
# ---------------------------------------------------------------------------

def _extract_repo_context(rag_instance: Any, repo_name: str) -> str:
    """
    Retrieve broadly representative content from the FAISS index.
    Uses a few generic architectural queries to pull diverse chunks.
    """
    queries = [
        f"What is {repo_name}? Project overview, purpose, main features.",
        f"Architecture design patterns, main modules, file structure of {repo_name}.",
        f"Tech stack, frameworks, libraries, dependencies used in {repo_name}.",
        f"Data flow, API endpoints, external integrations in {repo_name}.",
    ]

    seen_indices: set[int] = set()
    parts: list[str] = []

    for q in queries:
        try:
            retrieved = rag_instance.call(q)
            if retrieved and len(retrieved) > 0:
                docs = retrieved[0].documents if hasattr(retrieved[0], "documents") else []
                indices = retrieved[0].doc_indices if hasattr(retrieved[0], "doc_indices") else []
                for idx, doc in zip(indices, docs):
                    if idx not in seen_indices:
                        seen_indices.add(idx)
                        file_path = ""
                        if hasattr(doc, "meta_data") and isinstance(doc.meta_data, dict):
                            file_path = doc.meta_data.get("file_path", "")
                        parts.append(f"--- {file_path} ---")
                        parts.append(doc.text if hasattr(doc, "text") else str(doc))
                        parts.append("")
        except Exception as e:
            logger.warning("Retrieval query failed (%s): %s", q[:40], e)

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Phase 2a: LLM structured analysis (single call → JSON)
# ---------------------------------------------------------------------------

async def _run_llm_structured_analysis(
    input_context: str,
    repo_name: str,
    language: str,
    provider: str,
    model: Optional[str],
) -> str:
    """
    Call the LLM once with STRUCTURED_ANALYSIS_PROMPT and return raw text.
    Supports all configured providers (ollama, openai, google, openrouter,
    bedrock, azure, dashscope).
    """
    from api.config import get_model_config
    from api.prompts import STRUCTURED_ANALYSIS_PROMPT

    language_name = _get_language_name(language)

    prompt = STRUCTURED_ANALYSIS_PROMPT.format(
        language_name=language_name,
        repo_name=repo_name,
        input_json=input_context,
    )

    config = get_model_config(provider, model)
    model_kwargs = config["model_kwargs"]

    # -- Provider dispatch (non-streaming, single response) ----------------
    from adalflow.core.types import ModelType

    if provider == "ollama":
        from adalflow.components.model_client.ollama_client import OllamaClient

        # /no_think must be at the very beginning for qwen3
        ollama_prompt = "/no_think\n" + prompt

        client = OllamaClient()
        kwargs = {
            "model": model_kwargs["model"],
            "stream": False,
            "options": {
                "temperature": 0.3,
                "top_p": 0.7,
                "num_ctx": min(model_kwargs.get("num_ctx", 8000), 8000),
            },
        }
        api_kwargs = client.convert_inputs_to_api_kwargs(
            input=ollama_prompt, model_kwargs=kwargs, model_type=ModelType.LLM,
        )
        response = await client.acall(api_kwargs=api_kwargs, model_type=ModelType.LLM)
        return _extract_text_from_response(response, provider)

    if provider == "openrouter":
        from api.openrouter_client import OpenRouterClient

        client = OpenRouterClient()
        kwargs = {"model": model_kwargs["model"], "stream": False, "temperature": model_kwargs.get("temperature", 0.3)}
        api_kwargs = client.convert_inputs_to_api_kwargs(
            input=prompt, model_kwargs=kwargs, model_type=ModelType.LLM,
        )
        response = await client.acall(api_kwargs=api_kwargs, model_type=ModelType.LLM)
        return _extract_text_from_response(response, provider)

    if provider == "openai":
        from api.openai_client import OpenAIClient

        client = OpenAIClient()
        kwargs = {"model": model_kwargs["model"], "stream": False, "temperature": model_kwargs.get("temperature", 0.3)}
        api_kwargs = client.convert_inputs_to_api_kwargs(
            input=prompt, model_kwargs=kwargs, model_type=ModelType.LLM,
        )
        response = await client.acall(api_kwargs=api_kwargs, model_type=ModelType.LLM)
        return _extract_text_from_response(response, provider)

    if provider == "bedrock":
        from api.bedrock_client import BedrockClient

        client = BedrockClient()
        kwargs = {
            "model": model_kwargs["model"],
            "temperature": model_kwargs.get("temperature", 0.3),
            "top_p": model_kwargs.get("top_p", 0.9),
        }
        api_kwargs = client.convert_inputs_to_api_kwargs(
            input=prompt, model_kwargs=kwargs, model_type=ModelType.LLM,
        )
        response = await client.acall(api_kwargs=api_kwargs, model_type=ModelType.LLM)
        return _extract_text_from_response(response, provider)

    if provider == "azure":
        from api.azureai_client import AzureAIClient

        client = AzureAIClient()
        kwargs = {
            "model": model_kwargs["model"],
            "stream": False,
            "temperature": model_kwargs.get("temperature", 0.3),
            "top_p": model_kwargs.get("top_p", 0.9),
        }
        api_kwargs = client.convert_inputs_to_api_kwargs(
            input=prompt, model_kwargs=kwargs, model_type=ModelType.LLM,
        )
        response = await client.acall(api_kwargs=api_kwargs, model_type=ModelType.LLM)
        return _extract_text_from_response(response, provider)

    if provider == "dashscope":
        from api.dashscope_client import DashscopeClient

        client = DashscopeClient()
        kwargs = {
            "model": model_kwargs["model"],
            "stream": False,
            "temperature": model_kwargs.get("temperature", 0.3),
            "top_p": model_kwargs.get("top_p", 0.9),
        }
        api_kwargs = client.convert_inputs_to_api_kwargs(
            input=prompt, model_kwargs=kwargs, model_type=ModelType.LLM,
        )
        response = await client.acall(api_kwargs=api_kwargs, model_type=ModelType.LLM)
        return _extract_text_from_response(response, provider)

    # Default: Google Generative AI
    import google.generativeai as genai

    gen_model = genai.GenerativeModel(
        model_name=model_kwargs.get("model", "gemini-2.0-flash"),
        generation_config={
            "temperature": model_kwargs.get("temperature", 0.3),
            "top_p": model_kwargs.get("top_p", 0.9),
            "top_k": model_kwargs.get("top_k", 40),
        },
    )
    response = await gen_model.generate_content_async(prompt)
    return response.text if hasattr(response, "text") else str(response)


def _extract_text_from_response(response: Any, provider: str) -> str:
    """Pull plain text from various provider response shapes."""
    # Ollama generate API: response.response
    if hasattr(response, "response") and isinstance(getattr(response, "response", None), str) and getattr(response, "response", ""):
        return response.response

    # Ollama chat API: response.message.content
    if hasattr(response, "message"):
        msg = response.message
        if hasattr(msg, "content") and isinstance(msg.content, str):
            return msg.content

    if hasattr(response, "text"):
        return response.text

    # OpenAI-compatible: choices[0].message.content
    if hasattr(response, "choices") and response.choices:
        choice = response.choices[0]
        if hasattr(choice, "message") and hasattr(choice.message, "content"):
            return choice.message.content

    # Dict-like responses
    if isinstance(response, dict):
        if "choices" in response and response["choices"]:
            return response["choices"][0].get("message", {}).get("content", "")
        if "response" in response:
            return response["response"]

    # Fallback: collect async generator
    if hasattr(response, "__aiter__"):
        import asyncio
        chunks: list[str] = []

        async def _collect():
            async for chunk in response:
                text = getattr(chunk, "response", None) or getattr(chunk, "text", None) or str(chunk)
                chunks.append(text)

        try:
            asyncio.get_event_loop().run_until_complete(_collect())
        except RuntimeError:
            pass
        return "".join(chunks)

    return str(response)


def _postprocess_summary(text: str) -> str:
    """
    Clean up / repair common LLM output issues:
    - Remove markdown code fences
    - Strip thinking tags (<think>...</think>)
    - Normalize section headers to canonical English form
    - Remove stray leading/trailing whitespace
    """
    # Remove <think>...</think> blocks (qwen3 sometimes leaks them)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # Remove markdown fences
    text = re.sub(r"```(?:json|markdown)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text, flags=re.MULTILINE)

    # Normalize Chinese section headers to standard English format
    cn_to_en = {
        "项目名称": "Project Name",
        "项目概述": "Project Overview",
        "功能概述": "Project Overview",
        "架构与设计": "Architecture & Design",
        "架构设计": "Architecture & Design",
        "系统架构": "Architecture & Design",
        "技术栈与依赖": "Tech Stack & Dependencies",
        "技术栈": "Tech Stack & Dependencies",
        "关键模块与组件": "Key Modules & Components",
        "核心模块与组件": "Key Modules & Components",
        "关键模块": "Key Modules & Components",
        "核心模块": "Key Modules & Components",
        "数据流与处理": "Data Flow & Processing",
        "数据流": "Data Flow & Processing",
        "API与集成点": "API & Integration Points",
        "API接口": "API & Integration Points",
        "目标用户与使用场景": "Target Users & Use Cases",
        "目标用户": "Target Users & Use Cases",
        "适用人群": "Target Users & Use Cases",
    }

    lines = text.split("\n")
    result = []
    for line in lines:
        stripped = line.strip()
        # Try to match Chinese header pattern: "中文标题：" or "中文标题:"
        for cn, en in cn_to_en.items():
            if stripped.startswith(cn) and (":" in stripped or "：" in stripped):
                # Replace with English header format
                line = f"{en}:"
                break
        result.append(line)

    return "\n".join(result).strip()


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

async def analyze_repo_content(request: RepoAnalysisRequest) -> AnalyzedContent:
    """
    Full Phase 1 + Phase 2a pipeline for direct-repo-embedding exports.

    Phase 1: FAISS retrieval of representative code chunks (pure Python).
    Phase 2a: Single LLM call -> structured JSON -> AnalyzedContent.
    """
    overall_start = time.perf_counter()
    logger.info("analyze_repo_content: repo=%s", request.repo_name)

    from api.rag import RAG

    rag = RAG(provider=request.provider, model=request.model)

    # Parse filter arguments
    excluded_dirs = [d.strip() for d in request.excluded_dirs.split(",") if d.strip()] if request.excluded_dirs else None
    excluded_files = [f.strip() for f in request.excluded_files.split(",") if f.strip()] if request.excluded_files else None
    included_dirs = [d.strip() for d in request.included_dirs.split(",") if d.strip()] if request.included_dirs else None
    included_files = [f.strip() for f in request.included_files.split(",") if f.strip()] if request.included_files else None

    prepare_start = time.perf_counter()
    rag.prepare_retriever(
        request.repo_url,
        type=request.repo_type,
        access_token=request.access_token,
        excluded_dirs=excluded_dirs,
        excluded_files=excluded_files,
        included_dirs=included_dirs,
        included_files=included_files,
    )
    logger.info("Timing - prepare_retriever completed in %.2fs", time.perf_counter() - prepare_start)

    # Phase 1
    context_start = time.perf_counter()
    context_text = _extract_repo_context(rag, request.repo_name)
    logger.info("Timing - extract_repo_context completed in %.2fs (chars=%d)", time.perf_counter() - context_start, len(context_text))
    if not context_text.strip():
        logger.warning("No repo content retrieved; returning empty AnalyzedContent")
        return AnalyzedContent(repo_name=request.repo_name, repo_url=request.repo_url, language=request.language)

    # Phase 2a
    llm_start = time.perf_counter()
    raw_text = await _run_llm_structured_analysis(
        input_context=context_text,
        repo_name=request.repo_name,
        language=request.language,
        provider=request.provider,
        model=request.model,
    )
    logger.info("Timing - structured analysis LLM completed in %.2fs", time.perf_counter() - llm_start)

    build_start = time.perf_counter()
    raw_json = _extract_json_from_llm(raw_text)
    analyzed = _build_analyzed_content(raw_json, request.repo_name, request.repo_url, request.language, raw_llm_text=raw_text)
    logger.info("Timing - build_analyzed_content completed in %.2fs", time.perf_counter() - build_start)

    # Post-process raw_llm_text for fallback rendering
    if analyzed.raw_llm_text:
        analyzed.raw_llm_text = _postprocess_summary(analyzed.raw_llm_text)

    logger.info(
        "analyze_repo_content complete: repo_type_hint=%s, modules=%d, has_overview=%s, total=%.2fs",
        analyzed.repo_type_hint,
        len(analyzed.key_modules),
        bool(analyzed.project_overview),
        time.perf_counter() - overall_start,
    )
    return analyzed

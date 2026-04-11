"""
Template: Gamma PPTX export tool customization.

INSTRUCTIONS:
  Use this template when modifying or extending the Gamma PPTX export feature.
  Copy the relevant section and adapt to your use case.

This file is a reference — do NOT import it directly.
"""

# ============================================================================
# SECTION A: Export Tool Registration
# Location: api/agent/tools/export_tools.py -> build_export_tool_registry()
# ============================================================================

"""
registry.register(
    AgentTool(
        name="GENERATE_GAMMA_PPT",
        action_tag="[ACTION:GENERATE_GAMMA_PPT]",
        description="Generate a professionally designed PPTX presentation via Gamma.app.",
        keywords=(
            # English
            "gamma ppt",
            "gamma slides",
            "gamma presentation",
            "gamma deck",
            "gamma",
            "ai ppt",
            # Chinese
            "精美ppt",
            "gamma演示",
            "设计感ppt",
            "gamma幻灯片",
        ),
    )
)
"""


# ============================================================================
# SECTION B: Scheduler Preamble Messages
# Location: api/agent/scheduler.py -> _tool_preamble() both mappings
# ============================================================================

"""
# In the Chinese mapping dict:
"GENERATE_GAMMA_PPT": "我将通过 Gamma 为该仓库生成一份精美 PPT 演示文稿。",

# In the English mapping dict:
"GENERATE_GAMMA_PPT": "I will generate a professionally designed PPTX presentation via Gamma.",
"""


# ============================================================================
# SECTION C: Stage 2 Inference Signals
# Location: api/agent/scheduler.py -> _infer_from_recommendation()
# ============================================================================

"""
gamma_ppt_signals = ("gamma ppt", "gamma slides", "gamma", "精美ppt", "gamma演示", "设计感ppt", "ai ppt", "gamma幻灯片")
# ... add to scores dict:
scores["[ACTION:GENERATE_GAMMA_PPT]"] = gamma_ppt_score
"""


# ============================================================================
# SECTION D: Gamma API Configuration
# Location: api/gamma_ppt_export.py (top-level constants)
# ============================================================================

"""
# Environment variables:
GAMMA_API_KEY = os.environ.get("GAMMA_API_KEY", "")
GAMMA_POLL_INTERVAL = int(os.environ.get("GAMMA_POLL_INTERVAL", "5"))
GAMMA_TIMEOUT = int(os.environ.get("GAMMA_TIMEOUT", "300"))
GAMMA_NUM_CARDS = int(os.environ.get("GAMMA_NUM_CARDS", "10"))

# To customize the generation request:
payload = {
    "inputText": outline_text,
    "textMode": "condense",       # or "generate", "preserve"
    "format": "presentation",
    "numCards": 10,
    "exportAs": "pptx",           # or "pdf", "png"
    "textOptions": {
        "tone": "professional",   # or "casual", "formal"
        "audience": "developers and technical stakeholders",
        "amount": "detailed",     # or "brief", "medium", "extensive"
        "language": "en",         # ISO code
    },
    "imageOptions": {
        "source": "webFreeToUseCommercially",  # or "aiGenerated", "noImages"
    },
}
"""


# ============================================================================
# SECTION E: Gamma Outline Prompt
# Location: api/prompts.py -> GAMMA_PPT_OUTLINE_PROMPT
# ============================================================================

"""
GAMMA_PPT_OUTLINE_PROMPT = '''You are a professional presentation architect. ...

Write a structured outline that a presentation tool will expand into polished slides.
Use `---` (three dashes on its own line) to separate slides.
...

Project analysis:
{analysis_json}
'''
"""

"""
Icon mapping for onboard 5-act video.

Provides emoji + label heuristics so the renderer can show
"📥 GitHub URL" instead of just "GitHub URL". Keep these latin-1-safe
where possible (emoji works in Playwright HTML; the legacy Pillow
fallback path doesn't apply here — we always render via Playwright).

Functions:
    - guess_input_icon(label) -> emoji
    - guess_output_icon(label) -> emoji
    - guess_process_icon(label) -> emoji
    - tech_icon(name) -> emoji  (Python, JS, Docker, ...)
    - rank_medal(rank) -> emoji  (🥇🥈🥉) for contributor showcase
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Generic role icons (fallback if no specific match)
# ---------------------------------------------------------------------------

_DEFAULT_INPUT = "📥"
_DEFAULT_PROCESS = "⚙️"
_DEFAULT_OUTPUT = "📤"

# ---------------------------------------------------------------------------
# Keyword → icon for the 3-box mental model (Act 3)
# Order matters: first match wins.
# ---------------------------------------------------------------------------

_INPUT_KEYWORDS = [
    (re.compile(r"\b(url|link|github|repo|repository)\b", re.I), "🔗"),
    (re.compile(r"\b(file|upload|document|pdf)\b", re.I), "📄"),
    (re.compile(r"\b(image|photo|picture)\b", re.I), "🖼️"),
    (re.compile(r"\b(audio|voice|sound|music)\b", re.I), "🎤"),
    (re.compile(r"\b(video|stream|recording)\b", re.I), "🎬"),
    (re.compile(r"\b(query|question|prompt|ask|chat)\b", re.I), "💬"),
    (re.compile(r"\b(text|input|string|content)\b", re.I), "✍️"),
    (re.compile(r"\b(user|person|developer|customer)\b", re.I), "👤"),
    (re.compile(r"\b(data|dataset|csv|json)\b", re.I), "📊"),
    (re.compile(r"\b(code|script|source)\b", re.I), "📝"),
]

_PROCESS_KEYWORDS = [
    (re.compile(r"\b(ai|llm|gpt|model|neural|ml)\b", re.I), "🧠"),
    (re.compile(r"\b(analy[sz]e|parse|extract|read)\b", re.I), "🔍"),
    (re.compile(r"\b(transform|convert|translat|render)\b", re.I), "🔄"),
    (re.compile(r"\b(retriev|search|find|index)\b", re.I), "🔎"),
    (re.compile(r"\b(embed|vector|similarit)\b", re.I), "🧮"),
    (re.compile(r"\b(generat|writ|compose|creat)\b", re.I), "✨"),
    (re.compile(r"\b(train|learn|fit|optimi[sz]e)\b", re.I), "🏋️"),
    (re.compile(r"\b(pipeline|workflow|orchestr|schedule)\b", re.I), "🔀"),
]

_OUTPUT_KEYWORDS = [
    (re.compile(r"\b(pdf|report|document|guide)\b", re.I), "📄"),
    (re.compile(r"\b(slide|ppt|presentation|deck)\b", re.I), "📊"),
    (re.compile(r"\b(video|movie|mp4|walkthrough)\b", re.I), "🎬"),
    (re.compile(r"\b(image|poster|infographic|diagram)\b", re.I), "🖼️"),
    (re.compile(r"\b(audio|voice|narration|tts)\b", re.I), "🎤"),
    (re.compile(r"\b(answer|reply|response|explanation)\b", re.I), "💡"),
    (re.compile(r"\b(summary|overview|abstract)\b", re.I), "📋"),
    (re.compile(r"\b(visuali[sz]|chart|graph|plot)\b", re.I), "📈"),
    (re.compile(r"\b(insight|recommendation|finding)\b", re.I), "🔮"),
]

# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def _match(rules: list, label: str, default: str) -> str:
    text = (label or "").strip()
    if not text:
        return default
    for pattern, icon in rules:
        if pattern.search(text):
            return icon
    return default


def guess_input_icon(label: str) -> str:
    return _match(_INPUT_KEYWORDS, label, _DEFAULT_INPUT)


def guess_process_icon(label: str) -> str:
    return _match(_PROCESS_KEYWORDS, label, _DEFAULT_PROCESS)


def guess_output_icon(label: str) -> str:
    return _match(_OUTPUT_KEYWORDS, label, _DEFAULT_OUTPUT)


# ---------------------------------------------------------------------------
# Tech stack icons (used by Act 1 badge corner)
# ---------------------------------------------------------------------------

_TECH_ICONS = {
    "python": "🐍",
    "javascript": "🟨",
    "typescript": "🔷",
    "go": "🐹",
    "rust": "🦀",
    "java": "☕",
    "ruby": "💎",
    "c++": "➕",
    "c": "🇨",
    "swift": "🦅",
    "kotlin": "🤖",
    "react": "⚛️",
    "next.js": "▲",
    "vue": "💚",
    "angular": "🔺",
    "fastapi": "⚡",
    "flask": "🍶",
    "django": "🟢",
    "node.js": "🟢",
    "docker": "🐳",
    "kubernetes": "☸️",
    "postgres": "🐘",
    "mongodb": "🍃",
    "redis": "🔴",
    "tensorflow": "🟠",
    "pytorch": "🔥",
}


def tech_icon(name: str) -> str:
    """Return a small icon for a tech-stack label, or empty string if unknown."""
    if not name:
        return ""
    return _TECH_ICONS.get(name.strip().lower(), "")


# ---------------------------------------------------------------------------
# Contributor ranking medals (fallback when avatar URL is unavailable)
# ---------------------------------------------------------------------------

_MEDALS = ["🥇", "🥈", "🥉"]


def rank_medal(rank: int) -> str:
    """Return 🥇🥈🥉 for ranks 1-3, otherwise a generic dot."""
    if 1 <= rank <= 3:
        return _MEDALS[rank - 1]
    return "▫️"


# ---------------------------------------------------------------------------
# Setup-step icons (Act 5)
# ---------------------------------------------------------------------------

_SETUP_KEYWORDS = [
    (re.compile(r"\b(clone|git)\b", re.I), "📥"),
    (re.compile(r"\b(install|pip|npm|yarn|cargo|brew|apt)\b", re.I), "📦"),
    (re.compile(r"\b(env|config|\.env|setting|secret|key)\b", re.I), "🔑"),
    (re.compile(r"\b(open|browser|navigate|visit|http)\b", re.I), "🌐"),
    (re.compile(r"\b(test|pytest|jest)\b", re.I), "🧪"),
    (re.compile(r"\b(build|compile|bundle)\b", re.I), "🔨"),
    # Run-style keywords. Order matters — keep BELOW "open"/"test" so
    # "open http://localhost" doesn't get matched as "run".
    (re.compile(r"\b(run|start|launch|serve|exec|python|node|go|deno|cargo|java)\b", re.I), "▶️"),
]


def guess_setup_step_icon(step_text: str) -> str:
    """Return an icon for one Act 5 setup step (clone / install / run / open / ...)."""
    return _match(_SETUP_KEYWORDS, step_text, "🔸")


# ---------------------------------------------------------------------------
# Prerequisite chip icons (Act 5, before the numbered checklist)
# ---------------------------------------------------------------------------

_PREREQ_KEYWORDS = [
    (re.compile(r"\bpython\b", re.I), "🐍"),
    (re.compile(r"\b(node|node\.?js|npm|yarn)\b", re.I), "🟢"),
    (re.compile(r"\b(docker|container)\b", re.I), "🐳"),
    (re.compile(r"\b(go(?:lang)?)\b", re.I), "🐹"),
    (re.compile(r"\brust\b", re.I), "🦀"),
    (re.compile(r"\b(java|jdk|jre|maven|gradle)\b", re.I), "☕"),
    (re.compile(r"\b(ruby|rails|gem)\b", re.I), "💎"),
    (re.compile(r"\b(api[\s-]?key|secret|token|credential)\b", re.I), "🔑"),
    (re.compile(r"\b(account|signup|register)\b", re.I), "👤"),
    (re.compile(r"\b(browser|chrome|firefox)\b", re.I), "🌐"),
    (re.compile(r"\bgit\b", re.I), "📦"),
    (re.compile(r"\b(database|postgres|mysql|mongo|redis|sqlite)\b", re.I), "🗄️"),
    (re.compile(r"\b(gpu|cuda|nvidia)\b", re.I), "🎮"),
]


def guess_prereq_icon(label: str) -> str:
    """Return an icon for one Act 5 prerequisite chip (Python / Node / Docker / ...)."""
    return _match(_PREREQ_KEYWORDS, label, "📋")

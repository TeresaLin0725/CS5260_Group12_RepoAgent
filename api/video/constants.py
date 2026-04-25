"""Video pipeline constants — single source of truth for all magic numbers."""

VIDEO_SIZE = (1280, 720)
SCENE_DURATION_DEFAULT = 7
SCENE_DURATION_MIN = 4
# Upper bound on a single act's visual duration. Must comfortably fit the
# longest TTS narration we ever produce — otherwise the visual cuts to the
# next act while the previous act's audio is still speaking, causing two
# voices to overlap on the timeline (concatenate_videoclips schedules each
# clip's audio at its start, so trailing audio bleeds into the next clip).
# 30s fits a 5-sentence Act 1 intro at OpenAI TTS speed (~130 wpm).
SCENE_DURATION_MAX = 30
AUDIO_PADDING_SECONDS = 1.0
MAX_SCENES = 10
MAX_BULLETS = 3
MAX_BULLET_CHARS = 48
MAX_EXPANSION_SCENES = max(2, MAX_SCENES - 3)
MAX_KEYWORDS = 4
TRANSITION_SECONDS = 0.35
VIDEO_FPS = 24
NARRATION_MAX_CHARS = 280
MAX_NODE_DESC_CHARS = 30
MAX_SUBTITLE_CHARS = 160

"""Video export job progress tracking (in-memory, keyed by job_id)."""

from typing import Optional

_job_progress: dict[str, dict] = {}

PROGRESS_STEPS = [
    "Analyzing repository structure...",
    "Generating narration script...",
    "Generating TTS audio...",
    "Rendering scene images...",
    "Composing final MP4...",
]


def update_progress(job_id: Optional[str], step: int, message: Optional[str] = None) -> None:
    """Update progress for a job. step is 1-based index into PROGRESS_STEPS."""
    if not job_id:
        return
    total = len(PROGRESS_STEPS)
    _job_progress[job_id] = {
        "step": step,
        "total": total,
        "message": message or (PROGRESS_STEPS[step - 1] if 1 <= step <= total else "Working..."),
        "done": step > total,
    }


def get_progress(job_id: str) -> Optional[dict]:
    return _job_progress.get(job_id)


def clear_progress(job_id: str) -> None:
    _job_progress.pop(job_id, None)

"""
End-to-end test for the video pipeline with TTS + Playwright rendering.

Bypasses LLM and FAISS — uses a mock AnalyzedContent with realistic data
to test: fallback narration -> TTS generation -> multi-frame Playwright rendering
        -> subtitle/highlight sync -> MoviePy composition.

Usage:
    cd /home/seakon-ai/nus/CS5260_Group12_RepoAgent
    source .venv/bin/activate
    python tests/test_video_e2e.py
"""

import asyncio
import os
import shutil
import sys
import time

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.content_analyzer import (
    AnalyzedContent,
    ModuleInfo,
    ModuleProgression,
    TechStack,
)


def build_mock_analyzed() -> AnalyzedContent:
    """Build a realistic AnalyzedContent for testing."""
    return AnalyzedContent(
        repo_name="RepoHelper",
        repo_url="https://github.com/example/repohelper",
        language="en",
        repo_type_hint="webapp",
        project_overview=(
            "RepoHelper is an AI-powered repository documentation generator. "
            "It analyzes GitHub, GitLab, and Bitbucket repositories and produces "
            "structured technical documentation, Mermaid diagrams, interactive AI chat, "
            "and multi-format exports including PDF, PowerPoint, and video walkthroughs."
        ),
        architecture=[
            "The system uses a two-tier architecture with a Next.js frontend and a FastAPI Python backend.",
            "The frontend communicates with the backend via REST APIs and WebSocket for real-time chat.",
            "The backend orchestrates LLM calls, vector search, and format-specific export renderers.",
        ],
        tech_stack=TechStack(
            languages=["Python 3.12", "TypeScript"],
            frameworks=["FastAPI", "Next.js 15"],
            key_libraries=["FAISS", "Pillow", "MoviePy", "edge-tts"],
            infrastructure=["Docker", "FAISS vector store"],
        ),
        key_modules=[
            ModuleInfo(name="data_pipeline", responsibility="Clones repositories, indexes code files, and creates vector embeddings for search."),
            ModuleInfo(name="rag", responsibility="Retrieval Augmented Generation module that queries FAISS and feeds context to LLMs."),
            ModuleInfo(name="content_analyzer", responsibility="Orchestrates Phase 1 extraction and Phase 2a structured LLM analysis."),
            ModuleInfo(name="export_service", responsibility="Unified orchestration layer routing analyzed content to PDF, PPT, or Video renderers."),
            ModuleInfo(name="video_export", responsibility="Converts structured analysis into narration scripts and renders MP4 video walkthroughs."),
            ModuleInfo(name="simple_chat", responsibility="HTTP streaming chat endpoint supporting multiple LLM providers."),
        ],
        data_flow=[
            "A user submits a repository URL through the frontend chat interface.",
            "The backend clones the repo, chunks the code, and builds a FAISS vector index.",
            "When an export is requested, representative code chunks are retrieved and sent to an LLM for structured analysis.",
            "The analysis result is routed to the appropriate renderer which produces the final output.",
        ],
        api_points=[
            "REST endpoints for chat streaming, export triggers, and model configuration.",
            "WebSocket channel for real-time interactive documentation viewer.",
        ],
        target_users=(
            "Developers who want to quickly understand unfamiliar codebases. "
            "A new team member joins a project, submits the repo URL, and within minutes "
            "receives a video walkthrough explaining how the system is built from the ground up."
        ),
        module_progression=[
            ModuleProgression(
                name="data_pipeline",
                stage="core",
                role="Clones repos and builds the searchable code index.",
                solves="Without this, the system has no access to any repository content at all.",
                position="Entry point of the backend, feeds into RAG.",
            ),
            ModuleProgression(
                name="rag",
                stage="core",
                role="Retrieves relevant code chunks and feeds them to LLMs with context.",
                solves="Without this, LLM calls would have no repository context and produce generic answers.",
                position="Sits between the data pipeline and all LLM-based features.",
            ),
            ModuleProgression(
                name="content_analyzer",
                stage="core",
                role="Produces a structured AnalyzedContent object from raw code via a single LLM call.",
                solves="Without this, each export format would need its own ad-hoc LLM prompting, leading to inconsistency.",
                position="Consumes RAG output and feeds all export renderers.",
            ),
            ModuleProgression(
                name="export_service",
                stage="expansion",
                role="Routes analyzed content to format-specific renderers like PDF, PPT, and Video.",
                solves="Without this, adding a new export format would require modifying multiple integration points.",
                position="Orchestration layer between content_analyzer and individual renderers.",
            ),
            ModuleProgression(
                name="video_export",
                stage="expansion",
                role="Generates narration scripts and renders MP4 video walkthroughs with TTS audio.",
                solves="Without this, users can only read static documents. Video makes complex architectures accessible to non-technical stakeholders.",
                position="Called by export_service, uses Pillow for rendering and MoviePy for composition.",
            ),
        ],
        deployment_info="Containerized with Docker, supports local development with hot-reload.",
    )


async def run_test():
    """Run the full video pipeline with mock data."""
    from api.video_export import (
        _build_storyline_scenes,
        _normalize_scenes,
        _fallback_narration_script,
        _scene_to_card_content,
        _build_scene_clip,
        _compose_final_video,
        _read_file_bytes,
        SCENE_DURATION_MIN,
        SCENE_DURATION_MAX,
        AUDIO_PADDING_SECONDS,
        TRANSITION_SECONDS,
    )
    from api.tts_service import generate_all_scene_audio
    from api.scene_renderer import render_scene_to_png, close_browser
    import tempfile
    from pathlib import Path

    try:
        from moviepy import ImageClip, AudioFileClip, concatenate_videoclips
    except ImportError:
        from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips

    analyzed = build_mock_analyzed()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_file = os.path.join(project_root, "test_output_video.mp4")
    preview_dir = os.path.join(project_root, "test_scene_previews")

    # Clean and recreate preview dir
    if os.path.exists(preview_dir):
        shutil.rmtree(preview_dir)
    os.makedirs(preview_dir)

    print("=" * 60)
    print("Video Pipeline E2E Test (Multi-frame + Comic + Subtitles)")
    print("=" * 60)

    # Step 1: Build fallback narration (no LLM)
    step_start = time.perf_counter()
    scenes = _fallback_narration_script(analyzed)
    scenes = _normalize_scenes(scenes, analyzed.repo_name)
    print(f"\n[1/5] Narration script built: {len(scenes)} scenes ({time.perf_counter() - step_start:.2f}s)")
    for i, s in enumerate(scenes, 1):
        print(f"  Scene {i}: [{s['section']}] {s['title']}")
        print(f"    Narration: {s['narration'][:100]}...")

    with tempfile.TemporaryDirectory(prefix="repohelper_test_") as tmpdir:
        tmp_path = Path(tmpdir)

        # Step 2: Generate TTS audio
        step_start = time.perf_counter()
        audio_paths = await generate_all_scene_audio(scenes, str(tmp_path), language="en")
        has_audio = any(p is not None for p in audio_paths)
        print(f"\n[2/5] TTS generation: {sum(1 for p in audio_paths if p)}/{len(scenes)} scenes with audio ({time.perf_counter() - step_start:.2f}s)")
        for i, (s, ap) in enumerate(zip(scenes, audio_paths), 1):
            dur = s.get("audio_duration", 0)
            print(f"  Scene {i}: audio={'yes' if ap else 'no'}, duration={dur:.1f}s")

        # Step 3: Update durations from audio
        for scene in scenes:
            audio_duration = scene.get("audio_duration")
            if audio_duration and audio_duration > 0:
                scene["duration_seconds"] = max(
                    SCENE_DURATION_MIN,
                    min(audio_duration + AUDIO_PADDING_SECONDS, SCENE_DURATION_MAX),
                )
        print(f"\n[3/5] Scene durations adjusted:")
        for i, s in enumerate(scenes, 1):
            print(f"  Scene {i}: {s['duration_seconds']:.1f}s")

        # Step 4: Multi-frame rendering with subtitles + highlights
        step_start = time.perf_counter()
        total = len(scenes)
        clips = []
        expansion_counter = 0
        total_frames = 0

        for index, scene in enumerate(scenes, start=1):
            card = _scene_to_card_content(scene, analyzed, index, total)
            card["narration"] = scene.get("narration", "")

            if scene.get("section") == "expansion":
                expansion_counter += 1

            scene_duration = scene["duration_seconds"]
            audio_path = audio_paths[index - 1] if index - 1 < len(audio_paths) else None
            segments = card.get("narration_segments") or [{"text": "", "highlight_labels": [], "duration_fraction": 1.0}]

            print(f"  Scene {index}/{total}: [{scene['section']}] {len(segments)} frames")
            for seg_idx, seg in enumerate(segments):
                print(f"    Frame {seg_idx}: subtitle=\"{seg['text'][:50]}...\" highlight={seg['highlight_labels']}")

            if len(segments) > 1:
                # Multi-frame path
                sub_clips = []
                for seg_idx, seg in enumerate(segments):
                    frame_path = tmp_path / f"scene_{index:02d}_f{seg_idx:02d}.png"
                    seg_duration = max(0.5, scene_duration * seg["duration_fraction"])

                    await render_scene_to_png(
                        card, str(frame_path),
                        expansion_index=expansion_counter or 1,
                        subtitle_text=seg["text"],
                        highlight_labels=seg["highlight_labels"],
                    )

                    # Save first frame as preview
                    if seg_idx == 0:
                        preview_path = os.path.join(preview_dir, f"scene_{index:02d}_{scene['section']}.png")
                        shutil.copy2(str(frame_path), preview_path)
                    # Also save all frames for inspection
                    all_frame_preview = os.path.join(preview_dir, f"scene_{index:02d}_f{seg_idx:02d}.png")
                    shutil.copy2(str(frame_path), all_frame_preview)

                    sub_clip = ImageClip(str(frame_path))
                    if hasattr(sub_clip, "with_duration"):
                        sub_clip = sub_clip.with_duration(seg_duration)
                    else:
                        sub_clip = sub_clip.set_duration(seg_duration)
                    sub_clips.append(sub_clip)
                    total_frames += 1

                scene_clip = concatenate_videoclips(sub_clips, method="compose")

                if audio_path and os.path.exists(audio_path):
                    try:
                        audio_clip = AudioFileClip(audio_path)
                        if hasattr(scene_clip, "with_audio"):
                            scene_clip = scene_clip.with_audio(audio_clip)
                        else:
                            scene_clip = scene_clip.set_audio(audio_clip)
                    except Exception as e:
                        print(f"    Warning: audio attach failed: {e}")

                if hasattr(scene_clip, "fadein"):
                    scene_clip = scene_clip.fadein(TRANSITION_SECONDS).fadeout(TRANSITION_SECONDS)
                clips.append(scene_clip)
            else:
                # Single frame path
                image_path = tmp_path / f"scene_{index:02d}.png"
                await render_scene_to_png(
                    card, str(image_path),
                    expansion_index=expansion_counter or 1,
                    subtitle_text=segments[0]["text"],
                    highlight_labels=segments[0]["highlight_labels"],
                )
                preview_path = os.path.join(preview_dir, f"scene_{index:02d}_{scene['section']}.png")
                shutil.copy2(str(image_path), preview_path)

                clips.append(_build_scene_clip(str(image_path), scene_duration, audio_path=audio_path))
                total_frames += 1

        await close_browser()
        print(f"\n[4/5] Rendered {total_frames} frames across {len(clips)} scenes ({time.perf_counter() - step_start:.2f}s)")

        # Step 5: Compose final video
        step_start = time.perf_counter()
        video_path = tmp_path / "test_output.mp4"
        _compose_final_video(clips, str(video_path), has_audio=has_audio)
        elapsed = time.perf_counter() - step_start
        print(f"\n[5/5] Final MP4 composed ({elapsed:.2f}s)")

        # Copy to project root
        payload = _read_file_bytes(str(video_path))
        with open(output_file, "wb") as f:
            f.write(payload)

    size_kb = len(payload) / 1024
    print(f"\n{'=' * 60}")
    print(f"SUCCESS! Video saved to: {output_file}")
    print(f"Size: {size_kb:.1f} KB, Audio: {has_audio}, Total frames: {total_frames}")
    print(f"Previews saved to: {preview_dir}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(run_test())

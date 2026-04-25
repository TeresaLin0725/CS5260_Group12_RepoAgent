#!/usr/bin/env python3
"""
End-to-end integration test: pick a repo URL, render the full 5-act
onboard video, drop the MP4 on the desktop.

This bypasses the FastAPI server entirely — it runs analyze →
TTS → render → mp4 in-process so you don't have to restart the
backend after .env tweaks.

Usage:
    python scripts/generate_5act_video.py <repo_url> [--out PATH]
                                          [--provider openai|ollama]
                                          [--model MODEL]
                                          [--language en|zh]

Defaults:
    --out      ./outputs/<repo>_5act_<ts>.mp4 in the project root.
               Override with --out if you want it elsewhere
               (e.g. --out "/mnt/c/Users/Admin/Desktop/foo.mp4").
    --provider whatever DEEPWIKI_GENERATOR_PROVIDER says, falling back
               to "openai".
    --language en

Examples:
    # Tiny popular Python CLI lib, fast end-to-end on OpenAI.
    python scripts/generate_5act_video.py https://github.com/pallets/click

    # Output straight to the Windows desktop on WSL.
    python scripts/generate_5act_video.py https://github.com/pallets/click \\
        --out /mnt/c/Users/Admin/Desktop/click_5act.mp4

    # Use local Ollama (slower, free).
    python scripts/generate_5act_video.py https://github.com/pallets/click \\
        --provider ollama --model qwen3:14b
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path

# Force the new pipeline regardless of what .env says, so you can
# re-run without restarting the backend.
os.environ["VIDEO_PIPELINE"] = "onboard_5act"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _default_output_dir() -> Path:
    """Project-relative outputs/ directory; created on first use."""
    return PROJECT_ROOT / "outputs"


def _slugify(repo_url: str) -> str:
    name = repo_url.rstrip("/").split("/")[-1]
    return name.replace(" ", "_") or "repo"


async def _generate(repo_url: str, output_path: Path, provider: str,
                    model: str | None, language: str) -> None:
    from dotenv import load_dotenv
    load_dotenv()

    from api.content_analyzer import RepoAnalysisRequest, analyze_repo_content
    from api.video.onboard_5act import render_onboard_5act_video

    request = RepoAnalysisRequest(
        repo_url=repo_url,
        provider=provider,
        model=model,
        language=language,
        repo_type="github",
    )

    print(f"\n→ Analyzing {repo_url}")
    print(f"  provider={provider}  model={model or '<default>'}  language={language}")
    t0 = time.time()
    analyzed = await analyze_repo_content(request)
    print(f"  ✓ analyzed in {time.time() - t0:.1f}s "
          f"(repo_type={analyzed.repo_type_hint}, "
          f"modules={len(analyzed.key_modules)}, "
          f"metaphor_segments={len(analyzed.metaphor_story or [])})")

    print(f"\n→ Rendering 5-act video")
    t1 = time.time()
    mp4 = await render_onboard_5act_video(analyzed)
    print(f"  ✓ rendered in {time.time() - t1:.1f}s ({len(mp4):,} bytes)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(mp4)
    print(f"\n✓ Saved: {output_path}")
    print(f"  total time: {time.time() - t0:.1f}s\n")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate a full 5-act onboard video for any repo URL.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("repo_url", help="GitHub repo URL")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output MP4 path (default: desktop on WSL, else cwd)")
    ap.add_argument("--provider", default=os.environ.get("DEEPWIKI_GENERATOR_PROVIDER", "openai"),
                    help="LLM provider for analysis (openai/ollama/google/...)")
    ap.add_argument("--model", default=None,
                    help="Specific model name; leave blank for provider default")
    ap.add_argument("--language", default="en", choices=["en", "zh", "ja", "ko"],
                    help="Output language for narration")
    ap.add_argument("--quiet", action="store_true",
                    help="Suppress most logging (only errors + final path)")
    args = ap.parse_args()

    if args.quiet:
        logging.basicConfig(level=logging.ERROR)
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
        # Quiet down a few noisy modules.
        for name in ["urllib3", "httpx", "watchfiles"]:
            logging.getLogger(name).setLevel(logging.WARNING)

    output = args.out
    if output is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        output = _default_output_dir() / f"{_slugify(args.repo_url)}_5act_{ts}.mp4"

    try:
        asyncio.run(_generate(
            repo_url=args.repo_url,
            output_path=output,
            provider=args.provider,
            model=args.model,
            language=args.language,
        ))
    except KeyboardInterrupt:
        print("\n✗ interrupted", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"\n✗ failed: {e}", file=sys.stderr)
        logging.exception("generate_5act_video failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

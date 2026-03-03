#!/usr/bin/env python3
"""
Test runner for the RepoHelper project.

Usage:
    python tests/run_tests.py              # run all tests
    python tests/run_tests.py --unit       # unit tests only
    python tests/run_tests.py --integration
    python tests/run_tests.py --api
    python tests/run_tests.py --check-env  # just check the environment
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_environment():
    """Print environment information useful for debugging test failures."""
    print("Checking test environment...")

    # .env
    env_file = project_root / ".env"
    if env_file.exists():
        print("  .env file found")
        try:
            from dotenv import load_dotenv
            load_dotenv(env_file)
        except ImportError:
            pass
    else:
        print("  .env file NOT found – some tests may be skipped")

    # API keys
    for key, purpose in {
        "GOOGLE_API_KEY": "Google AI embedder live tests",
        "OPENAI_API_KEY": "OpenAI embedder live tests",
    }.items():
        status = "set" if os.getenv(key) else "NOT set"
        print(f"  {key}: {status} ({purpose})")

    # Python packages
    for pkg in ("adalflow", "google.generativeai", "requests", "pytest"):
        try:
            __import__(pkg)
            print(f"  {pkg}: available")
        except ImportError:
            print(f"  {pkg}: MISSING")


def run_pytest(test_dirs: list[str], verbose: bool = False):
    """Run pytest on the given directories."""
    cmd = [sys.executable, "-m", "pytest"]
    if verbose:
        cmd.append("-v")
    cmd.extend(test_dirs)
    result = subprocess.run(cmd, cwd=project_root)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run RepoHelper tests")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--api", action="store_true", help="Run API tests only")
    parser.add_argument("--check-env", action="store_true", help="Only check environment")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    check_environment()

    if args.check_env:
        return

    dirs: list[str] = []
    if args.unit:
        dirs.append("tests/unit")
    if args.integration:
        dirs.append("tests/integration")
    if args.api:
        dirs.append("tests/api")
    if not dirs:
        dirs = ["test", "tests/unit", "tests/integration", "tests/api"]

    print(f"\nRunning tests in: {', '.join(dirs)}")
    success = run_pytest(dirs, verbose=args.verbose)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

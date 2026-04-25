"""
Git metadata extractor — reads commit history, contributors, and releases
from a locally-cloned GitHub repo to support evolutionary/storytelling
narratives in export formats.

Strategy:
  - Local git (via dulwich) for commit iteration — no rate limits, rich data
  - GitHub REST API for contributor stats (with avatars) + releases

Public API:
  - CommitTimeline (pydantic): the structured output
  - extract_commit_timeline(local_path, repo_url, access_token) → CommitTimeline

This module is best-effort: any step can fail (shallow clone, private repo,
network error) and we return a partial/empty timeline rather than raising.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from datetime import datetime, timezone
from typing import List, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# How many recent commits to extract for the narrative.
# 50 is a sweet spot: enough to see the story, not too much for the LLM prompt.
DEFAULT_COMMIT_LIMIT = 50

# Top N contributors to keep
DEFAULT_CONTRIBUTOR_LIMIT = 8

# Max releases to keep
DEFAULT_RELEASE_LIMIT = 10


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class CommitTimelineEntry(BaseModel):
    sha: str = ""              # short SHA (8 chars)
    message: str = ""          # first line, cleaned
    author: str = ""           # name (no email)
    date: str = ""             # ISO 8601


class ContributorInfo(BaseModel):
    login: str = ""
    commit_count: int = 0
    avatar_url: Optional[str] = None
    # Optional GitHub-wide reputation signals (only populated for top contributors
    # that we explicitly enrich via _fetch_contributor_profile). Used to pick
    # "headline" contributors for the onboard video Act 1.
    followers: int = 0
    public_repos: int = 0
    bio: str = ""
    name: str = ""             # display name; falls back to login if empty


class ReleaseInfo(BaseModel):
    tag: str = ""
    date: str = ""             # ISO 8601
    name: str = ""
    # Raw release notes from GitHub (markdown). Often contains a clean
    # changelog when the maintainer wrote one; empty for many small repos.
    body: str = ""
    # Final one-liner shown under the milestone in the Act 1 timeline.
    # Resolved via a 3-step chain (see fill_release_summaries):
    #   1. cleaned `body` if non-empty
    #   2. heuristic on inter-release commit messages
    #   3. (optional, async) LLM fallback when steps 1 & 2 produce nothing.
    summary: str = ""


class RepoStats(BaseModel):
    """Social / liveness signals from the GitHub repo metadata endpoint.

    Useful for the onboarding "is this project worth my time?" check:
    - High stars + recent push = trusted, alive
    - 0 stars + no push for years = abandoned
    """
    stars: int = 0
    watchers: int = 0          # actual subscribers (more meaningful than watchers_count)
    forks: int = 0
    open_issues: int = 0
    pushed_at: str = ""        # ISO 8601 of latest push (liveness signal)
    description: str = ""      # one-line repo blurb from GitHub
    topics: List[str] = Field(default_factory=list)
    license: str = ""


class CommitTimeline(BaseModel):
    """Full commit-history narrative payload attached to AnalyzedContent."""

    commits: List[CommitTimelineEntry] = Field(default_factory=list)
    contributors: List[ContributorInfo] = Field(default_factory=list)
    releases: List[ReleaseInfo] = Field(default_factory=list)
    stats: Optional[RepoStats] = None

    # Simple derived stats for convenience
    total_commits_scanned: int = 0
    first_commit_date: str = ""
    latest_commit_date: str = ""

    def is_empty(self) -> bool:
        return not (self.commits or self.contributors or self.releases or self.stats)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_github_owner_repo(repo_url: str) -> Optional[tuple[str, str]]:
    """Extract (owner, repo) from a GitHub URL. Returns None if not GitHub."""
    if not repo_url:
        return None
    try:
        parsed = urlparse(repo_url.strip())
        if "github" not in (parsed.netloc or "").lower():
            return None
        path = parsed.path.strip("/")
        if path.endswith(".git"):
            path = path[:-4]
        parts = path.split("/")
        if len(parts) < 2:
            return None
        return parts[0], parts[1]
    except Exception:
        return None


def _clean_author(raw: str) -> str:
    """Extract display name from 'Name <email>' format."""
    if not raw:
        return ""
    # 'Seakon <seakon@example.com>' → 'Seakon'
    return raw.split("<")[0].strip().rstrip(",;")


def _clean_message(raw: str) -> str:
    """Keep first line, strip whitespace, cap length."""
    if not raw:
        return ""
    first_line = raw.splitlines()[0].strip() if raw.strip() else ""
    if len(first_line) > 200:
        first_line = first_line[:197] + "..."
    return first_line


# ---------------------------------------------------------------------------
# Commit extraction (dulwich + subprocess fallback)
# ---------------------------------------------------------------------------

def _extract_commits_dulwich(local_path: str, limit: int) -> List[CommitTimelineEntry]:
    """Extract commits using dulwich (pure-Python, no external git needed)."""
    try:
        from dulwich.repo import Repo
    except ImportError:
        logger.warning("dulwich not available; skipping commit extraction")
        return []

    try:
        repo = Repo(local_path)
    except Exception as e:
        logger.warning("Could not open repo at %s with dulwich: %s", local_path, e)
        return []

    commits: List[CommitTimelineEntry] = []
    try:
        walker = repo.get_walker(max_entries=limit)
        for entry in walker:
            commit = entry.commit
            try:
                sha = commit.id.decode("ascii")[:8]
                message = _clean_message(commit.message.decode("utf-8", errors="replace"))
                author = _clean_author(commit.author.decode("utf-8", errors="replace"))
                iso_date = datetime.fromtimestamp(
                    commit.author_time, tz=timezone.utc
                ).isoformat()
                commits.append(CommitTimelineEntry(
                    sha=sha, message=message, author=author, date=iso_date,
                ))
            except Exception as inner:
                logger.debug("Skipping malformed commit: %s", inner)
                continue
    except Exception as e:
        logger.warning("dulwich walker failed: %s", e)
    finally:
        try:
            repo.close()
        except Exception:
            pass

    return commits


def _extract_commits_subprocess(local_path: str, limit: int) -> List[CommitTimelineEntry]:
    """Fallback: parse 'git log' output directly."""
    # Separator unlikely to appear in commit messages
    sep = "<<<<COMMIT_SEP>>>>"
    fmt = f"%H%n%an%n%aI%n%s%n{sep}"
    try:
        result = subprocess.run(
            ["git", "log", f"-{limit}", f"--pretty=format:{fmt}"],
            cwd=local_path,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except Exception as e:
        logger.warning("git log subprocess failed: %s", e)
        return []

    commits: List[CommitTimelineEntry] = []
    for block in result.stdout.split(sep):
        block = block.strip()
        if not block:
            continue
        lines = block.splitlines()
        if len(lines) < 4:
            continue
        commits.append(CommitTimelineEntry(
            sha=lines[0][:8],
            author=_clean_author(lines[1]),
            date=lines[2].strip(),
            message=_clean_message(lines[3]),
        ))
    return commits


# ---------------------------------------------------------------------------
# GitHub REST API (contributors + releases)
# ---------------------------------------------------------------------------

def _github_api_headers(access_token: Optional[str]) -> dict:
    headers = {"Accept": "application/vnd.github+json"}
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"
    return headers


_CONTRIBUTOR_CACHE_TTL_SECONDS = 24 * 3600  # 1 day
_CONTRIBUTOR_CACHE_PATH = os.path.join(
    os.path.expanduser("~"), ".adalflow", "contributor_cache.json"
)


def _load_contributor_cache() -> dict:
    """Load on-disk cache of {login: {"fetched_at": ts, "data": {...}}}."""
    if not os.path.exists(_CONTRIBUTOR_CACHE_PATH):
        return {}
    try:
        import json as _json
        with open(_CONTRIBUTOR_CACHE_PATH, "r", encoding="utf-8") as f:
            return _json.load(f) or {}
    except Exception:
        return {}


def _save_contributor_cache(cache: dict) -> None:
    try:
        import json as _json
        os.makedirs(os.path.dirname(_CONTRIBUTOR_CACHE_PATH), exist_ok=True)
        with open(_CONTRIBUTOR_CACHE_PATH, "w", encoding="utf-8") as f:
            _json.dump(cache, f, ensure_ascii=False)
    except Exception as e:
        logger.debug("Failed to save contributor cache: %s", e)


def _fetch_contributor_profile(login: str, access_token: Optional[str]) -> Optional[dict]:
    """Fetch a single GitHub user profile via GET /users/{login}.

    Used to enrich top contributors with followers/public_repos/bio so we
    can pick "headline" contributors for the onboard video Act 1.

    Caches results to ~/.adalflow/contributor_cache.json with a 1-day TTL
    to bound API rate-limit cost across repeated runs.
    """
    if not login or not login.strip():
        return None
    try:
        import time as _time
        cache = _load_contributor_cache()
        entry = cache.get(login)
        if entry and isinstance(entry, dict):
            fetched_at = float(entry.get("fetched_at") or 0)
            if _time.time() - fetched_at < _CONTRIBUTOR_CACHE_TTL_SECONDS:
                return entry.get("data") or None
    except Exception:
        cache = {}

    try:
        import requests
    except ImportError:
        return None

    url = f"https://api.github.com/users/{login}"
    try:
        resp = requests.get(url, headers=_github_api_headers(access_token), timeout=10)
        if resp.status_code != 200:
            logger.debug("user profile API returned %d for %s", resp.status_code, login)
            return None
        data = resp.json()
    except Exception as e:
        logger.debug("user profile fetch failed for %s: %s", login, e)
        return None

    profile = {
        "followers": int(data.get("followers") or 0),
        "public_repos": int(data.get("public_repos") or 0),
        "bio": (data.get("bio") or "").strip()[:200],
        "name": (data.get("name") or login).strip()[:80],
    }
    try:
        import time as _time
        cache[login] = {"fetched_at": _time.time(), "data": profile}
        _save_contributor_cache(cache)
    except Exception:
        pass
    return profile


def _enrich_top_contributors(
    contributors: List[ContributorInfo],
    access_token: Optional[str],
    limit: int = 5,
) -> None:
    """Enrich the top-N contributors in place with GitHub-wide reputation data.

    Only fetches profiles for the first `limit` contributors (already sorted
    by commit_count from the GitHub API). Caches per-user via
    _fetch_contributor_profile, so repeat runs on the same set of repos are
    free after the first call within the TTL window.
    """
    for c in contributors[:limit]:
        if not c.login or c.login.endswith("[bot]"):
            continue
        profile = _fetch_contributor_profile(c.login, access_token)
        if not profile:
            continue
        c.followers = profile.get("followers", 0)
        c.public_repos = profile.get("public_repos", 0)
        c.bio = profile.get("bio", "")
        c.name = profile.get("name", c.login)


def _fetch_repo_stats(owner: str, repo: str, access_token: Optional[str]) -> Optional[RepoStats]:
    """Fetch stars, watchers, forks, license, topics from /repos/{owner}/{repo}."""
    try:
        import requests
    except ImportError:
        return None
    url = f"https://api.github.com/repos/{owner}/{repo}"
    try:
        resp = requests.get(url, headers=_github_api_headers(access_token), timeout=15)
        if resp.status_code != 200:
            logger.info("repo stats API returned %d for %s/%s", resp.status_code, owner, repo)
            return None
        data = resp.json()
    except Exception as e:
        logger.warning("repo stats fetch failed: %s", e)
        return None

    license_name = ""
    license_obj = data.get("license")
    if isinstance(license_obj, dict):
        license_name = license_obj.get("spdx_id") or license_obj.get("name") or ""

    return RepoStats(
        stars=int(data.get("stargazers_count") or 0),
        watchers=int(data.get("subscribers_count") or 0),  # real subscribers, not the legacy field
        forks=int(data.get("forks_count") or 0),
        open_issues=int(data.get("open_issues_count") or 0),
        pushed_at=data.get("pushed_at") or "",
        description=(data.get("description") or "").strip(),
        topics=[str(t) for t in (data.get("topics") or [])][:10],
        license=license_name,
    )


def _fetch_contributors(owner: str, repo: str, access_token: Optional[str]) -> List[ContributorInfo]:
    try:
        import requests
    except ImportError:
        return []
    url = f"https://api.github.com/repos/{owner}/{repo}/contributors?per_page={DEFAULT_CONTRIBUTOR_LIMIT}&anon=true"
    try:
        resp = requests.get(url, headers=_github_api_headers(access_token), timeout=15)
        if resp.status_code != 200:
            logger.info("contributors API returned %d for %s/%s", resp.status_code, owner, repo)
            return []
        data = resp.json()
    except Exception as e:
        logger.warning("contributors fetch failed: %s", e)
        return []

    out: List[ContributorInfo] = []
    for entry in data[:DEFAULT_CONTRIBUTOR_LIMIT]:
        if not isinstance(entry, dict):
            continue
        out.append(ContributorInfo(
            login=entry.get("login", "") or entry.get("name", "") or "anonymous",
            commit_count=int(entry.get("contributions", 0) or 0),
            avatar_url=entry.get("avatar_url"),
        ))
    return out


def _fetch_releases(owner: str, repo: str, access_token: Optional[str]) -> List[ReleaseInfo]:
    try:
        import requests
    except ImportError:
        return []
    url = f"https://api.github.com/repos/{owner}/{repo}/releases?per_page={DEFAULT_RELEASE_LIMIT}"
    try:
        resp = requests.get(url, headers=_github_api_headers(access_token), timeout=15)
        if resp.status_code != 200:
            logger.info("releases API returned %d for %s/%s", resp.status_code, owner, repo)
            return []
        data = resp.json()
    except Exception as e:
        logger.warning("releases fetch failed: %s", e)
        return []

    out: List[ReleaseInfo] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        out.append(ReleaseInfo(
            tag=entry.get("tag_name", "") or "",
            date=entry.get("published_at", "") or entry.get("created_at", "") or "",
            name=entry.get("name", "") or entry.get("tag_name", "") or "",
            body=(entry.get("body") or "").strip(),
        ))
    return out


# ---------------------------------------------------------------------------
# Release summary builders (C → A chain; B is async, lives in content_analyzer)
# ---------------------------------------------------------------------------

# Liberal match for commits that ship user-facing change. We prefer to
# include too much rather than drop something interesting — the goal is to
# avoid "merge / chore / typo" noise dominating the 3-line summary.
_SUBSTANTIVE_PREFIX = __import__("re").compile(
    r"^\s*(feat(?:ure)?|add|fix|bug(?:fix)?|update|impl(?:ement)?|support|"
    r"enable|allow|introduce|refactor|improve|enhance|optim(?:i[sz]e)?|"
    r"perf|migrate|rewrite|new|major|"
    r"feat\([^)]+\)|fix\([^)]+\))[\s:\-/]",
    __import__("re").IGNORECASE,
)
# Explicit no-go list (commits the user almost never wants in a release blurb).
_SKIP_PREFIX = __import__("re").compile(
    r"^\s*(merge\b|chore\b|docs?[:\(\s]|typo\b|ci[:\(\s]|style[:\(\s]|"
    r"format[:\(\s]|bump\s|release\s|version\s|wip\b|tmp\b)",
    __import__("re").IGNORECASE,
)
# Strip leading "feat:" / "fix(scope):" / etc. from a normalized message
# so the resulting phrase reads naturally in a summary.
_PREFIX_STRIP = __import__("re").compile(
    r"^\s*[a-zA-Z]+(?:\([^)]+\))?\s*[:\-]\s*",
)


def _clean_release_body(body: str, max_chars: int = 140) -> str:
    """Turn raw GitHub release notes (markdown) into a one-line summary.

    Strategy: drop GitHub footers, strip markdown decorations, take the
    first line that looks like prose, truncate at sentence boundary if
    possible.
    """
    if not body:
        return ""
    text = body.strip()
    # Cut everything from the GitHub auto-generated diff / contributors footer.
    for marker in ("**Full Changelog**", "## What's Changed", "## New Contributors"):
        idx = text.find(marker)
        if idx > 0:
            text = text[:idx].strip()
    lines = [
        ln.strip().lstrip("*-•").strip()
        for ln in text.splitlines()
    ]
    candidates = [
        ln for ln in lines
        # Skip empty, headings (only #'s), badge images, and HTML tags.
        if ln and not ln.startswith(("#", "<", "![", "[!"))
    ]
    if not candidates:
        return ""
    # Strip inline markdown like **bold** / `code` / [link](url).
    first = candidates[0]
    first = __import__("re").sub(r"\*\*([^*]+)\*\*", r"\1", first)
    first = __import__("re").sub(r"`([^`]+)`", r"\1", first)
    first = __import__("re").sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", first)
    if len(first) > max_chars:
        # Try to break at a sentence end before the limit.
        cut = first[:max_chars]
        last_period = max(cut.rfind(". "), cut.rfind("。"))
        if last_period >= max_chars * 0.5:
            first = cut[: last_period + 1].strip()
        else:
            first = cut.rstrip(",.;: ").rstrip() + "…"
    return first


def _is_substantive_commit(message: str) -> bool:
    """True if the commit subject reads like a user-facing change."""
    msg = (message or "").strip()
    if not msg:
        return False
    if _SKIP_PREFIX.match(msg):
        return False
    if _SUBSTANTIVE_PREFIX.match(msg):
        return True
    # No conventional prefix → accept short messages that don't smell like noise.
    # Helps with repos that don't follow conventional commits.
    return len(msg.split()) >= 3 and not msg.lower().startswith(("revert", "remove "))


def _extract_change_phrase(message: str, max_chars: int = 60) -> str:
    """Turn one commit subject into a change phrase suitable for joining.

    Examples:
        "feat: add async support" -> "Add async support"
        "fix(parser): handle empty inputs gracefully" -> "Handle empty inputs gracefully"
        "Update README" -> "Update README"
    """
    msg = (message or "").strip()
    if not msg:
        return ""
    # Collapse to first line.
    msg = msg.splitlines()[0].strip()
    msg = _PREFIX_STRIP.sub("", msg)
    # Sentence case (uppercase first letter, lowercase rest if it was ALL CAPS).
    if msg and msg[0].islower():
        msg = msg[0].upper() + msg[1:]
    if len(msg) > max_chars:
        msg = msg[: max_chars - 1].rstrip(",.;: ") + "…"
    return msg


def _heuristic_release_summary(
    in_range_commits: List[CommitTimelineEntry], max_phrases: int = 3,
) -> str:
    """Build a "Adds X · Fixes Y · Refactors Z" summary from inter-release commits."""
    substantive = [c for c in in_range_commits if _is_substantive_commit(c.message)]
    if not substantive:
        return ""
    seen: set = set()
    phrases: List[str] = []
    for c in substantive:
        phrase = _extract_change_phrase(c.message)
        key = phrase.lower()
        if not phrase or key in seen:
            continue
        seen.add(key)
        phrases.append(phrase)
        if len(phrases) >= max_phrases:
            break
    return " · ".join(phrases)


def _commits_between_releases(
    commits: List[CommitTimelineEntry],
    later_date: str,
    earlier_date: Optional[str],
) -> List[CommitTimelineEntry]:
    """Return commits whose date sits in (earlier_date, later_date].

    earlier_date = None means "any commit at or before later_date" (used
    for the oldest release in the list).
    """
    if not later_date:
        return []
    return [
        c for c in commits
        if c.date and c.date <= later_date
        and (not earlier_date or c.date > earlier_date)
    ]


def fill_release_summaries(timeline: "CommitTimeline") -> None:
    """Populate ReleaseInfo.summary in place using the C → A chain.

    C: GitHub-provided release body (cleaned to one line)
    A: heuristic on inter-release commit messages

    The B (LLM) fallback lives in content_analyzer because it needs the
    user's configured provider/model. Releases left without a summary
    here can be filled by ``fill_missing_release_summaries_with_llm``.
    """
    if not timeline or not timeline.releases:
        return
    # Releases come sorted newest-first from the GitHub API.
    releases = timeline.releases
    commits = timeline.commits or []
    for i, release in enumerate(releases):
        # Step C — maintainer-written notes.
        body_summary = _clean_release_body(release.body, max_chars=140)
        if body_summary:
            release.summary = body_summary
            continue
        # Step A — heuristic over commits between the previous (older)
        # release and this one. `releases[i+1]` is the older neighbor
        # because the list is newest-first.
        older_date = (
            releases[i + 1].date if i + 1 < len(releases) else None
        )
        in_range = _commits_between_releases(
            commits, later_date=release.date, earlier_date=older_date,
        )
        release.summary = _heuristic_release_summary(in_range, max_phrases=3)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def extract_commit_timeline(
    local_path: str,
    repo_url: str = "",
    access_token: Optional[str] = None,
    commit_limit: int = DEFAULT_COMMIT_LIMIT,
) -> CommitTimeline:
    """
    Extract commit history + contributors + releases for narrative generation.

    This function is best-effort and **never raises**. On any failure
    (missing dependencies, private repo, network error, invalid path)
    it returns a partial or empty ``CommitTimeline``.

    Args:
        local_path: Path to a locally cloned repo (should contain a .git directory).
        repo_url: GitHub URL, used to fetch contributors/releases via REST API.
        access_token: Optional GitHub PAT for private repos / higher rate limits.
        commit_limit: How many recent commits to extract (default 50).

    Returns:
        CommitTimeline. Always a valid object; may be empty.
    """
    timeline = CommitTimeline()

    try:
        if not local_path or not os.path.isdir(local_path):
            logger.info("extract_commit_timeline: invalid local_path=%s", local_path)
            return timeline

        git_dir = os.path.join(local_path, ".git")
        if not os.path.exists(git_dir):
            logger.info("extract_commit_timeline: no .git dir at %s", local_path)
            return timeline

        # Try dulwich first, fall back to subprocess git log
        commits = _extract_commits_dulwich(local_path, commit_limit)
        if not commits:
            commits = _extract_commits_subprocess(local_path, commit_limit)

        timeline.commits = commits
        timeline.total_commits_scanned = len(commits)
        if commits:
            # commits are in reverse-chronological order (newest first)
            timeline.latest_commit_date = commits[0].date
            timeline.first_commit_date = commits[-1].date

        # Enrich with GitHub API data if we can parse the URL
        owner_repo = _parse_github_owner_repo(repo_url)
        if owner_repo:
            owner, repo = owner_repo
            try:
                timeline.stats = _fetch_repo_stats(owner, repo, access_token)
            except Exception as e:
                logger.info("repo stats fetch skipped: %s", e)
            try:
                timeline.contributors = _fetch_contributors(owner, repo, access_token)
                # Enrich the top contributors with GitHub-wide reputation
                # signals (followers, public_repos, bio, display name).
                # Cached on disk; first-run cost = N extra GET /users/{login}.
                if timeline.contributors:
                    _enrich_top_contributors(timeline.contributors, access_token, limit=5)
            except Exception as e:
                logger.info("contributors fetch skipped: %s", e)
            try:
                timeline.releases = _fetch_releases(owner, repo, access_token)
                # Fill in the human-readable summary for each release using
                # the C (GitHub body) → A (commit-msg heuristic) chain.
                # Step B (LLM) runs later in content_analyzer where the
                # user's provider/model is known.
                fill_release_summaries(timeline)
            except Exception as e:
                logger.info("releases fetch skipped: %s", e)
            stats_summary = (
                f"{timeline.stats.stars}⭐ {timeline.stats.forks}🍴" if timeline.stats else "no stats"
            )
            logger.info(
                "commit_timeline for %s/%s: %d commits, %d contributors, %d releases, %s",
                owner, repo, len(timeline.commits), len(timeline.contributors), len(timeline.releases),
                stats_summary,
            )
        else:
            logger.info(
                "commit_timeline (no GitHub URL): %d commits extracted",
                len(timeline.commits),
            )
    except Exception as e:
        # Absolute last resort — ensure we never break the caller
        logger.warning("extract_commit_timeline swallowed unexpected error: %s", e)

    return timeline


def format_timeline_for_prompt(timeline: CommitTimeline) -> str:
    """Serialize timeline to a compact text block for LLM prompt injection."""
    if timeline.is_empty():
        return ""

    lines = []
    if timeline.stats:
        lines.append("=== REPO SOCIAL STATS ===")
        s = timeline.stats
        lines.append(f"- {s.stars} stars, {s.watchers} watchers, {s.forks} forks, {s.open_issues} open issues")
        if s.pushed_at:
            lines.append(f"- Last push: {s.pushed_at.split('T')[0]}")
        if s.license:
            lines.append(f"- License: {s.license}")
        if s.topics:
            lines.append(f"- Topics: {', '.join(s.topics)}")
        if s.description:
            lines.append(f"- GitHub description: {s.description}")
        lines.append("")

    if timeline.commits:
        lines.append("=== RECENT COMMIT HISTORY (newest first) ===")
        for c in timeline.commits[:40]:
            date = c.date.split("T")[0] if c.date else "?"
            lines.append(f"- {date} [{c.sha}] {c.author}: {c.message}")
    if timeline.contributors:
        lines.append("")
        lines.append("=== TOP CONTRIBUTORS ===")
        for ctr in timeline.contributors[:5]:
            lines.append(f"- {ctr.login} ({ctr.commit_count} commits)")
    if timeline.releases:
        lines.append("")
        lines.append("=== RELEASES ===")
        for r in timeline.releases[:5]:
            date = r.date.split("T")[0] if r.date else "?"
            lines.append(f"- {date}  {r.tag}  {r.name}")

    return "\n".join(lines)

#!/usr/bin/env python3
"""
API endpoint tests for RepoHelper backend.

These tests require the FastAPI server to be running on port 8001.

Usage:
    # First start the backend server:
    #   python -m api.main
    # Then run tests:
    pytest tests/api/test_api.py -v
"""

import os
import sys
import json
import pytest
import requests
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

BASE_URL = os.environ.get("REPOHELPER_API_URL", "http://localhost:8001")


def _server_is_running() -> bool:
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


skip_if_no_server = pytest.mark.skipif(
    not _server_is_running(),
    reason="Backend server is not running"
)


@skip_if_no_server
class TestHealthEndpoint:
    def test_health_returns_200(self):
        r = requests.get(f"{BASE_URL}/health")
        assert r.status_code == 200


@skip_if_no_server
class TestAuthEndpoints:
    def test_auth_status(self):
        r = requests.get(f"{BASE_URL}/auth/status")
        assert r.status_code == 200
        data = r.json()
        assert "auth_required" in data
        assert isinstance(data["auth_required"], bool)

    def test_auth_validate_without_code(self):
        r = requests.post(
            f"{BASE_URL}/auth/validate",
            json={"code": ""},
        )
        # Should return 200 with success=false, or 4xx depending on config
        assert r.status_code in (200, 400, 422)


@skip_if_no_server
class TestModelConfigEndpoint:
    def test_models_config_returns_providers(self):
        r = requests.get(f"{BASE_URL}/models/config")
        assert r.status_code == 200
        data = r.json()
        assert "providers" in data
        assert "defaultProvider" in data
        assert isinstance(data["providers"], list)
        assert len(data["providers"]) > 0


@skip_if_no_server
class TestLangConfigEndpoint:
    def test_lang_config(self):
        r = requests.get(f"{BASE_URL}/lang/config")
        assert r.status_code == 200
        data = r.json()
        # Should return language configuration
        assert isinstance(data, dict)


@skip_if_no_server
class TestChatStreamEndpoint:
    def test_stream_requires_repo_url(self):
        """Chat stream should fail gracefully without a valid repo."""
        r = requests.post(
            f"{BASE_URL}/chat/completions/stream",
            json={
                "repo_url": "",
                "messages": [{"role": "user", "content": "hello"}],
            },
            stream=True,
            timeout=10,
        )
        # Should either return 4xx or start streaming an error
        assert r.status_code in (200, 400, 422, 500)

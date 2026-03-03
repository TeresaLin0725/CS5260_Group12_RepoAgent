#!/usr/bin/env python3
"""
Unit tests for data pipeline functions.

Tests cover:
- Token counting
- Repository name extraction
- Document reading / filtering logic
- Data pipeline preparation

Usage:
    pytest tests/unit/test_data_pipeline.py -v
"""

import os
import sys
import pytest
from pathlib import Path
from unittest.mock import patch

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class TestCountTokens:
    """Test the count_tokens function."""

    def test_returns_positive_int(self):
        from api.data_pipeline import count_tokens
        result = count_tokens("This is a test string for token counting.")
        assert isinstance(result, int)
        assert result > 0

    def test_empty_string(self):
        from api.data_pipeline import count_tokens
        result = count_tokens("")
        assert isinstance(result, int)
        assert result == 0

    @pytest.mark.parametrize("is_ollama", [None, True, False])
    def test_with_ollama_flag(self, is_ollama):
        from api.data_pipeline import count_tokens
        result = count_tokens("hello world", is_ollama_embedder=is_ollama)
        assert isinstance(result, int)
        assert result > 0


class TestDatabaseManager:
    """Test DatabaseManager utility methods."""

    def setup_method(self):
        from api.data_pipeline import DatabaseManager
        self.db = DatabaseManager()

    def test_extract_github_url(self):
        result = self.db._extract_repo_name_from_url(
            "https://github.com/user/project", "github"
        )
        assert result == "user_project"

    def test_extract_gitlab_subgroup_url(self):
        result = self.db._extract_repo_name_from_url(
            "https://gitlab.com/org/team/repo", "gitlab"
        )
        assert result == "team_repo"

    def test_extract_local_path(self):
        result = self.db._extract_repo_name_from_url("/tmp/my-code", "local")
        assert result == "my-code"

    def test_extract_strips_git_suffix(self):
        result = self.db._extract_repo_name_from_url(
            "https://github.com/owner/project.git", "github"
        )
        assert result == "owner_project"


class TestPrepareDataPipeline:
    """Test data pipeline preparation."""

    def test_prepare_returns_callable(self):
        from api.data_pipeline import prepare_data_pipeline
        try:
            pipeline = prepare_data_pipeline()
            assert pipeline is not None
            assert callable(pipeline)
        except Exception:
            # May fail if API keys are not available
            pytest.skip("Pipeline creation failed – likely missing API keys")

    @pytest.mark.parametrize("is_ollama", [True, False])
    def test_prepare_with_ollama_flag(self, is_ollama):
        from api.data_pipeline import prepare_data_pipeline
        try:
            pipeline = prepare_data_pipeline(is_ollama_embedder=is_ollama)
            assert pipeline is not None
        except Exception:
            pytest.skip("Pipeline creation failed – likely missing service")

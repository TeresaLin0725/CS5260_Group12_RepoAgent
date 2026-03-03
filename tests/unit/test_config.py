#!/usr/bin/env python3
"""
Unit tests for configuration module.

Tests cover:
- API key loading from environment
- Auth mode settings
- Model config retrieval
- Default excluded dirs/files

Usage:
    pytest tests/unit/test_config.py -v
"""

import os
import sys
import pytest
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class TestConfigConstants:
    """Test configuration constants and defaults."""

    def test_default_excluded_dirs_is_list(self):
        from api.config import DEFAULT_EXCLUDED_DIRS
        assert isinstance(DEFAULT_EXCLUDED_DIRS, list)
        assert len(DEFAULT_EXCLUDED_DIRS) > 0

    def test_default_excluded_files_is_list(self):
        from api.config import DEFAULT_EXCLUDED_FILES
        assert isinstance(DEFAULT_EXCLUDED_FILES, list)
        assert len(DEFAULT_EXCLUDED_FILES) > 0

    def test_common_dirs_excluded(self):
        from api.config import DEFAULT_EXCLUDED_DIRS
        # node_modules and .git should always be excluded
        dir_names = [d.lower() for d in DEFAULT_EXCLUDED_DIRS]
        assert any("node_modules" in d for d in dir_names)
        assert any(".git" in d for d in dir_names)


class TestAuthConfig:
    """Test authentication configuration."""

    def test_wiki_auth_mode_is_bool(self):
        from api.config import WIKI_AUTH_MODE
        assert isinstance(WIKI_AUTH_MODE, bool)

    def test_wiki_auth_code_is_string(self):
        from api.config import WIKI_AUTH_CODE
        assert isinstance(WIKI_AUTH_CODE, str)


class TestModelConfig:
    """Test model configuration retrieval."""

    def test_get_model_config_returns_dict(self):
        from api.config import get_model_config
        config = get_model_config()
        assert isinstance(config, dict)

    def test_model_config_has_model_client(self):
        from api.config import get_model_config
        config = get_model_config()
        assert 'model_client' in config

    def test_model_config_has_model_kwargs(self):
        from api.config import get_model_config
        config = get_model_config()
        assert 'model_kwargs' in config
        assert isinstance(config['model_kwargs'], dict)
        assert 'model' in config['model_kwargs']

    def test_configs_has_providers(self):
        from api.config import configs
        assert 'providers' in configs
        providers = configs['providers']
        assert isinstance(providers, dict)
        assert len(providers) > 0

    def test_each_provider_has_models(self):
        from api.config import configs
        for pid, pconfig in configs['providers'].items():
            assert 'models' in pconfig, f"Provider {pid} missing 'models' key"

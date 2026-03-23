# pyright: reportMissingImports=false
from __future__ import annotations

import os
from pathlib import Path

import pytest

from finbert_llm_comparison.env import load_openai_api_key


def test_load_openai_api_key_from_open_api_key_env_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPEN_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("OPEN_API_KEY=test-key\n", encoding="utf-8")

    loaded = load_openai_api_key(env_file)

    assert loaded == "test-key"
    assert os.environ["OPENAI_API_KEY"] == "test-key"


def test_load_openai_api_key_prefers_existing_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "shell-key")
    monkeypatch.delenv("OPEN_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=file-key\n", encoding="utf-8")

    loaded = load_openai_api_key(env_file)

    assert loaded == "shell-key"


def test_load_openai_api_key_raises_when_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPEN_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(OSError):
        load_openai_api_key(tmp_path / ".env")

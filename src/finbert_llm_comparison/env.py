from __future__ import annotations

import os
from pathlib import Path


def _parse_env_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("export "):
        stripped = stripped[len("export ") :].strip()

    key, separator, value = stripped.partition("=")
    if not separator:
        raise ValueError(f"Invalid env line: {line!r}")

    key = key.strip()
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return key, value


def load_env_file(env_file: Path) -> None:
    if not env_file.exists():
        return

    for line in env_file.read_text(encoding="utf-8").splitlines():
        parsed = _parse_env_line(line)
        if parsed is None:
            continue
        key, value = parsed
        os.environ.setdefault(key, value)


def load_openai_api_key(env_file: Path) -> str:
    load_env_file(env_file)

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    open_api_key = os.environ.get("OPEN_API_KEY")

    if not openai_api_key and open_api_key:
        os.environ["OPENAI_API_KEY"] = open_api_key
        openai_api_key = open_api_key

    if not openai_api_key:
        raise OSError(
            "OpenAI API 키가 필요합니다. "
            f"{env_file} 파일에 OPEN_API_KEY 또는 OPENAI_API_KEY를 설정하세요."
        )

    return openai_api_key

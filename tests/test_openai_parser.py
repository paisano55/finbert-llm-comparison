# pyright: reportMissingImports=false
import pytest

from finbert_llm_comparison.openai_runner import parse_openai_labels


def test_parse_openai_labels_valid_json() -> None:
    text = '{"labels": ["positive", "neutral", "negative"]}'
    assert parse_openai_labels(text, expected_count=3) == ["positive", "neutral", "negative"]


def test_parse_openai_labels_rejects_mismatch() -> None:
    text = '{"labels": ["positive"]}'
    with pytest.raises(ValueError):
        parse_openai_labels(text, expected_count=2)

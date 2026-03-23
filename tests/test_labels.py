# pyright: reportMissingImports=false
import pytest

from finbert_llm_comparison.labels import normalize_label


def test_normalize_label_basic() -> None:
    assert normalize_label("positive") == "positive"
    assert normalize_label("neutral") == "neutral"
    assert normalize_label("negative") == "negative"


def test_normalize_label_invalid() -> None:
    with pytest.raises(ValueError):
        normalize_label("mixed")

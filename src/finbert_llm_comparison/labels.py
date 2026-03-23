# pyright: reportMissingImports=false
from __future__ import annotations

from .types import SentimentLabel


def normalize_label(raw_label: str) -> SentimentLabel:
    value = raw_label.strip().lower()
    if value in {"positive", "pos"}:
        return "positive"
    if value in {"neutral", "neu"}:
        return "neutral"
    if value in {"negative", "neg"}:
        return "negative"
    raise ValueError(f"Unsupported label value: {raw_label!r}")

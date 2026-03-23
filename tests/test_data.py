# pyright: reportMissingImports=false
from __future__ import annotations

import zipfile
from pathlib import Path

from finbert_llm_comparison.data import (
    _load_phrasebank_records,
    _parse_phrasebank_lines,
    build_finbert_reference_test_indices,
)


def test_finbert_reference_split_size() -> None:
    indices = build_finbert_reference_test_indices(total_size=100, random_state=0)
    assert len(indices) == 20
    assert len(set(indices)) == 20


def test_parse_phrasebank_lines() -> None:
    records = _parse_phrasebank_lines(
        [
            "Operating profit rose strongly @positive",
            "Company outlook remained unchanged @neutral",
            "Net loss widened in the quarter @negative",
        ]
    )

    assert records[0].text == "Operating profit rose strongly"
    assert records[0].label == "positive"
    assert records[1].label == "neutral"
    assert records[2].label == "negative"


def test_load_phrasebank_records_from_hf_archive(tmp_path: Path, monkeypatch) -> None:
    archive_path = tmp_path / "FinancialPhraseBank-v1.0.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "FinancialPhraseBank-v1.0/Sentences_50Agree.txt",
            "Revenue grew in the quarter @positive\n",
        )

    monkeypatch.setattr(
        "finbert_llm_comparison.data.hf_hub_download",
        lambda **kwargs: str(archive_path),
    )

    records = _load_phrasebank_records("sentences_50agree")

    assert len(records) == 1
    assert records[0].text == "Revenue grew in the quarter"
    assert records[0].label == "positive"

# pyright: reportMissingImports=false
from __future__ import annotations

import random
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download
from sklearn.model_selection import train_test_split

from .labels import normalize_label
from .types import DatasetInfo, SentenceRecord

_PHRASEBANK_REPO_ID = "takala/financial_phrasebank"
_PHRASEBANK_ARCHIVE_FILENAME = "data/FinancialPhraseBank-v1.0.zip"
_CONFIG_TO_ARCHIVE_MEMBER = {
    "sentences_50agree": "FinancialPhraseBank-v1.0/Sentences_50Agree.txt",
    "sentences_66agree": "FinancialPhraseBank-v1.0/Sentences_66Agree.txt",
    "sentences_75agree": "FinancialPhraseBank-v1.0/Sentences_75Agree.txt",
    "sentences_allagree": "FinancialPhraseBank-v1.0/Sentences_AllAgree.txt",
}


def _parse_phrasebank_lines(lines: list[str]) -> list[SentenceRecord]:
    records: list[SentenceRecord] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        text, separator, raw_label = stripped.rpartition("@")
        if not separator:
            raise ValueError(f"Unexpected PhraseBank row format: {line!r}")

        text = text.strip()
        if not text:
            continue

        records.append(SentenceRecord(text=text, label=normalize_label(raw_label)))
    return records


def _load_phrasebank_records(dataset_config: str) -> list[SentenceRecord]:
    archive_member = _CONFIG_TO_ARCHIVE_MEMBER.get(dataset_config)
    if archive_member is None:
        supported = ", ".join(sorted(_CONFIG_TO_ARCHIVE_MEMBER))
        raise ValueError(
            f"Unsupported dataset config: {dataset_config!r}. Supported values: {supported}"
        )

    archive_path = Path(
        hf_hub_download(
            repo_id=_PHRASEBANK_REPO_ID,
            filename=_PHRASEBANK_ARCHIVE_FILENAME,
            repo_type="dataset",
        )
    )
    with zipfile.ZipFile(archive_path) as archive:
        with archive.open(archive_member) as raw_file:
            decoded_lines = raw_file.read().decode("iso-8859-1").splitlines()
    return _parse_phrasebank_lines(decoded_lines)


def build_finbert_reference_test_indices(total_size: int, random_state: int = 0) -> list[int]:
    all_indices = list(range(total_size))
    _, test_indices = train_test_split(all_indices, test_size=0.2, random_state=random_state)
    return test_indices


def load_evaluation_dataset(
    mode: str,
    dataset_config: str,
    fallback_sample_size: int,
    fallback_seed: int,
) -> tuple[list[SentenceRecord], DatasetInfo]:
    records = _load_phrasebank_records(dataset_config)

    if mode == "finbert_reference":
        test_indices = build_finbert_reference_test_indices(len(records), random_state=0)
        test_records = [records[idx] for idx in test_indices]
        info = DatasetInfo(
            source="takala/financial_phrasebank:data/FinancialPhraseBank-v1.0.zip",
            config=dataset_config,
            strategy="finbert_method_equivalent_test_split_20pct_random_state_0",
            split="test",
            size=len(test_records),
        )
        return test_records, info

    if mode == "phrasebank_1000":
        sample_size = min(fallback_sample_size, len(records))
        sampled = random.Random(fallback_seed).sample(records, sample_size)
        info = DatasetInfo(
            source="takala/financial_phrasebank:data/FinancialPhraseBank-v1.0.zip",
            config=dataset_config,
            strategy=f"random_sample_{sample_size}_seed_{fallback_seed}",
            split="train",
            size=len(sampled),
        )
        return sampled, info

    raise ValueError(f"Unsupported dataset mode: {mode}")

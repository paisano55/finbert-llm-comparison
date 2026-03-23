# pyright: reportMissingImports=false
from __future__ import annotations

import random

from datasets import ClassLabel, load_dataset
from sklearn.model_selection import train_test_split

from .labels import normalize_label
from .types import DatasetInfo, SentenceRecord


def _to_sentence_records(dataset) -> list[SentenceRecord]:
    label_feature = dataset.features.get("label")
    records: list[SentenceRecord] = []
    for row in dataset:
        text = str(row["sentence"]).strip()
        if not text:
            continue

        label_value = row["label"]
        if isinstance(label_feature, ClassLabel):
            label_text = label_feature.int2str(int(label_value))
        else:
            label_text = str(label_value)
        records.append(SentenceRecord(text=text, label=normalize_label(str(label_text))))
    return records


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
    phrasebank = load_dataset("takala/financial_phrasebank", dataset_config, split="train")
    records = _to_sentence_records(phrasebank)

    if mode == "finbert_reference":
        test_indices = build_finbert_reference_test_indices(len(records), random_state=0)
        test_records = [records[idx] for idx in test_indices]
        info = DatasetInfo(
            source="takala/financial_phrasebank",
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
            source="takala/financial_phrasebank",
            config=dataset_config,
            strategy=f"random_sample_{sample_size}_seed_{fallback_seed}",
            split="train",
            size=len(sampled),
        )
        return sampled, info

    raise ValueError(f"Unsupported dataset mode: {mode}")

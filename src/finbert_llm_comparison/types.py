# pyright: reportMissingImports=false
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

SentimentLabel = Literal["positive", "neutral", "negative"]


@dataclass(frozen=True)
class SentenceRecord:
    text: str
    label: SentimentLabel


@dataclass(frozen=True)
class DatasetInfo:
    source: str
    config: str
    strategy: str
    split: str
    size: int


@dataclass(frozen=True)
class UsageStats:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class ModelRunResult:
    predictions: list[SentimentLabel]
    elapsed_seconds: float
    usage: UsageStats | None = None


@dataclass(frozen=True)
class BenchmarkConfig:
    dataset_mode: Literal["finbert_reference", "phrasebank_1000"] = "finbert_reference"
    dataset_config: str = "sentences_50agree"
    fallback_sample_size: int = 1000
    fallback_seed: int = 42
    run_target: Literal["both", "finbert", "openai"] = "both"
    finbert_model_name: str = "ProsusAI/finbert"
    finbert_batch_size: int = 64
    openai_model_name: str = "gpt-4o-mini"
    openai_batch_size: int = 16
    openai_max_retries: int = 3
    openai_retry_base_seconds: float = 1.0


@dataclass(frozen=True)
class BenchmarkReport:
    dataset: DatasetInfo
    run_target: Literal["both", "finbert", "openai"]
    finbert: dict[str, float | str] | None
    openai: dict[str, float | str | int] | None

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset": asdict(self.dataset),
            "run_target": self.run_target,
            "finbert": self.finbert,
            "openai": self.openai,
        }

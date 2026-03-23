# pyright: reportMissingImports=false
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score

from .data import load_evaluation_dataset
from .finbert_runner import FinBERTSentimentRunner
from .openai_runner import OpenAIBatchSentimentRunner
from .types import BenchmarkConfig, BenchmarkReport, SentimentLabel


def _metrics(gold: Sequence[SentimentLabel], pred: Sequence[SentimentLabel]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(gold, pred)),
        "macro_f1": float(f1_score(gold, pred, average="macro")),
    }


def run_benchmark(config: BenchmarkConfig, output_path: Path) -> BenchmarkReport:
    records, dataset_info = load_evaluation_dataset(
        mode=config.dataset_mode,
        dataset_config=config.dataset_config,
        fallback_sample_size=config.fallback_sample_size,
        fallback_seed=config.fallback_seed,
    )
    gold: list[SentimentLabel] = [record.label for record in records]

    finbert_runner = FinBERTSentimentRunner(
        model_name=config.finbert_model_name,
        batch_size=config.finbert_batch_size,
    )
    finbert_result = finbert_runner.predict(records)
    finbert_metrics = _metrics(gold=gold, pred=finbert_result.predictions)

    openai_runner = OpenAIBatchSentimentRunner(
        model_name=config.openai_model_name,
        batch_size=config.openai_batch_size,
        max_retries=config.openai_max_retries,
        retry_base_seconds=config.openai_retry_base_seconds,
    )
    openai_result = openai_runner.predict(records)
    openai_metrics = _metrics(gold=gold, pred=openai_result.predictions)

    report = BenchmarkReport(
        dataset=dataset_info,
        finbert={
            "model": config.finbert_model_name,
            **finbert_metrics,
            "elapsed_seconds": finbert_result.elapsed_seconds,
        },
        openai={
            "model": config.openai_model_name,
            **openai_metrics,
            "elapsed_seconds": openai_result.elapsed_seconds,
            "prompt_tokens": openai_result.usage.prompt_tokens if openai_result.usage else 0,
            "completion_tokens": openai_result.usage.completion_tokens
            if openai_result.usage
            else 0,
            "total_tokens": openai_result.usage.total_tokens if openai_result.usage else 0,
        },
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report

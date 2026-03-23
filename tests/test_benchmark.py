# pyright: reportMissingImports=false
from __future__ import annotations

import json
from pathlib import Path

from finbert_llm_comparison import benchmark
from finbert_llm_comparison.types import BenchmarkConfig, DatasetInfo, ModelRunResult, SentenceRecord, UsageStats


class _FakeFinBERTSentimentRunner:
    called = False

    def __init__(self, model_name: str, batch_size: int) -> None:
        self.model_name = model_name
        self.batch_size = batch_size

    def predict(self, records: list[SentenceRecord]) -> ModelRunResult:
        _FakeFinBERTSentimentRunner.called = True
        return ModelRunResult(
            predictions=[record.label for record in records],
            elapsed_seconds=1.0,
        )


class _FakeOpenAIBatchSentimentRunner:
    called = False

    def __init__(
        self,
        model_name: str,
        batch_size: int,
        max_retries: int,
        retry_base_seconds: float,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_base_seconds = retry_base_seconds

    def predict(self, records: list[SentenceRecord]) -> ModelRunResult:
        _FakeOpenAIBatchSentimentRunner.called = True
        return ModelRunResult(
            predictions=[record.label for record in records],
            elapsed_seconds=2.0,
            usage=UsageStats(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )


def _install_fakes(monkeypatch) -> None:
    records = [
        SentenceRecord(text="profit increased", label="positive"),
        SentenceRecord(text="guidance unchanged", label="neutral"),
    ]
    dataset = DatasetInfo(
        source="test",
        config="sentences_50agree",
        strategy="fixture",
        split="test",
        size=len(records),
    )

    monkeypatch.setattr(
        benchmark,
        "load_evaluation_dataset",
        lambda **kwargs: (records, dataset),
    )
    monkeypatch.setattr(benchmark, "FinBERTSentimentRunner", _FakeFinBERTSentimentRunner)
    monkeypatch.setattr(benchmark, "OpenAIBatchSentimentRunner", _FakeOpenAIBatchSentimentRunner)
    _FakeFinBERTSentimentRunner.called = False
    _FakeOpenAIBatchSentimentRunner.called = False


def test_run_benchmark_finbert_only(tmp_path: Path, monkeypatch) -> None:
    _install_fakes(monkeypatch)

    report = benchmark.run_benchmark(
        config=BenchmarkConfig(run_target="finbert"),
        output_path=tmp_path / "report.json",
    )

    assert report.run_target == "finbert"
    assert report.finbert is not None
    assert report.openai is None
    assert _FakeFinBERTSentimentRunner.called is True
    assert _FakeOpenAIBatchSentimentRunner.called is False


def test_run_benchmark_openai_only(tmp_path: Path, monkeypatch) -> None:
    _install_fakes(monkeypatch)

    report = benchmark.run_benchmark(
        config=BenchmarkConfig(run_target="openai"),
        output_path=tmp_path / "report.json",
    )

    assert report.run_target == "openai"
    assert report.finbert is None
    assert report.openai is not None
    assert report.openai["total_tokens"] == 15
    assert _FakeFinBERTSentimentRunner.called is False
    assert _FakeOpenAIBatchSentimentRunner.called is True


def test_run_benchmark_both_serializes_all_sections(tmp_path: Path, monkeypatch) -> None:
    _install_fakes(monkeypatch)
    output_path = tmp_path / "report.json"

    benchmark.run_benchmark(
        config=BenchmarkConfig(run_target="both"),
        output_path=output_path,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["run_target"] == "both"
    assert payload["finbert"]["accuracy"] == 1.0
    assert payload["openai"]["total_tokens"] == 15

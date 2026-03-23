# pyright: reportMissingImports=false
from __future__ import annotations

from dataclasses import dataclass

from finbert_llm_comparison import cli


@dataclass(frozen=True)
class _DummyReport:
    def to_dict(self) -> dict[str, object]:
        return {
            "dataset": {
                "source": "test",
                "config": "sentences_50agree",
                "strategy": "fixture",
                "split": "test",
                "size": 1,
            },
            "run_target": "finbert",
            "finbert": {
                "model": "dummy",
                "accuracy": 1.0,
                "macro_f1": 1.0,
                "elapsed_seconds": 1.0,
            },
            "openai": None,
        }


def test_main_finbert_only_skips_openai_key_and_transfer_checks(monkeypatch, capsys) -> None:
    called = {"load_openai_api_key": False, "run_benchmark": False}

    monkeypatch.setattr(
        cli,
        "load_openai_api_key",
        lambda path: called.__setitem__("load_openai_api_key", True),
    )
    monkeypatch.setattr(cli, "run_benchmark", lambda config, output_path: _DummyReport())
    monkeypatch.setattr(cli.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        "sys.argv",
        ["fin-sentiment-benchmark", "--run-target", "finbert"],
    )

    cli.main()

    captured = capsys.readouterr()
    assert called["load_openai_api_key"] is False
    assert "CUDA available: False" in captured.out


def test_main_openai_only_skips_finbert_batch_validation(monkeypatch) -> None:
    monkeypatch.setattr(cli, "load_openai_api_key", lambda path: "test-key")
    monkeypatch.setattr(cli, "run_benchmark", lambda config, output_path: _DummyReport())
    monkeypatch.setattr(
        "sys.argv",
        [
            "fin-sentiment-benchmark",
            "--run-target",
            "openai",
            "--allow-openai-data-transfer",
            "--finbert-batch-size",
            "0",
        ],
    )

    cli.main()

# pyright: reportMissingImports=false
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from .benchmark import run_benchmark
from .types import BenchmarkConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark FinBERT vs OpenAI financial sentiment analysis"
    )
    parser.add_argument(
        "--dataset-mode",
        choices=["finbert_reference", "phrasebank_1000"],
        default="finbert_reference",
        help=(
            "finbert_reference: FinBERT repo 방식(72/8/20 중 test), "
            "phrasebank_1000: 1000개 랜덤 샘플"
        ),
    )
    parser.add_argument("--dataset-config", default="sentences_50agree")
    parser.add_argument("--fallback-sample-size", type=int, default=1000)
    parser.add_argument("--fallback-seed", type=int, default=42)
    parser.add_argument("--finbert-model", default="ProsusAI/finbert")
    parser.add_argument("--finbert-batch-size", type=int, default=64)
    parser.add_argument("--openai-model", default="gpt-4o-mini")
    parser.add_argument("--openai-batch-size", type=int, default=16)
    parser.add_argument("--openai-max-retries", type=int, default=3)
    parser.add_argument("--openai-retry-base-seconds", type=float, default=1.0)
    parser.add_argument("--output-path", default="outputs/benchmark_result.json")
    parser.add_argument(
        "--allow-openai-data-transfer",
        action="store_true",
        help="동의 플래그: 데이터 문장을 OpenAI API로 전송하는 것을 허용",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise OSError("OPENAI_API_KEY 환경변수가 필요합니다.")
    if not args.allow_openai_data_transfer:
        raise OSError(
            "OpenAI 비교 실행은 문장 텍스트를 외부(OpenAI)로 전송합니다. "
            "동의 시 --allow-openai-data-transfer 플래그를 추가하세요."
        )

    if args.finbert_batch_size <= 0:
        raise ValueError("--finbert-batch-size 는 1 이상이어야 합니다.")
    if args.openai_batch_size <= 0:
        raise ValueError("--openai-batch-size 는 1 이상이어야 합니다.")
    if args.openai_max_retries < 1:
        raise ValueError("--openai-max-retries 는 1 이상이어야 합니다.")

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    config = BenchmarkConfig(
        dataset_mode=args.dataset_mode,
        dataset_config=args.dataset_config,
        fallback_sample_size=args.fallback_sample_size,
        fallback_seed=args.fallback_seed,
        finbert_model_name=args.finbert_model,
        finbert_batch_size=args.finbert_batch_size,
        openai_model_name=args.openai_model,
        openai_batch_size=args.openai_batch_size,
        openai_max_retries=args.openai_max_retries,
        openai_retry_base_seconds=args.openai_retry_base_seconds,
    )

    report = run_benchmark(config=config, output_path=Path(args.output_path))
    print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

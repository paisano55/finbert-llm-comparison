# pyright: reportMissingImports=false
from __future__ import annotations

import json
import time

from openai import OpenAI
from tqdm import tqdm

from .labels import normalize_label
from .types import ModelRunResult, SentenceRecord, SentimentLabel, UsageStats

SYSTEM_PROMPT = (
    "You are a financial sentiment classifier. "
    "Classify each sentence into one label: positive, neutral, or negative. "
    'Return strict JSON only in this exact format: {"labels": ["positive", ...]}.'
)


def parse_openai_labels(raw_text: str, expected_count: int) -> list[SentimentLabel]:
    text = raw_text.strip()
    if not text:
        raise ValueError("OpenAI response text is empty")

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"OpenAI response is not valid JSON object: {raw_text!r}")

    payload = json.loads(text[start : end + 1])
    labels = payload.get("labels")
    if not isinstance(labels, list):
        raise ValueError("OpenAI JSON must contain list field 'labels'")
    if len(labels) != expected_count:
        raise ValueError(f"Label count mismatch: expected {expected_count}, got {len(labels)}")

    return [normalize_label(str(label)) for label in labels]


class OpenAIBatchSentimentRunner:
    def __init__(
        self,
        model_name: str,
        batch_size: int,
        max_retries: int,
        retry_base_seconds: float,
    ) -> None:
        self.client = OpenAI()
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_base_seconds = retry_base_seconds

    def _build_user_prompt(self, batch_texts: list[str]) -> str:
        lines = ["Classify the sentiment for each financial sentence."]
        lines.append("Sentences:")
        for idx, sentence in enumerate(batch_texts, start=1):
            lines.append(f"{idx}. {sentence}")
        lines.append(
            'Return JSON only with key \'labels\'. Example: {"labels":["positive","neutral"]}.'
        )
        return "\n".join(lines)

    def _run_single_batch(self, batch_texts: list[str]) -> tuple[list[SentimentLabel], UsageStats]:
        prompt = self._build_user_prompt(batch_texts)
        cumulative_usage = UsageStats()

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    temperature=0,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                )

                usage = response.usage
                attempt_usage = UsageStats(
                    prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                    completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
                    total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
                )
                cumulative_usage = UsageStats(
                    prompt_tokens=cumulative_usage.prompt_tokens + attempt_usage.prompt_tokens,
                    completion_tokens=cumulative_usage.completion_tokens
                    + attempt_usage.completion_tokens,
                    total_tokens=cumulative_usage.total_tokens + attempt_usage.total_tokens,
                )

                content = response.choices[0].message.content or ""
                labels = parse_openai_labels(content, expected_count=len(batch_texts))
                return labels, cumulative_usage
            except Exception:
                if attempt >= self.max_retries - 1:
                    raise
                delay = self.retry_base_seconds * (2**attempt)
                time.sleep(delay)

        raise RuntimeError("Retry loop exited unexpectedly")

    def predict(self, records: list[SentenceRecord]) -> ModelRunResult:
        texts = [record.text for record in records]
        predictions: list[SentimentLabel] = []
        total_usage = UsageStats()

        started = time.perf_counter()
        for start in tqdm(
            range(0, len(texts), self.batch_size),
            desc="OpenAI batched inference",
            unit="batch",
        ):
            batch_texts = texts[start : start + self.batch_size]
            batch_predictions, batch_usage = self._run_single_batch(batch_texts)
            predictions.extend(batch_predictions)
            total_usage = UsageStats(
                prompt_tokens=total_usage.prompt_tokens + batch_usage.prompt_tokens,
                completion_tokens=total_usage.completion_tokens + batch_usage.completion_tokens,
                total_tokens=total_usage.total_tokens + batch_usage.total_tokens,
            )

        elapsed = time.perf_counter() - started
        return ModelRunResult(predictions=predictions, elapsed_seconds=elapsed, usage=total_usage)

# pyright: reportMissingImports=false
from __future__ import annotations

from time import perf_counter

import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .labels import normalize_label
from .types import ModelRunResult, SentenceRecord, SentimentLabel


class FinBERTSentimentRunner:
    def __init__(self, model_name: str, batch_size: int) -> None:
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.id_to_label = self._build_id_to_label_map()

    def _build_id_to_label_map(self) -> dict[int, str]:
        id_to_label: dict[int, str] = {}
        raw = getattr(self.model.config, "id2label", {})
        for key, value in raw.items():
            id_to_label[int(key)] = str(value)
        return id_to_label

    def predict(self, records: list[SentenceRecord]) -> ModelRunResult:
        texts = [record.text for record in records]
        predictions: list[SentimentLabel] = []

        started = perf_counter()
        for start in tqdm(
            range(0, len(texts), self.batch_size),
            desc="FinBERT inference",
            unit="batch",
        ):
            batch_texts = texts[start : start + self.batch_size]
            encoded = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512,
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}

            with torch.inference_mode():
                logits = self.model(**encoded).logits
                pred_ids = torch.argmax(logits, dim=-1).tolist()

            for pred_id in pred_ids:
                raw_label = self.id_to_label.get(pred_id, str(pred_id))
                predictions.append(normalize_label(raw_label))

        elapsed = perf_counter() - started
        return ModelRunResult(predictions=predictions, elapsed_seconds=elapsed)

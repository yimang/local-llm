"""Offline inference shared by CLI, evaluation and benchmarks."""

import time

import torch

from cardseg.artifacts import load_artifact
from cardseg.data import collate, encode
from cardseg.decode import decode
from cardseg.model import choose_device


class Predictor:
    def __init__(self, path, device="auto"):
        torch.set_num_threads(4)
        self.device = choose_device(device)
        self.model, self.tokenizer, self.metadata = load_artifact(path, self.device)

    def sync(self):
        if self.device == "mps":
            torch.mps.synchronize()

    def predict(self, query, timing=False):
        self.sync()
        start = time.perf_counter()
        if not isinstance(query, str):
            raise ValueError("query must be a string")
        result: dict = dict(query=query, segments=[])
        if not query.strip():
            if timing:
                result.update(
                    timing_ms=dict(
                        tokenization=0.0,
                        forward=0.0,
                        decoding=0.0,
                        total=(time.perf_counter() - start) * 1000,
                    ),
                    bio_repairs=0,
                    tokens=0,
                )
            return result
        row = encode(self.tokenizer, query, max_length=self.metadata["max_length"])
        tokenized = time.perf_counter()
        batch = collate([row], self.tokenizer.pad_token_id, self.device)
        with torch.no_grad():
            logits = self.model(batch["input_ids"], batch["attention_mask"])
        self.sync()
        forwarded = time.perf_counter()
        result["segments"], repairs = decode(
            query, row["offsets"], logits[0].argmax(-1).tolist()
        )
        end = time.perf_counter()
        if timing:
            result["timing_ms"] = dict(
                tokenization=(tokenized - start) * 1000,
                forward=(forwarded - tokenized) * 1000,
                decoding=(end - forwarded) * 1000,
                total=(end - start) * 1000,
            )
            result["bio_repairs"] = repairs
            result["tokens"] = len(row["input_ids"])
        return result

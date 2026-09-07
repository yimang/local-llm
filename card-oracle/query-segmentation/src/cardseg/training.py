"""Small utilities shared by frozen-head training and encoder adaptation."""

import hashlib
import json
import math
from pathlib import Path

import torch

from cardseg.data import collate
from cardseg.decode import decode


def fingerprint(parameters):
    digest = hashlib.sha256()
    for p in parameters:
        digest.update(p.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def validate_config(config):
    for key in ("epochs", "batch_size", "max_length", "patience"):
        value = config[key]
        if type(value) is not int or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("learning_rate", "head_learning_rate", "weight_decay"):
        if key == "head_learning_rate" and key not in config:
            continue
        value = config[key]
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or (value < 0 if key == "weight_decay" else value <= 0)
        ):
            constraint = "nonnegative" if key == "weight_decay" else "positive"
            raise ValueError(f"{key} must be finite and {constraint}")


def decode_batch(rows, logits):
    predictions, repairs = [], 0
    for row, token_logits in zip(rows, logits.detach().cpu(), strict=True):
        spans, fixed = decode(
            row["query"], row["offsets"], token_logits[: len(row["offsets"])]
        )
        predictions.append(spans)
        repairs += fixed
    return predictions, repairs


def write_reload_reference(path, model, tokenizer, rows):
    """Capture the in-memory trained model, never a reloaded artifact.

    Called after training/export; leaves the model on CPU in evaluation mode.
    """
    model.to("cpu").eval()
    reference = []
    with torch.no_grad():
        for row in rows:
            batch = collate([row], tokenizer.pad_token_id, "cpu")
            logits = model(batch["input_ids"], batch["attention_mask"])[0]
            spans, _ = decode(row["query"], row["offsets"], logits)
            reference.append(
                dict(query=row["query"], logits=logits.tolist(), spans=spans)
            )
    Path(path).write_text(json.dumps(reference) + "\n")

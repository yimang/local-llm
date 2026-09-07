"""Validated character-span to BIO alignment, retaining original offsets."""

import json
from pathlib import Path

import torch

TYPES = [
    "SUBJECT",
    "GAME",
    "SET",
    "CARD_NUMBER",
    "YEAR",
    "GRADER",
    "GRADE",
    "CONDITION",
]
LABELS = ["O"] + [f"{prefix}-{kind}" for kind in TYPES for prefix in ("B", "I")]


def encode(tokenizer, query, spans=None, max_length=128):
    if not isinstance(query, str):
        raise ValueError("query must be a string")
    previous = 0
    for span in spans or []:
        start, end = span["start"], span["end"]
        if (
            span["label"] not in TYPES
            or type(start) is not int
            or type(end) is not int
            or not previous <= start < end <= len(query)
            or not query[start:end].strip()
        ):
            raise ValueError(f"invalid span: {span}")
        previous = end
    encoded = tokenizer(query, return_offsets_mapping=True, truncation=False)
    if len(encoded["input_ids"]) > max_length:
        raise ValueError(f"input exceeds {max_length} tokens; truncation is disabled")
    offsets, targets, seen = [], [], set()
    for start, end in encoded["offset_mapping"]:
        while start < end and query[start].isspace():
            start += 1
        while end > start and query[end - 1].isspace():
            end -= 1
        offsets.append((start, end))
        if start == end:
            targets.append(-100)
            continue
        matches = [
            (i, s)
            for i, s in enumerate(spans or [])
            if start < s["end"] and end > s["start"]
        ]
        if not matches:
            targets.append(0)
            continue
        i, span = matches[0]
        if len(matches) != 1 or start < span["start"] or end > span["end"]:
            raise ValueError(f"token {(start, end)} crosses annotation boundary")
        targets.append(LABELS.index(f"{'I' if i in seen else 'B'}-{span['label']}"))
        seen.add(i)
    if spans is not None:
        if len(seen) != len(spans):
            raise ValueError("annotation has no aligned tokens")
        for span in spans:
            covered = [
                o
                for o, t in zip(offsets, targets, strict=True)
                if t > 0 and o[0] >= span["start"] and o[1] <= span["end"]
            ]
            if (
                min(o[0] for o in covered) != span["start"]
                or max(o[1] for o in covered) != span["end"]
            ):
                raise ValueError(f"annotation boundaries cannot round trip: {span}")
    return dict(
        query=query,
        spans=spans,
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        offsets=offsets,
        labels=targets,
    )


def read_data(path, tokenizer, max_length=128):
    rows = []
    for line, raw in enumerate(Path(path).read_text().splitlines(), 1):
        try:
            row = json.loads(raw)
            if not isinstance(row["spans"], list):
                raise ValueError("spans must be a list; use [] for negative examples")
            rows.append(encode(tokenizer, row["query"], row["spans"], max_length))
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"{path}:{line}: {exc}") from exc
    if not rows:
        raise ValueError(f"{path}: empty dataset")
    return rows


def collate(rows, pad_id, device):
    length = max(len(r["input_ids"]) for r in rows)
    result = {}
    for key, fill in [("input_ids", pad_id), ("attention_mask", 0), ("labels", -100)]:
        result[key] = torch.tensor(
            [r[key] + [fill] * (length - len(r[key])) for r in rows], device=device
        )
    return result

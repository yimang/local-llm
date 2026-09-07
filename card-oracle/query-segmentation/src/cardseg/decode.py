"""Shared BIO decoding; no template completion or catalog heuristics."""

import torch

from cardseg.data import LABELS


def decode(query, offsets, logits):
    """Pool overlapping character offsets before applying BIO boundaries.

    Byte-level tokens may cover the same Unicode character. Average their raw
    logits to choose one tag for the whole overlapping group. Touching offsets
    remain separate so adjacent entities retain their B boundaries.
    """
    logits = torch.as_tensor(logits).detach().cpu()
    if logits.shape != (len(offsets), len(LABELS)):
        raise ValueError("logits must have shape (number of offsets, number of labels)")
    predictions = logits.argmax(-1).tolist()
    groups = []
    for i, (start, end) in enumerate(offsets):
        if start == end:
            continue
        if groups and start < groups[-1][1]:
            old_start, old_end, indices = groups[-1]
            groups[-1] = (min(old_start, start), max(old_end, end), [*indices, i])
        else:
            groups.append((start, end, [i]))
    spans, repairs = [], 0
    active = None
    for start, end, indices in groups:
        prediction = (
            predictions[indices[0]]
            if len(indices) == 1
            else int(logits[indices].mean(0).argmax())
        )
        tag = LABELS[prediction]
        if tag == "O":
            active = None
            continue
        prefix, kind = tag.split("-", 1)
        if prefix == "I" and active is not None and active["label"] == kind:
            active["end"] = end
            active["text"] = query[active["start"] : end]
        else:
            repairs += int(prefix == "I")
            active = dict(start=start, end=end, label=kind, text=query[start:end])
            spans.append(active)
    return spans, repairs

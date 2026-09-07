"""Shared BIO decoding; no template completion or catalog heuristics."""

from cardseg.data import LABELS


def decode(query, offsets, predictions):
    spans, repairs = [], 0
    active = None
    for (start, end), prediction in zip(offsets, predictions, strict=True):
        if start == end:
            continue
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

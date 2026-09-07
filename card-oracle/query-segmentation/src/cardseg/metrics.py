"""Exact character-boundary and label metrics."""

from cardseg.data import TYPES


def keys(spans):
    return {(s["start"], s["end"], s["label"]) for s in spans}


def score(golds, predictions):
    def stats(gold, pred):
        tp = sum(len(g & p) for g, p in zip(gold, pred, strict=True))
        ng, np = sum(map(len, gold)), sum(map(len, pred))
        return dict(
            precision=tp / np if np else 0,
            recall=tp / ng if ng else 0,
            f1=2 * tp / (ng + np) if ng + np else 0,
            support=ng,
        )

    gs, ps = list(map(keys, golds)), list(map(keys, predictions))
    result = stats(gs, ps)
    result["query_exact_match"] = sum(
        g == p for g, p in zip(gs, ps, strict=True)
    ) / len(gs)
    result["per_label"] = {
        k: stats(
            [{s for s in g if s[2] == k} for g in gs],
            [{s for s in p if s[2] == k} for p in ps],
        )
        for k in TYPES
    }
    supported = [s["f1"] for s in result["per_label"].values() if s["support"]]
    result["macro_f1"] = sum(supported) / len(supported) if supported else 0
    return result

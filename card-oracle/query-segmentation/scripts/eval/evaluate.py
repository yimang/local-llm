# ruff: noqa: E402
"""Synthetic exact-span evaluation using the identical demo inference path."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cardseg.data import read_data
from cardseg.inference import Predictor
from cardseg.metrics import keys, score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "artifacts/baseline-demo")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    predictor = Predictor(args.model_dir, args.device)
    rows = read_data(args.input, predictor.tokenizer, predictor.metadata["max_length"])
    predictions, repairs = [], 0
    with (args.output / "predictions.jsonl").open("w") as stream:
        for i, row in enumerate(rows):
            result = predictor.predict(row["query"], timing=True)
            predictions.append(result["segments"])
            repairs += result["bio_repairs"]
            missing = keys(row["spans"]) - keys(result["segments"])
            extra = keys(result["segments"]) - keys(row["spans"])
            result.update(
                gold=row["spans"], missing=sorted(missing), extra=sorted(extra)
            )
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            if (i + 1) % 100 == 0:
                print(f"Evaluated {i + 1}/{len(rows)}", flush=True)
    metrics = score([r["spans"] for r in rows], predictions)
    metrics.update(
        evaluation="synthetic",
        device=predictor.device,
        bio_repairs=repairs,
        queries=len(rows),
    )
    slices = {}
    for name, predicate in [
        ("raw", lambda r: any(s["label"] == "CONDITION" for s in r["spans"])),
        ("graded", lambda r: any(s["label"] == "GRADER" for s in r["spans"])),
        ("set", lambda r: any(s["label"] == "SET" for s in r["spans"])),
        ("number", lambda r: any(s["label"] == "CARD_NUMBER" for s in r["spans"])),
        ("tokens_le_16", lambda r: len(r["input_ids"]) <= 16),
        ("tokens_gt_16", lambda r: len(r["input_ids"]) > 16),
    ]:
        indices = [i for i, r in enumerate(rows) if predicate(r)]
        if indices:
            slices[name] = score(
                [rows[i]["spans"] for i in indices], [predictions[i] for i in indices]
            )
    metrics["slices"] = slices
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

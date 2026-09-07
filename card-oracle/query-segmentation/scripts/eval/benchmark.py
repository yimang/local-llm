# ruff: noqa: E402
"""Synchronized batch-one latency; model load is measured separately."""

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cardseg.inference import Predictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "artifacts/baseline-demo")
    parser.add_argument("--input", type=Path, default=ROOT / "data/synthetic/val.jsonl")
    parser.add_argument("--device", choices=["cpu", "mps", "auto"], default="auto")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 1 or args.warmups < 0:
        parser.error("samples must be positive and warmups nonnegative")
    if args.output.exists():
        parser.error("output already exists")
    queries = [
        json.loads(line)["query"] for line in args.input.read_text().splitlines()
    ]
    start = time.perf_counter()
    predictor = Predictor(args.model_dir, args.device)
    predictor.sync()
    load_seconds = time.perf_counter() - start
    for i in range(args.warmups):
        predictor.predict(queries[i % len(queries)])
    results = []
    for i in range(args.samples):
        results.append(predictor.predict(queries[i % len(queries)], timing=True))
    hardware = subprocess.run(
        ["sysctl", "-n", "machdep.cpu.brand_string", "hw.memsize"],
        capture_output=True,
        text=True,
    )
    report: dict = dict(
        device=predictor.device,
        load_seconds=load_seconds,
        samples=args.samples,
        warmups=args.warmups,
        platform=platform.platform(),
        hardware=hardware.stdout.strip(),
        precision="float32",
        versions=predictor.metadata["summary"]["versions"],
        downloads_excluded=True,
        latency_ms={},
    )
    for key in results[0]["timing_ms"]:
        values = [r["timing_ms"][key] for r in results]
        report["latency_ms"][key] = dict(
            zip(
                ["p50", "p95", "p99"],
                np.percentile(values, [50, 95, 99]).tolist(),
                strict=True,
            )
        )
    report["token_lengths"] = dict(
        min=min(r["tokens"] for r in results), max=max(r["tokens"] for r in results)
    )
    report["character_lengths"] = dict(
        min=min(len(r["query"]) for r in results),
        max=max(len(r["query"]) for r in results),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

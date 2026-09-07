# ruff: noqa: E402
"""Single-query, persistent interactive, or JSONL offline segmentation."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cardseg.inference import Predictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "artifacts/baseline-demo")
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--timing", action="store_true")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--query")
    modes.add_argument("--interactive", action="store_true")
    modes.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and not args.input:
        parser.error("--output requires --input")
    if args.input and args.output and args.input.resolve() == args.output.resolve():
        parser.error("input and output must differ")
    predictor = Predictor(args.model_dir, args.device)
    print(f"Loaded offline model on {predictor.device}", file=sys.stderr)
    output = args.output.open("x") if args.output else sys.stdout
    source = (
        args.input.open()
        if args.input
        else (sys.stdin if args.interactive else [args.query])
    )
    errors = 0
    try:
        for line, raw in enumerate(source, 1):
            record = {}
            try:
                record = json.loads(raw) if args.input else {"query": raw.rstrip("\n")}
                if not isinstance(record, dict):
                    raise ValueError("record must be a JSON object")
                result = predictor.predict(record["query"], args.timing)
                if "id" in record:
                    result["id"] = record["id"]
            except (ValueError, KeyError, TypeError) as exc:
                errors += 1
                result = dict(error=str(exc), line=line)
                if isinstance(record, dict) and "id" in record:
                    result["id"] = record["id"]
                print(json.dumps(result), file=sys.stderr)
                continue
            print(
                json.dumps(result, ensure_ascii=False, indent=None if args.input else 2),
                file=output,
                flush=True,
            )
    finally:
        if args.output:
            output.close()
        if args.input and not isinstance(source, list):
            source.close()
    return int(errors > 0)


if __name__ == "__main__":
    sys.exit(main())

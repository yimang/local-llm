# ruff: noqa: E402
"""Verify offline fresh-process reload against logits from the saving process."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cardseg.artifacts import load_artifact
from cardseg.data import collate, encode
from cardseg.decode import decode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "artifacts/baseline-demo")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--write-reference", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    model, tokenizer, metadata = load_artifact(args.model_dir)
    if args.write_reference:
        queries = [
            "roronoa zoro psa 10 op01 2022",
            "pikachu 2023 pal near mint",
            "2025 ogn ahri bgs 9.5",
        ]
        reference = []
        for query in queries:
            row = encode(tokenizer, query, max_length=metadata["max_length"])
            batch = collate([row], tokenizer.pad_token_id, "cpu")
            with torch.no_grad():
                logits = model(batch["input_ids"], batch["attention_mask"])[0].cpu()
            spans, _ = decode(query, row["offsets"], logits.argmax(-1).tolist())
            reference.append(dict(query=query, logits=logits.tolist(), spans=spans))
        args.reference.parent.mkdir(parents=True, exist_ok=True)
        args.reference.write_text(json.dumps(reference) + "\n")
        env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        subprocess.run(
            [
                sys.executable,
                __file__,
                "--model-dir",
                str(args.model_dir),
                "--reference",
                str(args.reference),
            ],
            check=True,
            env=env,
        )
        return
    reference = json.loads(args.reference.read_text())
    max_error = 0.0
    for item in reference:
        row = encode(tokenizer, item["query"], max_length=metadata["max_length"])
        batch = collate([row], tokenizer.pad_token_id, "cpu")
        with torch.no_grad():
            logits = model(batch["input_ids"], batch["attention_mask"])[0].cpu()
        expected = torch.tensor(item["logits"])
        torch.testing.assert_close(logits, expected, atol=1e-5, rtol=1e-5)
        assert torch.equal(logits.argmax(-1), expected.argmax(-1))
        spans, _ = decode(item["query"], row["offsets"], logits.argmax(-1).tolist())
        assert spans == item["spans"]
        max_error = max(max_error, float((logits - expected).abs().max()))
    print(
        json.dumps(
            dict(
                offline_reload_passed=True,
                queries=len(reference),
                max_logit_absolute_error=max_error,
                atol=1e-5,
                rtol=1e-5,
            )
        )
    )


if __name__ == "__main__":
    main()

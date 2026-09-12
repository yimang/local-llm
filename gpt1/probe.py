"""Run a fixed qualitative prompt suite through the checkpoint generator."""

import json
import subprocess
import sys
from pathlib import Path

PROMPTS = [
    ("encyclopedic style", "The river flows through the city and"),
    ("biography", "He was born in 1950 and began his career as a"),
    ("geography", "The capital of France is"),
    ("science", "Water freezes at a temperature of"),
    ("everyday cause and effect", "It started raining, so I opened my"),
    (
        "story continuity",
        "Alice put the red ball in the box. When she opened the box, she saw",
    ),
    ("arithmetic", "Two plus two equals"),
    ("pattern continuation", "One, two, three, four,"),
    ("definition", "A cat is a small"),
    ("negation", "The door was locked, so she could not"),
]


def main():
    results = []
    for category, prompt in PROMPTS:
        for temperature, seed in [(0.8, 42), (0.8, 7), (0, 42)]:
            output = subprocess.check_output(
                [
                    sys.executable,
                    "-m",
                    "gpt1.generate",
                    "--device",
                    "cpu",
                    "--prompt",
                    prompt,
                    "--max-new-tokens",
                    "35",
                    "--temperature",
                    str(temperature),
                    "--top-k",
                    "40",
                    "--seed",
                    str(seed),
                ],
                text=True,
            )
            record = {"category": category, **json.loads(output)}
            results.append(record)
            print(
                json.dumps(
                    {
                        k: record[k]
                        for k in (
                            "category",
                            "prompt",
                            "temperature",
                            "seed",
                            "completion",
                        )
                    }
                ),
                flush=True,
            )
    steps = {row["checkpoint_step"] for row in results}
    if len(steps) != 1:
        raise ValueError("Checkpoint changed during the probe suite")
    path = Path(f"gpt1/results/completion-probes-step-{steps.pop()}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()

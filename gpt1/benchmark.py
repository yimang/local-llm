"""Measure synchronized forward/backward/update throughput on the target hardware."""

import argparse
import json
import time
from pathlib import Path

import torch

from gpt1.model import GPT, ModelConfig
from gpt1.train import device_for, optimizer_for, synchronize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("gpt1/configs/small.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    settings = json.loads(args.config.read_text())
    config = ModelConfig(**settings["model"])
    device = device_for(args.device)
    results = []
    for batch in args.batches:
        torch.manual_seed(42)
        model = GPT(config).to(device).train()
        optimizer = optimizer_for(
            model, settings["learning_rate"], settings["weight_decay"]
        )
        x = torch.randint(config.vocab_size, (batch, config.context), device=device)
        y = torch.randint(config.vocab_size, (batch, config.context), device=device)
        durations = []
        for step in range(args.steps + 3):
            synchronize(device)
            tick = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            model(x, y)[1].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            synchronize(device)
            if step >= 3:
                durations.append(time.perf_counter() - tick)
        throughput = batch * config.context * len(durations) / sum(durations)
        record = {
            "device": device,
            "torch": torch.__version__,
            "microbatch": batch,
            "parameters": sum(p.numel() for p in model.parameters()),
            "tokens_per_second": throughput,
            "hours_100m_excluding_eval": 1e8 / throughput / 3600,
            "hours_300m_excluding_eval": 3e8 / throughput / 3600,
        }
        if device == "mps":
            record["current_allocated_bytes"] = torch.mps.current_allocated_memory()
            record["driver_allocated_bytes"] = torch.mps.driver_allocated_memory()
        elif device == "cuda":
            record["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        results.append(record)
        print(json.dumps(record), flush=True)
        del model, optimizer, x, y
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()

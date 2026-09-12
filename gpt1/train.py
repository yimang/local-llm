"""Pretrain a GPT-1 decoder; checkpoint model, optimizer, and random state."""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from gpt1.model import GPT, ModelConfig


def device_for(name):
    if name == "auto":
        return (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    return name


def synchronize(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()


class Tokens:
    def __init__(self, path, context):
        self.tokens = np.memmap(path, dtype="<u2", mode="r")
        self.context = context
        if len(self.tokens) <= context:
            raise ValueError("Dataset needs more tokens than the context length")

    def batch(self, batch_size, rng, device):
        starts = rng.integers(0, len(self.tokens) - self.context, size=batch_size)
        ids = np.stack([self.tokens[i : i + self.context + 1] for i in starts]).astype(
            np.int64
        )
        return torch.from_numpy(ids[:, :-1].copy()).to(device), torch.from_numpy(
            ids[:, 1:].copy()
        ).to(device)


def optimizer_for(model, learning_rate, weight_decay):
    return torch.optim.AdamW(
        [
            {
                "params": [p for p in model.parameters() if p.ndim >= 2],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for p in model.parameters() if p.ndim < 2],
                "weight_decay": 0.0,
            },
        ],
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
    )


def learning_rate(step, settings):
    warmup = settings["warmup_steps"]
    if step <= warmup:
        return settings["learning_rate"] * step / max(1, warmup)
    progress = (step - warmup) / max(1, settings["max_steps"] - warmup)
    return settings["learning_rate"] * 0.5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, data, settings, device):
    model.eval()
    rng = np.random.default_rng(12345)  # Fixed held-out windows across checkpoints.
    losses = []
    for _ in range(settings["eval_batches"]):
        x, y = data.batch(settings["microbatch"], rng, device)
        losses.append(model(x, y)[1].item())
    model.train()
    return sum(losses) / len(losses)


def save_checkpoint(path, model, optimizer, rng, step, settings, fingerprint, device):
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "settings": settings,
        "data_sha256": fingerprint,
        "numpy_rng": rng.bit_generator.state,
        "torch_rng": torch.get_rng_state(),
        "device": device,
    }
    if device == "mps":
        state["device_rng"] = torch.mps.get_rng_state()
    elif device.startswith("cuda"):
        state["device_rng"] = torch.cuda.get_rng_state_all()
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def run(settings, data_path, output, device="auto", resume=None, stop_after=None):
    device = device_for(device)
    if output.exists() and any(output.iterdir()) and resume is None:
        raise ValueError(
            "Nonempty run directory; use --resume or choose a new directory"
        )
    output.mkdir(parents=True, exist_ok=True)
    manifest_bytes = (data_path / "manifest.json").read_bytes()
    fingerprint = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = json.loads(manifest_bytes)
    if (data_path / "incomplete.json").exists():
        raise ValueError("Dataset preparation is incomplete")
    if settings["model"]["vocab_size"] != manifest["vocab_size"]:
        raise ValueError("Model vocabulary differs from the prepared tokenizer")
    if (
        min(
            settings[k]
            for k in (
                "microbatch",
                "accumulation",
                "max_steps",
                "eval_batches",
                "eval_every",
                "save_every",
                "log_every",
            )
        )
        < 1
    ):
        raise ValueError(
            "Batch sizes, steps, and logging/evaluation/checkpoint intervals must be positive"
        )
    if stop_after is not None and stop_after < 1:
        raise ValueError("Stopping step must be positive")
    torch.manual_seed(settings["seed"])
    rng = np.random.default_rng(settings["seed"])
    model = GPT(ModelConfig(**settings["model"])).to(device)
    optimizer = optimizer_for(
        model, settings["learning_rate"], settings["weight_decay"]
    )
    step = 0
    if resume:
        # Only load checkpoints produced locally by this experiment.
        state = torch.load(resume, map_location="cpu", weights_only=False)
        if state["settings"] != settings or state["data_sha256"] != fingerprint:
            raise ValueError(
                "Resume requires identical configuration and dataset manifest"
            )
        if state["device"] != device:
            raise ValueError(
                "Resume requires the same device backend for random-state restoration"
            )
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        rng.bit_generator.state = state["numpy_rng"]
        torch.set_rng_state(state["torch_rng"])
        if device == "mps":
            torch.mps.set_rng_state(state["device_rng"])
        elif device.startswith("cuda"):
            torch.cuda.set_rng_state_all(state["device_rng"])
        step = state["step"]
    end_step = min(settings["max_steps"], stop_after or settings["max_steps"])
    if end_step <= step:
        raise ValueError("Stopping step must be after the checkpoint step")
    context = model.config.context
    training = Tokens(data_path / "train.bin", context)
    validation = Tokens(data_path / "validation.bin", context)
    metadata = {
        "settings": settings,
        "device": device,
        "torch": torch.__version__,
        "parameters": sum(p.numel() for p in model.parameters()),
        "data_sha256": fingerprint,
        "data_path": str(data_path.resolve()),
    }
    (output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata), flush=True)
    tokens_per_step = settings["microbatch"] * settings["accumulation"] * context
    start_step = step
    started = time.perf_counter()
    model.train()
    with (output / "metrics.jsonl").open("a") as log:
        while step < end_step:
            synchronize(device)
            tick = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            lr = learning_rate(step + 1, settings)
            for group in optimizer.param_groups:
                group["lr"] = lr
            total_loss = 0.0
            for _ in range(settings["accumulation"]):
                x, y = training.batch(settings["microbatch"], rng, device)
                _, loss = model(x, y)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss")
                total_loss += loss.detach().item() / settings["accumulation"]
                (loss / settings["accumulation"]).backward()
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            synchronize(device)
            elapsed = time.perf_counter() - tick
            step += 1
            record = {
                "step": step,
                "tokens_processed": step * tokens_per_step,
                "train_loss": total_loss,
                "learning_rate": lr,
                "grad_norm": float(norm),
                "tokens_per_second": tokens_per_step / elapsed,
            }
            if step % settings["eval_every"] == 0 or step == end_step:
                val_loss = evaluate(model, validation, settings, device)
                record.update(
                    validation_loss=val_loss, perplexity=math.exp(min(val_loss, 50))
                )
            log.write(json.dumps(record) + "\n")
            log.flush()
            if step % settings["log_every"] == 0 or "validation_loss" in record:
                print(json.dumps(record), flush=True)
            if step % settings["save_every"] == 0 or step == end_step:
                save_checkpoint(
                    output / "latest.pt",
                    model,
                    optimizer,
                    rng,
                    step,
                    settings,
                    fingerprint,
                    device,
                )
        summary = {
            "last_step": step,
            "tokens_this_session": (step - start_step) * tokens_per_step,
            "wall_seconds": time.perf_counter() - started,
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("gpt1/configs/small.json"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--device", choices=["auto", "cpu", "mps", "cuda"], default="auto"
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--stop-after", type=int, help="Stop at this absolute step; preserve LR horizon"
    )
    args = parser.parse_args()
    run(
        json.loads(args.config.read_text()),
        args.data,
        args.output,
        args.device,
        args.resume,
        args.stop_after,
    )


if __name__ == "__main__":
    main()

# ruff: noqa: E402
"""Train a reproducible frozen-encoder baseline; test data is never read."""

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HOME", str(ROOT / "data/cache/huggingface"))

import torch
import yaml
from cardseg.artifacts import save_artifact
from cardseg.data import collate, read_data
from cardseg.decode import decode
from cardseg.metrics import score
from cardseg.model import Segmenter, choose_device
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract(model, tokenizer, rows, batch_size, device, label):
    features = []
    start = time.perf_counter()
    for i in range(0, len(rows), batch_size):
        chunk = rows[i : i + batch_size]
        batch = collate(chunk, tokenizer.pad_token_id, device)
        hidden = model.features(batch["input_ids"], batch["attention_mask"]).cpu()
        features.extend(
            hidden[j, : len(row["input_ids"])].clone() for j, row in enumerate(chunk)
        )
        if i % (batch_size * 25) == 0:
            print(
                f"{label}: {i + len(chunk)}/{len(rows)} ({time.perf_counter() - start:.1f}s)",
                flush=True,
            )
    return features


def evaluate_head(head, rows, features, batch_size):
    predictions, repairs = [], 0
    head.eval()
    with torch.no_grad():
        for i in range(0, len(rows), batch_size):
            logits = head(pad_sequence(features[i : i + batch_size], batch_first=True))
            for row, pred in zip(
                rows[i : i + batch_size], logits.argmax(-1).tolist(), strict=True
            ):
                spans, fixed = decode(
                    row["query"], row["offsets"], pred[: len(row["offsets"])]
                )
                predictions.append(spans)
                repairs += fixed
    metrics = score([r["spans"] for r in rows], predictions)
    metrics["bio_repairs"] = repairs
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/train.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/baseline")
    parser.add_argument(
        "--tiny",
        type=int,
        help="Intentional overfit on N training examples; validation is the same subset",
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"])
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"{args.output} already exists; choose a new output")
    config = yaml.safe_load(args.config.read_text())
    for key in ("epochs", "device"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if args.tiny:
        config["patience"] = config["epochs"]
    if any(
        config[k] <= 0
        for k in ("epochs", "batch_size", "max_length", "patience", "learning_rate")
    ):
        parser.error(
            "epochs, batch size, length, patience, and learning rate must be positive"
        )
    if args.tiny is not None and args.tiny <= 0:
        parser.error("--tiny must be positive")
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.set_num_threads(4)
    device = choose_device(config["device"])
    print(
        f"Encoder device: {device}; head optimization: cpu; float32; attention: sdpa",
        flush=True,
    )
    if config["device"] == "auto" and device == "cpu":
        print("Auto device: MPS unavailable, using CPU", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"], revision=config.get("revision")
    )
    train_path, val_path = [
        ROOT / f"data/synthetic/{s}.jsonl" for s in ("train", "val")
    ]
    train = read_data(train_path, tokenizer, config["max_length"])
    val = (
        read_data(val_path, tokenizer, config["max_length"])
        if not args.tiny
        else train[: args.tiny]
    )
    if args.tiny:
        train = train[: args.tiny]
    model = Segmenter(
        config["model"],
        head_hidden=config.get("head_hidden", 0),
        revision=config.get("revision"),
    ).to(device)
    frozen_before = hashlib.sha256(
        b"".join(p.detach().cpu().numpy().tobytes() for p in model.encoder.parameters())
    ).hexdigest()
    head_before = copy.deepcopy(model.head.state_dict())
    estimated = (
        sum(len(r["input_ids"]) for r in train + val)
        * model.encoder.config.hidden_size
        * 4
    )
    print(
        f"All annotations aligned; RAM feature cache estimate {estimated / 2**20:.1f} MiB. No persistent cache reuse.",
        flush=True,
    )
    start = time.perf_counter()
    train_features = extract(
        model, tokenizer, train, config["batch_size"], device, "train features"
    )
    val_features = (
        extract(
            model, tokenizer, val, config["batch_size"], device, "validation features"
        )
        if not args.tiny
        else train_features
    )
    model.head.cpu()
    optimizer = torch.optim.AdamW(
        model.head.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    best_f1, stale, best_state, history = -1, 0, None, []
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        order = torch.randperm(len(train)).tolist()
        loss_total, count = 0.0, 0
        for i in range(0, len(order), config["batch_size"]):
            indices = order[i : i + config["batch_size"]]
            x = pad_sequence([train_features[j] for j in indices], batch_first=True)
            y = pad_sequence(
                [torch.tensor(train[j]["labels"]) for j in indices],
                batch_first=True,
                padding_value=-100,
            )
            logits = model.head(x)
            loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tokens = int((y != -100).sum())
            loss_total += loss.item() * tokens
            count += tokens
        metrics = evaluate_head(model.head, val, val_features, config["batch_size"])
        record = dict(epoch=epoch, loss=loss_total / count, validation=metrics)
        history.append(record)
        print(
            json.dumps(
                dict(
                    epoch=epoch,
                    loss=record["loss"],
                    f1=metrics["f1"],
                    query_exact_match=metrics["query_exact_match"],
                )
            ),
            flush=True,
        )
        if metrics["f1"] > best_f1:
            best_f1, stale = metrics["f1"], 0
            best_state = copy.deepcopy(model.head.state_dict())
            best_epoch = epoch
        else:
            stale += 1
        if stale >= config["patience"]:
            break
    assert best_state is not None
    model.head.load_state_dict(best_state)
    frozen_after = hashlib.sha256(
        b"".join(p.detach().cpu().numpy().tobytes() for p in model.encoder.parameters())
    ).hexdigest()
    assert frozen_before == frozen_after, "Encoder changed during head-only training"
    assert any(
        not torch.equal(head_before[k].cpu(), v) for k, v in best_state.items()
    ), "Head did not update"
    summary = dict(
        config=config,
        best_epoch=best_epoch,
        best_validation=history[best_epoch - 1]["validation"],
        history=history,
        encoder_device=device,
        head_device="cpu",
        precision="float32",
        attention="sdpa",
        model_revision=model.encoder.config._commit_hash,
        dataset_hashes={"train": digest(train_path), "val": digest(val_path)},
        versions={
            p: importlib.metadata.version(p)
            for p in ["torch", "transformers", "tokenizers", "safetensors"]
        },
        python=platform.python_version(),
        platform=platform.platform(),
        elapsed_seconds=time.perf_counter() - start,
        tiny_examples=args.tiny,
        encoder_unchanged=True,
        encoder_sha256=frozen_after,
        determinism="Seeded Python/PyTorch; hardware-dependent numerical nondeterminism possible",
    )
    save_artifact(args.output, model, tokenizer, summary)
    print(f"Saved {args.output}; best epoch {best_epoch}; F1 {best_f1:.6f}", flush=True)


if __name__ == "__main__":
    main()

# ruff: noqa: E402
"""Adapt the final encoder layers after selecting a frozen-head baseline."""

import argparse
import copy
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cardseg.artifacts import load_artifact, save_artifact
from cardseg.data import collate, read_data
from cardseg.metrics import score
from cardseg.model import choose_device
from cardseg.training import (
    decode_batch,
    fingerprint,
    validate_config,
    write_reload_reference,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/train-tail.yaml")
    parser.add_argument("--source", type=Path, default=ROOT / "artifacts/baseline-mlp")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/baseline-tail")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    config = yaml.safe_load(args.config.read_text())
    try:
        validate_config(config)
    except ValueError as exc:
        parser.error(str(exc))
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.set_num_threads(4)
    device = choose_device(config["device"])
    model, tokenizer, metadata = load_artifact(args.source, device)
    layers = config["trainable_layers"]
    if not 1 <= layers <= len(model.encoder.layers):
        parser.error("trainable_layers must be between 1 and the encoder layer count")
    if config["head_hidden"] != metadata["summary"]["config"].get("head_hidden", 0):
        parser.error("head architecture must match the source checkpoint")
    model.trainable_layers = layers
    model.encoder.layers[-layers:].requires_grad_(True)
    model.encoder.final_norm.requires_grad_(True)
    trainable_names = {name for name, p in model.named_parameters() if p.requires_grad}
    frozen = [p for p in model.encoder.parameters() if not p.requires_grad]
    adaptable = [p for p in model.encoder.parameters() if p.requires_grad]
    before = fingerprint(frozen)
    adaptable_before = fingerprint(adaptable)
    train = read_data(
        ROOT / "data/synthetic/train.jsonl", tokenizer, config["max_length"]
    )
    val = read_data(ROOT / "data/synthetic/val.jsonl", tokenizer, config["max_length"])
    optimizer = torch.optim.AdamW(
        [
            {"params": adaptable, "lr": config["learning_rate"]},
            {"params": model.head.parameters(), "lr": config["head_learning_rate"]},
        ],
        weight_decay=config["weight_decay"],
    )
    history, best, stale, best_state = [], -1, 0, None
    start = time.perf_counter()
    print(
        f"Fine-tuning last {layers} layers + final norm + head on {device}; no feature cache",
        flush=True,
    )
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        order = torch.randperm(len(train)).tolist()
        loss_total, count = 0.0, 0
        for i in range(0, len(order), config["batch_size"]):
            rows = [train[j] for j in order[i : i + config["batch_size"]]]
            batch = collate(rows, tokenizer.pad_token_id, device)
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = torch.nn.functional.cross_entropy(
                logits.flatten(0, 1), batch["labels"].flatten()
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([*adaptable, *model.head.parameters()], 1.0)
            optimizer.step()
            tokens = int((batch["labels"] != -100).sum())
            loss_total += loss.item() * tokens
            count += tokens
            if i % 1600 == 0:
                print(
                    f"epoch {epoch}: {i + len(rows)}/{len(train)} loss {loss_total / count:.5f}",
                    flush=True,
                )
        model.eval()
        predictions = []
        with torch.no_grad():
            for i in range(0, len(val), config["batch_size"]):
                rows = val[i : i + config["batch_size"]]
                batch = collate(rows, tokenizer.pad_token_id, device)
                logits = model(batch["input_ids"], batch["attention_mask"])
                spans, _ = decode_batch(rows, logits)
                predictions.extend(spans)
        metrics = score([r["spans"] for r in val], predictions)
        history.append(dict(epoch=epoch, loss=loss_total / count, validation=metrics))
        print(
            json.dumps(
                dict(
                    epoch=epoch,
                    loss=loss_total / count,
                    f1=metrics["f1"],
                    exact=metrics["query_exact_match"],
                )
            ),
            flush=True,
        )
        if metrics["f1"] > best:
            best, stale, best_epoch = metrics["f1"], 0, epoch
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
                if k in trainable_names
            }
        else:
            stale += 1
        if stale >= config["patience"]:
            break
    assert best_state is not None
    model.load_state_dict(best_state, strict=False)
    assert fingerprint(frozen) == before
    assert fingerprint(adaptable) != adaptable_before
    summary = copy.deepcopy(metadata["summary"])
    summary.update(
        config=config,
        history=history,
        best_epoch=best_epoch,
        best_validation=history[best_epoch - 1]["validation"],
        encoder_device=device,
        head_device=device,
        encoder_unchanged=False,
        frozen_layers_unchanged=True,
        elapsed_seconds=time.perf_counter() - start,
        initialization=str(args.source.resolve()),
        tiny_examples=None,
    )
    summary["dataset_hashes"] = {
        split: hashlib.sha256(
            (ROOT / f"data/synthetic/{split}.jsonl").read_bytes()
        ).hexdigest()
        for split in ("train", "val")
    }
    summary.pop("encoder_sha256", None)
    save_artifact(args.output, model, tokenizer, summary)
    write_reload_reference(
        args.output / "reload-reference.json", model, tokenizer, val[:3]
    )
    print(
        f"Saved {args.output}; best epoch {best_epoch}; validation F1 {best:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()

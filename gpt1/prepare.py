"""Stream a pinned Wikipedia snapshot into article-disjoint token datasets."""

import argparse
import hashlib
import itertools
import json
import re
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from datasets import load_dataset
from huggingface_hub import HfApi
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

SPECIAL = ["<|endoftext|>", "<|start|>", "<|sep|>", "<|classify|>", "<|pad|>"]


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_for(article_id, seed):
    bucket = int(digest(f"{seed}:{article_id}")[:16], 16) % 10000
    return "validation" if bucket < 100 else "test" if bucket < 200 else "train"


def clean(text):
    return "\n\n".join(
        line
        for part in text.splitlines()
        if (line := re.sub(r"[ \t]+", " ", part).strip())
    )


def accepted_rows(rows, seed, min_chars):
    seen_ids, seen_text = set(), set()
    for row in rows:
        article_id = str(row["id"])
        text = clean(row["text"])
        # Whitespace-insensitive exact deduplication; this is not near-deduplication.
        text_hash = digest(" ".join(text.split()))
        if len(text) < min_chars or article_id in seen_ids or text_hash in seen_text:
            continue
        seen_ids.add(article_id)
        seen_text.add(text_hash)
        yield {
            "id": article_id,
            "url": row.get("url", ""),
            "title": row.get("title", ""),
            "text": text,
            "text_sha256": text_hash,
            "split": split_for(article_id, seed),
        }


def make_tokenizer(texts, vocab_size):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(
        texts,
        trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=2,
            special_tokens=SPECIAL,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=True,
        ),
    )
    return tokenizer


def prepare(
    rows,
    output,
    targets,
    seed=42,
    vocab_size=16000,
    tokenizer_chars=10000000,
    min_chars=200,
    source=None,
):
    if not 261 <= vocab_size <= 65536 or min(targets.values()) <= 0:
        raise ValueError("Need vocabulary 261..65536 and positive token targets")
    if set(targets) != {"train", "validation", "test"}:
        raise ValueError("Token targets must include train, validation, and test")
    if tokenizer_chars < 1 or min_chars < 1:
        raise ValueError(
            "Tokenizer character target and minimum article length must be positive"
        )
    output.mkdir(parents=True, exist_ok=False)
    settings = {
        "source": source,
        "seed": seed,
        "targets": targets,
        "requested_vocab_size": vocab_size,
        "tokenizer_chars_target": tokenizer_chars,
        "min_chars": min_chars,
        "dedup": "whitespace-normalized exact text and article id",
        "split": "sha256(seed:article_id) modulo 10000; validation <100, test <200",
        "dtype": "uint16",
        "byte_order": "little",
        "document_boundary": SPECIAL[0],
    }
    (output / "incomplete.json").write_text(json.dumps(settings, indent=2) + "\n")
    accepted = iter(accepted_rows(rows, seed, min_chars))
    pool, train_chars = [], 0
    for row in accepted:
        pool.append(row)
        if row["split"] == "train":
            train_chars += len(row["text"])
        if train_chars >= tokenizer_chars:
            break
    if not train_chars:
        raise ValueError("No training articles available for the tokenizer")
    tokenizer = make_tokenizer(
        (row["text"] for row in pool if row["split"] == "train"), vocab_size
    )
    tokenizer.save(str(output / "tokenizer.json"))
    eos = tokenizer.token_to_id(SPECIAL[0])
    counts = {split: {"tokens": 0, "articles": 0, "text_bytes": 0} for split in targets}
    with ExitStack() as stack:
        binaries = {
            s: stack.enter_context((output / f"{s}.bin").open("wb")) for s in targets
        }
        documents = {
            s: stack.enter_context((output / f"{s}.jsonl").open("w")) for s in targets
        }
        for row in itertools.chain(pool, accepted):
            split = row["split"]
            count = counts[split]
            if count["tokens"] >= targets[split]:
                continue
            ids = [*tokenizer.encode(row["text"]).ids, eos]
            # Retain whole articles: targets may overshoot by one article per split.
            np.asarray(ids, dtype="<u2").tofile(binaries[split])
            row["token_offset"] = count["tokens"]
            row["token_count"] = len(ids)
            documents[split].write(json.dumps(row, ensure_ascii=False) + "\n")
            count["tokens"] += len(ids)
            count["articles"] += 1
            count["text_bytes"] += len(row["text"].encode("utf-8"))
            if sum(c["articles"] for c in counts.values()) % 5000 == 0:
                print(json.dumps(counts), flush=True)
            if all(counts[s]["tokens"] >= targets[s] for s in targets):
                break
    if any(counts[s]["tokens"] < targets[s] for s in targets):
        raise ValueError(f"Source exhausted before token targets: {counts}")
    manifest = {
        **settings,
        "counts": counts,
        "vocab_size": tokenizer.get_vocab_size(),
        "tokenizer_training_chars": train_chars,
        "tokenizer_sha256": digest((output / "tokenizer.json").read_text()),
        "files": {},
    }
    for path in sorted(output.iterdir()):
        if path.suffix in {".bin", ".jsonl"}:
            with path.open("rb") as stream:
                manifest["files"][path.name] = hashlib.file_digest(
                    stream, "sha256"
                ).hexdigest()
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "incomplete.json").unlink()
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-tokens", type=int, default=100000000)
    parser.add_argument("--heldout-tokens", type=int, default=1000000)
    parser.add_argument("--tokenizer-chars", type=int, default=10000000)
    parser.add_argument("--vocab-size", type=int, default=16000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--snapshot", default="20231101.en")
    parser.add_argument("--shuffle-buffer", type=int, default=1000)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(
            "Output already exists; choose a new directory to protect existing data"
        )
    repo = "wikimedia/wikipedia"
    revision = HfApi().dataset_info(repo, revision=args.revision).sha
    print(f"Streaming {repo}/{args.snapshot} at {revision}", flush=True)
    rows = load_dataset(
        repo, args.snapshot, revision=revision, split="train", streaming=True
    )
    rows = rows.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
    prepare(
        rows,
        args.output,
        {
            "train": args.train_tokens,
            "validation": args.heldout_tokens,
            "test": args.heldout_tokens,
        },
        seed=args.seed,
        vocab_size=args.vocab_size,
        tokenizer_chars=args.tokenizer_chars,
        source={
            "repo": repo,
            "snapshot": args.snapshot,
            "revision": revision,
            "shuffle_buffer": args.shuffle_buffer,
            "url": "https://huggingface.co/datasets/wikimedia/wikipedia",
            "license": "See source dataset card and Wikipedia attribution/share-alike terms",
        },
    )


if __name__ == "__main__":
    main()

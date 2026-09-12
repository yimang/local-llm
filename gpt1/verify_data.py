"""Verify file hashes, split isolation, token ranges, and document boundaries."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from gpt1.prepare import digest, split_for


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(path):
    manifest = json.loads((path / "manifest.json").read_text())
    if (path / "incomplete.json").exists():
        raise ValueError("Incomplete dataset")
    require(
        digest((path / "tokenizer.json").read_text()) == manifest["tokenizer_sha256"],
        "Tokenizer checksum mismatch",
    )
    for name, expected in manifest["files"].items():
        with (path / name).open("rb") as stream:
            require(
                hashlib.file_digest(stream, "sha256").hexdigest() == expected,
                f"File checksum mismatch: {name}",
            )
    tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
    eos = tokenizer.token_to_id("<|endoftext|>")
    ids, hashes = set(), set()
    for split, count in manifest["counts"].items():
        tokens = np.memmap(path / f"{split}.bin", dtype="<u2", mode="r")
        require(len(tokens) == count["tokens"], f"Token count mismatch: {split}")
        require(
            int(tokens.max()) < manifest["vocab_size"], f"Invalid token ID: {split}"
        )
        offset, articles, text_bytes = 0, 0, 0
        with (path / f"{split}.jsonl").open() as stream:
            for line in stream:
                row = json.loads(line)
                require(row["id"] not in ids, "Duplicate article ID")
                require(row["text_sha256"] not in hashes, "Duplicate article text")
                require(
                    digest(" ".join(row["text"].split())) == row["text_sha256"],
                    "Article text checksum mismatch",
                )
                require(
                    split_for(row["id"], manifest["seed"]) == split,
                    "Article split mismatch",
                )
                require(row["token_offset"] == offset, "Article offset mismatch")
                require(row["token_count"] > 0, "Empty article token sequence")
                offset += row["token_count"]
                require(offset <= len(tokens), "Article exceeds token file")
                require(tokens[offset - 1] == eos, "Missing document boundary")
                ids.add(row["id"])
                hashes.add(row["text_sha256"])
                articles += 1
                text_bytes += len(row["text"].encode("utf-8"))
        require(offset == count["tokens"], f"Document token count mismatch: {split}")
        require(articles == count["articles"], f"Article count mismatch: {split}")
        require(text_bytes == count["text_bytes"], f"Text byte count mismatch: {split}")
    print(json.dumps({"verified": True, "counts": manifest["counts"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    verify(parser.parse_args().data)

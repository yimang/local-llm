"""Self-contained, offline inference artifacts."""

import json
from pathlib import Path

from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from cardseg.data import LABELS
from cardseg.model import Segmenter


def save_artifact(path, model, tokenizer, summary):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    model.encoder.save_pretrained(path / "encoder", safe_serialization=True)
    tokenizer.save_pretrained(path / "tokenizer")
    save_file(
        {k: v.detach().cpu().contiguous() for k, v in model.head.state_dict().items()},
        str(path / "head.safetensors"),
    )
    metadata = dict(
        schema_version=1,
        labels=LABELS,
        max_length=summary["config"]["max_length"],
        summary=summary,
    )
    (path / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def load_artifact(path, device="cpu"):
    path = Path(path)
    metadata = json.loads((path / "metadata.json").read_text())
    if metadata["schema_version"] != 1 or metadata["labels"] != LABELS:
        raise ValueError("unsupported checkpoint schema or label mapping")
    tokenizer = AutoTokenizer.from_pretrained(path / "tokenizer", local_files_only=True)
    model = Segmenter(
        path / "encoder",
        local=True,
        head_hidden=metadata["summary"]["config"].get("head_hidden", 0),
    )
    model.head.load_state_dict(load_file(str(path / "head.safetensors")))
    return model.to(device).eval(), tokenizer, metadata

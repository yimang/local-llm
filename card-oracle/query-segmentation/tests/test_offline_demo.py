"""Integration checks when a local demo artifact is available; no downloads."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "artifacts/baseline-demo"
pytestmark = pytest.mark.skipif(
    not MODEL.exists(), reason="local trained artifact not present"
)


def run_cli(*args):
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/infer/segment.py"),
            "--model-dir",
            str(MODEL),
            "--device",
            "cpu",
            *args,
        ],
        capture_output=True,
        text=True,
        env=dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1"),
    )


def test_offline_interactive():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/infer/segment.py"),
            "--model-dir",
            str(MODEL),
            "--device",
            "cpu",
            "--interactive",
        ],
        input="roronoa zoro psa 10 op01 2022\n   \n",
        capture_output=True,
        text=True,
        env=dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1"),
    )
    assert result.returncode == 0, result.stderr
    rows = []
    remaining = result.stdout.strip()
    decoder = json.JSONDecoder()
    while remaining:
        row, end = decoder.raw_decode(remaining)
        rows.append(row)
        remaining = remaining[end:].lstrip()
    assert len(rows) == 2
    assert rows[1]["segments"] == []
    assert rows[0]["segments"]
    for span in rows[0]["segments"]:
        assert span["text"] == rows[0]["query"][span["start"] : span["end"]]


def test_batch_errors_continue(tmp_path):
    source, output = tmp_path / "input.jsonl", tmp_path / "output.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps({"id": "ok", "query": "pikachu psa 10"}),
                "{bad json",
                json.dumps({"id": "empty", "query": ""}),
                json.dumps({"id": "long", "query": "pikachu " * 200}),
            ]
        )
    )
    result = run_cli("--input", str(source), "--output", str(output))
    assert result.returncode == 1
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [r["id"] for r in rows] == ["ok", "empty"]
    assert '"line": 2' in result.stderr
    assert '"id": "long"' in result.stderr

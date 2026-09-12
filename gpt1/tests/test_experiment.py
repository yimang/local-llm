import json

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer

from gpt1.model import GPT, ModelConfig
from gpt1.prepare import accepted_rows, prepare, split_for
from gpt1.train import Tokens, run
from gpt1.verify_data import verify


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_causality_and_shift():
    torch.manual_seed(3)
    model = GPT(
        ModelConfig(vocab_size=32, context=8, layers=2, width=16, heads=2, dropout=0)
    )
    ids = torch.randint(32, (2, 8))
    changed = ids.clone()
    changed[:, 5:] = (changed[:, 5:] + 7) % 32
    torch.testing.assert_close(model(ids)[0][:, :5], model(changed)[0][:, :5])
    assert not torch.equal(model(ids)[0][:, 5:], model(changed)[0][:, 5:])
    target = torch.randint(32, (2, 8))
    target[:, -2:] = -100
    logits, loss = model(ids, target)
    expected = torch.nn.functional.cross_entropy(
        logits[:, :-2].reshape(-1, 32), target[:, :-2].reshape(-1)
    )
    torch.testing.assert_close(loss, expected)


def test_overfit():
    torch.manual_seed(4)
    model = GPT(
        ModelConfig(vocab_size=32, context=8, layers=1, width=32, heads=2, dropout=0)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    x = torch.arange(8).unsqueeze(0)
    y = x + 1
    initial = model(x, y)[1].item()
    for _ in range(70):
        optimizer.zero_grad()
        loss = model(x, y)[1]
        loss.backward()
        optimizer.step()
    assert model(x, y)[1].item() < initial * 0.05


def test_deduplication():
    rows = [
        {"id": "a", "text": "one two three"},
        {"id": "b", "text": "one  two\nthree"},
        {"id": "a", "text": "another article"},
        {"id": "c", "text": "four five six"},
    ]
    assert [r["id"] for r in accepted_rows(rows, 42, 1)] == ["a", "c"]


def test_preparation(tmp_path):
    rows = []
    for split in ("train", "validation", "test"):
        article_id = next(
            str(i) for i in range(10000) if split_for(str(i), 42) == split
        )
        rows.append({"id": article_id, "text": f"{split} article. " * 100})
    output = tmp_path / "data"
    manifest = prepare(
        rows,
        output,
        dict.fromkeys(["train", "validation", "test"], 100),
        vocab_size=300,
        tokenizer_chars=100,
    )
    tokenizer = Tokenizer.from_file(str(output / "tokenizer.json"))
    seen = set()
    for split in manifest["counts"]:
        documents = [
            json.loads(line)
            for line in (output / f"{split}.jsonl").read_text().splitlines()
        ]
        tokens = np.fromfile(output / f"{split}.bin", dtype="<u2")
        assert len(tokens) == manifest["counts"][split]["tokens"]
        assert tokens[-1] == tokenizer.token_to_id("<|endoftext|>")
        for document in documents:
            assert document["id"] not in seen
            seen.add(document["id"])
            assert split_for(document["id"], 42) == split
    x, y = Tokens(output / "train.bin", 8).batch(2, np.random.default_rng(3), "cpu")
    torch.testing.assert_close(x[:, 1:], y[:, :-1])
    verify(output)
    with (output / "train.bin").open("r+b") as stream:
        stream.write(b"\xff\xff")
    with pytest.raises(ValueError, match=r"File checksum mismatch: train\.bin"):
        verify(output)


def test_resume_equivalence(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "manifest.json").write_text(json.dumps({"vocab_size": 32}))
    for split in ("train", "validation"):
        (np.arange(400) % 32).astype("<u2").tofile(data / f"{split}.bin")
    settings = {
        "model": {
            "vocab_size": 32,
            "context": 8,
            "layers": 1,
            "width": 16,
            "heads": 2,
            "dropout": 0.1,
        },
        "seed": 42,
        "microbatch": 2,
        "accumulation": 2,
        "max_steps": 4,
        "warmup_steps": 1,
        "learning_rate": 0.01,
        "weight_decay": 0.01,
        "eval_every": 2,
        "eval_batches": 2,
        "save_every": 2,
        "log_every": 1,
    }
    uninterrupted = run(settings, data, tmp_path / "full", "cpu")
    run(settings, data, tmp_path / "resumed", "cpu", stop_after=2)
    resumed = run(
        settings,
        data,
        tmp_path / "resumed",
        "cpu",
        resume=tmp_path / "resumed/latest.pt",
    )
    for key, value in uninterrupted.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[key], rtol=0, atol=0)
    with pytest.raises(ValueError, match="Stopping step must be after"):
        run(
            settings,
            data,
            tmp_path / "resumed",
            "cpu",
            resume=tmp_path / "resumed/latest.pt",
            stop_after=2,
        )

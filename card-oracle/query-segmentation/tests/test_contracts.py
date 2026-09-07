import json

import pytest
import torch
from cardseg.data import LABELS, collate, encode, read_data
from cardseg.decode import decode
from cardseg.metrics import score


def label_logits(ids):
    return torch.nn.functional.one_hot(torch.tensor(ids), len(LABELS)).float()


class Tokenizer:
    def __init__(self, offsets):
        self.offsets = offsets

    def __call__(self, query, **kwargs):
        return dict(
            input_ids=list(range(len(self.offsets))),
            attention_mask=[1] * len(self.offsets),
            offset_mapping=self.offsets,
        )


def test_unicode_repeated_subwords_and_padding():
    query = "éon éon"
    spans = [
        dict(start=0, end=3, label="SUBJECT"),
        dict(start=4, end=7, label="SUBJECT"),
    ]
    tokenizer = Tokenizer([(0, 0), (0, 1), (1, 3), (3, 5), (5, 7), (0, 0)])
    row = encode(tokenizer, query, spans)
    assert row["labels"] == [-100, 1, 2, 1, 2, -100]
    decoded, repairs = decode(
        query, row["offsets"], label_logits([max(0, t) for t in row["labels"]])
    )
    assert [{k: s[k] for k in ("start", "end", "label")} for s in decoded] == spans
    assert repairs == 0
    short = encode(Tokenizer([(0, 0), (0, 1), (0, 0)]), "x")
    batch = collate([row, short], 99, "cpu")
    assert batch["labels"][1, 3:].tolist() == [-100] * 3
    assert batch["attention_mask"][1, 3:].tolist() == [0] * 3


@pytest.mark.parametrize(
    "spans",
    [
        [dict(start=-1, end=2, label="SUBJECT")],
        [dict(start=0, end=4, label="SUBJECT")],
        [dict(start=0, end=1, label="NOPE")],
        [dict(start=0, end=2, label="SUBJECT"), dict(start=1, end=3, label="GAME")],
    ],
)
def test_invalid_spans(spans):
    with pytest.raises(ValueError):
        encode(Tokenizer([(0, 3)]), "abc", spans)


def test_crossing_and_overlength():
    with pytest.raises(ValueError, match="crosses"):
        encode(Tokenizer([(0, 3)]), "abc", [dict(start=1, end=3, label="SUBJECT")])
    with pytest.raises(ValueError, match="exceeds"):
        encode(Tokenizer([(0, 1)] * 5), "a", max_length=4)


def test_repair_and_exact_metrics():
    spans, repairs = decode(
        "abc def",
        [(0, 3), (4, 7)],
        label_logits([LABELS.index("I-SUBJECT"), LABELS.index("I-GAME")]),
    )
    assert repairs == 2
    metrics = score([[dict(start=0, end=3, label="SUBJECT")]], [spans])
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 1
    assert metrics["f1"] == pytest.approx(2 / 3)
    assert metrics["query_exact_match"] == 0


def test_artifact_schema_rejected_before_loading_weights(tmp_path):
    import json

    from cardseg.artifacts import load_artifact

    (tmp_path / "metadata.json").write_text(
        json.dumps({"schema_version": 1, "labels": ["wrong"]})
    )
    with pytest.raises(ValueError, match="schema or label"):
        load_artifact(tmp_path)


def test_artifact_overwrite_guard(tmp_path):
    from cardseg.artifacts import save_artifact

    with pytest.raises(FileExistsError):
        save_artifact(tmp_path, None, None, {})


def test_whitespace_skips_tokenizer_and_model():
    from cardseg.inference import Predictor

    predictor = object.__new__(Predictor)
    predictor.device = "cpu"
    # No tokenizer/model attributes: calling either would fail this test.
    result = predictor.predict(" " * 10000, timing=True)
    assert result["segments"] == []
    assert result["timing_ms"]["forward"] == 0


@pytest.mark.parametrize("annotations", [None, {}, "SUBJECT", 1])
def test_dataset_rejects_non_list_annotations(tmp_path, annotations):
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps({"query": "abc", "spans": annotations}) + "\n")
    with pytest.raises(ValueError, match=r"train.jsonl:1: spans must be a list"):
        read_data(path, Tokenizer([(0, 3)]))


def test_explicit_negative_and_unlabeled_inference(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps({"query": "abc", "spans": []}) + "\n")
    tokenizer = Tokenizer([(0, 0), (0, 3), (0, 0)])
    assert read_data(path, tokenizer)[0]["labels"] == [-100, 0, -100]
    assert encode(tokenizer, "abc")["spans"] is None


def test_overlap_pools_logits_and_preserves_adjacent_entities():
    # A strong GAME vote beats two weak SUBJECT votes in a shared character.
    logits = label_logits([0, 1, 3, 1, 1, 0])
    logits[2, 3] = 5
    spans, _ = decode("🔥x", [(0, 0), (0, 1), (0, 1), (0, 1), (1, 2), (0, 0)], logits)
    assert spans == [
        dict(start=0, end=1, label="GAME", text="🔥"),
        dict(start=1, end=2, label="SUBJECT", text="x"),
    ]
    spans, _ = decode("ab", [(0, 1), (1, 2)], label_logits([1, 1]))
    assert [s["text"] for s in spans] == ["a", "b"]


def test_transitive_overlap_and_outside_vote():
    spans, repairs = decode("abcd", [(0, 2), (1, 3), (2, 4)], label_logits([2, 2, 2]))
    assert spans == [dict(start=0, end=4, label="SUBJECT", text="abcd")]
    assert repairs == 1
    spans, _ = decode("🔥", [(0, 1), (0, 1), (0, 1)], label_logits([0, 0, 1]))
    assert spans == []

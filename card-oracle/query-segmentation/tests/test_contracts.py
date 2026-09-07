import pytest
from cardseg.data import LABELS, collate, encode
from cardseg.decode import decode
from cardseg.metrics import score


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
    decoded, repairs = decode(query, row["offsets"], [max(0, t) for t in row["labels"]])
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
        "abc def", [(0, 3), (4, 7)], [LABELS.index("I-SUBJECT"), LABELS.index("I-GAME")]
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

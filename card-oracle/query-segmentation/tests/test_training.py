"""Training input validation and padded validation-batch decoding."""

import pytest
import torch
from cardseg.data import LABELS
from cardseg.training import decode_batch, validate_config


@pytest.mark.parametrize(
    "key,value",
    [
        ("epochs", 0),
        ("batch_size", 1.5),
        ("max_length", True),
        ("patience", -1),
        ("learning_rate", float("nan")),
        ("head_learning_rate", 0),
        ("weight_decay", -0.1),
    ],
)
def test_invalid_training_config(key, value):
    config = dict(
        epochs=1,
        batch_size=2,
        max_length=128,
        patience=1,
        learning_rate=0.001,
        weight_decay=0,
    )
    validate_config(config)
    config[key] = value
    with pytest.raises(ValueError, match=key):
        validate_config(config)


def test_batch_decoding_ignores_padding_and_counts_repairs():
    rows = [
        dict(query="a", offsets=[(0, 1)]),
        dict(query="bc", offsets=[(0, 1), (1, 2)]),
    ]
    logits = torch.zeros(2, 2, len(LABELS))
    logits[0, 0, LABELS.index("I-SUBJECT")] = 1
    logits[0, 1, LABELS.index("B-GAME")] = 1  # Padding must not produce a span.
    logits[1, :, LABELS.index("B-SUBJECT")] = 1
    predictions, repairs = decode_batch(rows, logits)
    assert repairs == 1
    assert [[s["text"] for s in spans] for spans in predictions] == [["a"], ["b", "c"]]

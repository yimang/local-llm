# Training

Run from the repository root using its shared uv environment.

```bash
uv run python card-oracle/query-segmentation/scripts/train/train.py
uv run python card-oracle/query-segmentation/scripts/train/train.py \
  --config card-oracle/query-segmentation/configs/train-mlp.yaml \
  --output card-oracle/query-segmentation/artifacts/baseline-mlp
uv run python card-oracle/query-segmentation/scripts/train/finetune.py
```

`train.py` freezes ModernBERT, computes float32 features once in process-local RAM,
and optimizes either a linear head or a 256-unit GELU head on CPU. The encoder uses
MPS when available. There is no persistent embedding cache and no resume support.
It prints the estimated feature footprint before extraction. `--tiny N` uses the
first N training examples for both optimization and intentional-overfit evaluation;
this is a wiring check, not a validation result.

`finetune.py` initializes from an offline frozen-head artifact and adapts the last
configured encoder layers plus final normalization and the head. It never reuses
frozen features. Embeddings and earlier layers remain frozen. Encoder dropout is
disabled consistently; the MLP head uses 0.1 dropout during optimization. Gradient
norm is clipped at 1.0. Both trainers use validation exact-span F1 for selection,
keep the earliest exact tie, and never open test data.

Artifacts include the encoder/tokenizer, safetensors head, stable BIO labels,
configuration, dataset hashes, model revision, versions, and epoch history.
Existing artifact directories are never overwritten. Seed 42 is used, but MPS
numerical results can vary across hardware/library versions.

The wiring test used `train-tiny.yaml --tiny 16 --output artifacts/tiny-overfit`
with `train.py`: learning rate 0.01, no weight decay, 3,000 allowed steps. It first
recovered all 16 queries at epoch 662. This intentional memorization setting was
not used for the 8,000-example baseline.

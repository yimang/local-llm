# Query segmentation

Extract typed spans from Pokémon, Riftbound, and One Piece search queries.
Data generation is implemented. Model training, evaluation, and inference are
planned; there is no trained checkpoint yet.

## Layout

```text
configs/                  # future train/eval configurations
data/
  seeds/                  # committed card records and grading/condition vocabularies
  synthetic/              # committed train.jsonl, val.jsonl, test.jsonl
  cache/                  # ignored future tokenization/embedding caches
scripts/
  data/generate.py        # working offline generator
  train/                  # future training entrypoints
  eval/                   # future quality and latency entrypoints
  infer/                  # future single-query and batch inference entrypoints
src/cardseg/              # future shared model, alignment, decoding, and I/O code
tests/                    # future regression tests
docs/data-spec.md         # data contract and generation design
artifacts/                # ignored model weights/checkpoints
reports/                  # ignored metrics, predictions, and latency reports
```

## Generate data

From the repository root:

```bash
uv run python card-oracle/query-segmentation/scripts/data/generate.py --overwrite
```

The generator uses only Python's standard library and can also run directly with
python3. Paths are relative to the script's project root, not the working directory.
Defaults: seed 42, 8,000 train / 1,000 validation / 1,000 test queries.
Existing files require --overwrite. Use --output-dir for a separate experiment.

Every query includes a subject, release year, set alias OR card number, and
grader/grade OR raw condition. GAME is optional. JSONL stores end-exclusive
character spans; tokenization/BIO alignment happens during future training.

## Model plan

Start with answerdotai/ModernBERT-base, frozen encoder, and a linear token head.
Eight span types produce 17 BIO labels including O. Use MPS where supported with
CPU fallback. Save tokenizer, encoder, head, and label mapping together for offline
inference. Select checkpoints on validation exact-span F1; never tune on test.

Evaluation will report exact-span precision/recall/F1, per-label support and F1,
and full-query exact match. Benchmark batch-one warm p50/p95/p99 separately from
model loading, synchronizing MPS during timing. Approximately 100 ms is a target
to measure, not a guarantee. Synthetic metrics do not establish real-query quality.

Seeds currently contain 421 records (150 Pokémon, 150 Riftbound, 121 One Piece).
One Piece is limited to Romance Dawn. The held-out split isolates normalized
subject names, not sets or template families. Dependencies for model work should
be added later with uv add at the repository root, updating its shared lockfile.

# Query segmentation

Extract typed character spans from Pokémon, Riftbound, and One Piece card searches.
Training, offline inference, exact-span evaluation, and local latency benchmarking
are implemented. See [the experiment report](docs/baseline-results.md) for measured
quality, latency, and limitations, and [PLAN.md](PLAN.md) for the original contract.

## Demo

From the repository root:

```bash
uv run python card-oracle/query-segmentation/scripts/infer/segment.py \
  --model-dir card-oracle/query-segmentation/artifacts/baseline-demo \
  --query "roronoa zoro psa 10 op01 2022"

uv run python card-oracle/query-segmentation/scripts/infer/segment.py \
  --model-dir card-oracle/query-segmentation/artifacts/baseline-demo \
  --interactive
```

Interactive mode loads once; type one query per line and press Ctrl-D to exit.
`--device cpu` or `--device mps` selects an explicit device. Stdout is JSON;
diagnostics go to stderr. `--input queries.jsonl --output predictions.jsonl`
processes a batch with optional IDs. Empty input returns no spans. Malformed and
overlength records produce explicit errors; valid subsequent records still run.
Artifacts load offline, with no catalog lookup or template completion.

## Reproduce training

Use the root uv environment and pinned Python 3.12. Downloaded weights, checkpoints,
and routine reports are ignored by Git. A new checkout must train or receive the
local artifact before running the demo. The frozen training CLI keeps its download
cache under `data/cache/huggingface/`; the model revision is pinned in its config.

```bash
uv sync
uv run python card-oracle/query-segmentation/scripts/train/train.py
uv run python card-oracle/query-segmentation/scripts/train/train.py \
  --config card-oracle/query-segmentation/configs/train-mlp.yaml \
  --output card-oracle/query-segmentation/artifacts/baseline-mlp
uv run python card-oracle/query-segmentation/scripts/train/finetune.py \
  --output card-oracle/query-segmentation/artifacts/baseline-tail
uv run python card-oracle/query-segmentation/scripts/train/finetune.py \
  --config card-oracle/query-segmentation/configs/train-full.yaml \
  --source card-oracle/query-segmentation/artifacts/baseline-tail \
  --output card-oracle/query-segmentation/artifacts/baseline-demo
```

Existing output directories are protected; choose new paths for repeat experiments
and pass their paths to the next stage. There is no optimizer-resume support.
Validation exact-span F1 selects each checkpoint. Test data is excluded from
training and model selection. Float32/SDPA works without CUDA or FlashAttention.

## Evaluation and checks

```bash
uv run python card-oracle/query-segmentation/scripts/eval/evaluate.py \
  --model-dir card-oracle/query-segmentation/artifacts/baseline-demo \
  --input card-oracle/query-segmentation/data/synthetic/test.jsonl \
  --output card-oracle/query-segmentation/reports/test
uv run python card-oracle/query-segmentation/scripts/eval/benchmark.py \
  --model-dir card-oracle/query-segmentation/artifacts/baseline-demo \
  --device cpu --samples 1000 \
  --output card-oracle/query-segmentation/reports/latency-cpu.json
uv run python card-oracle/query-segmentation/scripts/eval/verify_artifact.py \
  --model-dir card-oracle/query-segmentation/artifacts/baseline-demo \
  --reference card-oracle/query-segmentation/artifacts/baseline-demo/reload-reference.json
uv run pytest -q card-oracle/query-segmentation/tests
uv run ruff check .
uv run ty check
```

Repeat the benchmark with `--device mps` and a different output file for comparison.
The benchmark synchronizes MPS and separates model loading from warm requests.

## Data and scope

The committed 8,000/1,000/1,000 synthetic splits are unchanged. Labels are SUBJECT,
GAME, SET, CARD_NUMBER, YEAR, GRADER, GRADE, and CONDITION, represented as 17 BIO
classes. Offsets are zero-based, end-exclusive Python Unicode character indices.
Tokens with overlapping character offsets (such as byte-level emoji pieces) share
one BIO decision from their mean logits. Labeled datasets must supply a `spans`
list; use `[]` for negative examples. Null annotations are rejected.

Every generated query has a subject, year, set alias OR card number, and raw
condition OR grader/grade; GAME is optional. Held-out subject names share template
families and may share sets with training. Seeds cover 421 records across three
games; One Piece covers Romance Dawn only. Short searches, typos, negation, prices,
and unknown sets require separate real-query evaluation before production claims.

The generator remains available independently:

```bash
uv run python card-oracle/query-segmentation/scripts/data/generate.py \
  --output-dir /tmp/cardseg-data
```

See [the data specification](docs/data-spec.md) for the exact generation contract.

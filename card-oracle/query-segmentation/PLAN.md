# Query segmentation implementation plan

Updated September 7, 2026. This is the current plan for training, evaluation, and
inference. [The data specification](docs/data-spec.md) defines synthetic generation.

## Goal and current state

Run a local ModernBERT baseline on a MacBook that accepts a card search query and
returns typed character spans. Start by training only the classification head;
encoder fine-tuning is a later experiment. Target approximately 100 ms per warm
query, measured on the actual hardware rather than assumed.

Completed:

- Offline randomized generator at `scripts/data/generate.py`.
- Committed seed data: 150 Pokémon, 150 Riftbound, 121 One Piece records.
- Committed synthetic splits: 8,000 train, 1,000 validation, 1,000 test.
- Verified generation requirements, grade scales, normalized query uniqueness,
  subject separation, and byte-identical regeneration.

Not implemented: tokenizer alignment, model/head training, checkpoint loading,
inference, model evaluation, latency benchmarking, and automated regression tests.
Existing train/eval/infer directories contain documentation only.

## Fixed data and label contracts

Games: Pokémon, Riftbound, One Piece. Each generated query requires a subject,
release year, exactly one set alias or card number, and exactly one condition mode:
raw CONDITION or graded GRADER + GRADE. GAME is optional. Set names remain in seed
records; set qualifiers in generated queries use abbreviated aliases. No quantity,
budget, price, minimal-query, or typo templates are part of this baseline.

PSA uses whole numbers 1–10. Beckett/BGS, SGC, and CGC use 1–10 with half-points.
TAG uses half-points through 8.5, then 9 and 10, without 9.5. Raw conditions are Near
Mint, Lightly Played, Moderately Played, Heavily Played, and Damaged, including the
aliases in `data/seeds/attributes.json`. Grade tiers and historical grading systems
are outside this baseline.

Input records contain `query` and `spans`, with zero-based, end-exclusive Python
Unicode character offsets. There are eight span types:

`SUBJECT`, `GAME`, `SET`, `CARD_NUMBER`, `YEAR`, `GRADER`, `GRADE`, `CONDITION`.

Define a stable 17-label mapping: O first, then B-TYPE and I-TYPE for each type in
the order above. Store that mapping with each checkpoint. BIO labels are produced
at training time, not added to the committed synthetic data.

Generation constraints are not inference guarantees. Inference must return actual
predictions, including missing or inconsistent fields, rather than inventing spans
to complete the required template.

## Implementation layout

| Location | Responsibility |
| --- | --- |
| `configs/train.yaml` | Versioned baseline model and training settings |
| `src/cardseg/data.py` | JSONL validation, tokenizer alignment, batch collation |
| `src/cardseg/model.py` | Frozen encoder plus trainable token head |
| `src/cardseg/decode.py` | Shared token-to-character-span decoding |
| `src/cardseg/artifacts.py` | Checkpoint saving/loading and compatibility checks |
| `src/cardseg/metrics.py` | Span and query metrics |
| `scripts/train/train.py` | Training CLI |
| `scripts/eval/evaluate.py` | Quality evaluation CLI |
| `scripts/eval/benchmark.py` | Local batch-one latency CLI |
| `scripts/infer/segment.py` | Single-query, interactive, and batch inference CLI |
| `tests/` | Data/model/CLI regression tests |
| `artifacts/` | Ignored checkpoints and optional resumable training state |
| `reports/` | Ignored metrics, predictions, and performance reports |
| `data/cache/` | Ignored optional frozen-embedding caches |

Use the repository-root uv environment and lockfile. Add required dependencies
with `uv add`, and test against the repository's pinned Python version. Do not
create a separate virtual environment or lockfile. Keep entry scripts thin and
make the shared package importable through a consistent project-local bootstrap
without requiring users to configure PYTHONPATH. Run repository Ruff and ty checks.

## Training design

### Architecture

Load `answerdotai/ModernBERT-base` and its matching fast tokenizer. For a batch of
B queries with T tokens, the encoder returns [B, T, H]; a linear layer returns
[B, T, 17]. Read H from the model configuration (expected 768). Do not pool token
representations. No CRF, catalog lookup, or generative decoder is needed.

Freeze encoder parameters, run encoder forwards without gradients, and keep the
encoder in evaluation mode even when training the head. Optimize only the head
with token cross-entropy. Ignore padding and special tokens with target -100.

### Alignment and input validation

- Validate labels, bounds, sorted non-overlapping spans, and nonempty span text.
- Tokenize original text with offset mapping; do not normalize it before alignment.
- Assign B-TYPE to the first subword in a span and I-TYPE to subsequent subwords.
  Unannotated ordinary tokens get O.
- Account explicitly for tokenizer leading-whitespace offsets. Reject a token
  crossing a semantic annotation boundary; report file and line rather than
  silently dropping the example.
- Reject overlength inputs without truncation. Increase max length explicitly if
  actual data requires it.

### Initial experiment settings

| Setting | Initial value |
| --- | --- |
| Model | answerdotai/ModernBERT-base |
| Training mode | head-only |
| Device | auto: MPS when supported, otherwise CPU |
| Precision | float32 |
| Maximum token length | 128, including special tokens |
| Batch size | 16 |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0.01 |
| Maximum epochs | 20 |
| Early-stopping patience | 3 epochs |
| Selection metric | Validation exact-span micro F1 |
| Random seed | 42 |

These are starting values, not validated optimal hyperparameters. Choose the
earliest checkpoint on an exact metric tie. Log loss and validation metrics each
epoch; do not read the test split during training. Record actual device, model
revision, dependency versions, configuration, and local dataset hashes. Seed Python,
NumPy if used, and PyTorch; disclose hardware-dependent numerical nondeterminism.

Use an attention path supported on the tested Mac. Explicit MPS requests must fail
clearly if unsupported; auto fallback must be logged. Do not require CUDA or
FlashAttention. Diagnose genuine model/runtime errors rather than masking all
exceptions as fallback conditions.

An optional frozen-state cache may avoid repeated encoder passes. Key it by data
hash, tokenizer/model revision, max length, precision, and preprocessing version.
Estimate its disk footprint before enabling it. Never reuse it when training the
encoder. Implement uncached correctness first.

### Saved checkpoint

Save the encoder, tokenizer, head weights in safetensors, label maps, architecture
configuration, and training summary together under `artifacts/baseline/`. A fresh
process must load the artifact offline without the original training data or
network access. Validate schema/label compatibility when loading. Guard existing
artifacts against accidental overwrite. Optimizer state is optional and separate
from the inference artifact; do not claim resume support until implemented.

## Inference contract

Proposed commands below run from this project's directory after implementation:

```bash
uv run python scripts/infer/segment.py --model-dir artifacts/baseline \
  --query "roronoa zoro psa 10 op01 2022"
uv run python scripts/infer/segment.py --model-dir artifacts/baseline --interactive
uv run python scripts/infer/segment.py --model-dir artifacts/baseline \
  --input queries.jsonl --output predictions.jsonl
```

Require exactly one input mode. Interactive mode loads the model once and accepts
one query per line until EOF. Batch input contains `query` and an optional `id`,
which is copied to output. Diagnostics go to stderr; stdout is JSON only.

Illustrative output, not a verified prediction:

```json
{
  "query": "roronoa zoro psa 10 op01 2022",
  "segments": [
    {"text": "roronoa zoro", "start": 0, "end": 12, "label": "SUBJECT"},
    {"text": "psa", "start": 13, "end": 16, "label": "GRADER"},
    {"text": "10", "start": 17, "end": 19, "label": "GRADE"},
    {"text": "op01", "start": 20, "end": 24, "label": "SET"},
    {"text": "2022", "start": 25, "end": 29, "label": "YEAR"}
  ]
}
```

Use argmax token predictions, discard special/padding tokens, and merge BIO spans
using offsets. An I-TYPE after O or a different type starts a new span; count such
repairs in evaluation. Extract every output text from the original query. Preserve
multiple spans of the same type and return spans in text order.

Optional `--confidence` adds the arithmetic mean of selected-token softmax
probabilities per span. This is uncalibrated, not a correctness probability.
Optional `--debug-tokens` exposes token offsets and labels; optional `--timing`
adds runtime measurements. Do not silently suppress low-confidence spans.

Whitespace-only input returns no segments without a forward pass. Overlength or
malformed input returns an explicit error, never a partial parse. Batch errors
include input line/ID, allow other records to complete, and cause nonzero exit.
No canonical entity IDs, grade normalization, or ecommerce filter construction.

## Quality evaluation

`scripts/eval/evaluate.py` accepts a saved checkpoint, labeled JSONL, and output
location. Reuse inference decoding exactly. Report:

- Exact character-boundary-and-type span micro precision, recall, and F1.
- Per-label precision/recall/F1 and support; macro F1 over gold-supported labels.
- Full-query exact match: all spans and labels match, including no extra spans.
- Slices for raw/graded queries, set/number queries, and query lengths.
- BIO repair counts, per-query gold/predicted spans, and categorized errors.

Report game slices only where game attribution is reliable. An optional GAME span
does not identify every query's domain. A local seed join may support reporting,
but ambiguous matches must be marked unknown and never feed model predictions.

Use validation for checkpoint selection and experiments, then run the fixed test
set for final reporting. Token accuracy is diagnostic only because O can dominate.
Explicitly identify every current report as synthetic evaluation.

Current held-out names share grammar and often sets with training. All queries
include required qualifiers; short searches, negation, typos, and unknown sets are
not represented. One Piece covers only Romance Dawn. A later independently
annotated natural-query gold set is necessary before production claims; it is not
part of this baseline's completion criteria or a reason to alter current splits.

## Latency evaluation

`scripts/eval/benchmark.py` loads one artifact and runs batch-one queries. Measure
cold model load separately from warmed tokenization, model forward, decoding, and
total request latency. Exclude initial downloads and report that exclusion.

Start with 10 warmups and 100 measured requests for smoke testing; use at least
1,000 representative requests for a more useful tail-latency report. Report
p50/p95/p99, query/token lengths, hardware, actual device, precision, library
versions, and sample count. Synchronize MPS around timed GPU work. Compare CPU and
MPS using the same corpus. A local warm request below approximately 100 ms is the
target; model loading, network service, and future catalog work are excluded.

## Milestones and acceptance checks

1. **Data alignment and model wiring:** automated tests cover Unicode, repeated
   substrings, subwords, special/padding tokens, invalid spans, and overlength
   handling. A tiny intentional overfit experiment shows decreasing loss and
   recovers the training spans; this tests wiring, not generalization.
2. **Training and artifacts:** run the baseline against train/validation, save the
   best checkpoint, reload in a fresh process, and compare labels/spans exactly
   and logits within a documented tolerance. Check frozen encoder weights stay
   unchanged while head weights update.
3. **Inference:** single, persistent, and batch modes work offline; empty/invalid
   inputs and output text/offset round trips behave as specified.
4. **Evaluation:** test exact-span metrics on hand-checked fixtures, produce a
   synthetic test report and inspect representative errors. No invented accuracy
   threshold is required to declare a measured baseline complete.
5. **Performance and handoff:** produce actual Mac timings, run the repository's
   static checks and relevant tests, and document commands, results, and limits.

Train/eval configuration and code/tests are committed. Existing seeds and synthetic
splits remain committed. Downloaded model weights, caches, checkpoints, and routine
reports stay ignored. Do not regenerate data or change its schema as a hidden
training step. Full encoder fine-tuning, quantization, export/runtime optimization,
and catalog entity linking require subsequent experiments after this baseline.

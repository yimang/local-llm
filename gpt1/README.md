# Small GPT-1 reproduction

Train a 27.4M-parameter causal decoder from random initialization on 100M unique
Wikipedia tokens. This is a scaled reproduction of GPT-1's method; it does not
aim to match the original benchmark scores.

## Current scope

Implemented: streaming dataset preparation, tokenizer training, integrity audit,
GPT-1-style model, GPU benchmark, and resumable pretraining. Downstream task
fine-tuning and the pretrained-versus-scratch comparison are the next stage.
Pretraining alone does not reproduce the paper's central transfer-learning result.

The initial run reached **20,004,864 token presentations (step 1,221)** on the
Apple GPU using PyTorch MPS. Validation perplexity fell from 1,387 to 125.5.
Generated text has recognizable Wikipedia phrasing but remains unreliable and
repetitive. See [measured results and sample outputs](results/README.md).

The [benchmark plan and context handoff](BENCHMARK_PLAN.md) records the research
and next agreed action: a short full-size GPT-1 throughput benchmark on this Mac,
before choosing a longer training run. MLX porting is deferred.

Code, configuration, tests, and small result records are tracked in Git. Dataset
files and model weights are local artifacts; a fresh clone must prepare the data
and train a checkpoint before running inference. MLX is not used by this experiment.

## Setup and data

Run from the repository root. Dependencies are pinned by the repository's `uv.lock`.

```bash
uv sync --locked
HF_HOME=.cache/huggingface uv run python -m gpt1.prepare --output gpt1/data/wiki-100m
uv run python -m gpt1.verify_data gpt1/data/wiki-100m
```

The defaults prepare at least 100M training tokens and 1M each for validation and
test. Whole articles are retained, so each split can exceed its target by one
article. `manifest.json` records the resolved source revision, settings, counts,
and SHA-256 hashes. To reproduce an existing dataset, pass its revision explicitly:

```bash
HF_HOME=.cache/huggingface uv run python -m gpt1.prepare \
  --revision b04c8d1ceb2f5cd4588862100d08de323dccfbaa \
  --output gpt1/data/wiki-100m-recreated
```

Preparation shuffles source shards and uses a 1,000-article streaming shuffle
buffer with seed 42. This is a reproducible bounded sample, not a globally uniform
sample of Wikipedia. It removes articles shorter than 200 characters, duplicate
IDs, and whitespace-normalized exact text duplicates. Near-duplicate passages
are not removed. Hash-based article splits are assigned before tokenization;
deduplication is global across all splits. Paragraph order is retained.

The 16K byte-level BPE tokenizer is trained on approximately 10M characters from
training articles only. This differs from GPT-1's original tokenizer. All text is
tokenized with that same tokenizer. Each document ends with an explicit EOS token.
Training samples random contiguous windows of the packed token stream; attention
can cross EOS boundaries. Token presentations therefore measure sampling with
replacement, rather than guaranteed complete epochs over all unique tokens.

`*.bin` files use little-endian uint16 IDs; JSONL files retain article text, IDs,
source URLs, hashes, and token offsets. The source is
[Wikimedia Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia), snapshot
`20231101.en`. Preserve Wikipedia attribution and applicable share-alike terms
when redistributing the source text; see the dataset card. Generated data and
checkpoints are ignored by Git.

Preparation refuses to overwrite an existing directory. An interrupted run has
`incomplete.json` and must be restarted into a new directory; data preparation
does not yet resume downloads/partial outputs.

## Model and training

The model has six layers, width 512, eight heads, feed-forward width 2,048, and
context length 512. It uses post-LayerNorm residual blocks, learned positions,
tanh-approximate GELU, causal attention, 0.1 dropout, and tied embeddings. Training
uses FP32 and AdamW, gradient clipping at 1.0, and warmup followed by cosine decay.
Biases and LayerNorm parameters are excluded from weight decay.

```bash
# Benchmark three microbatch sizes on available hardware.
uv run python -m gpt1.benchmark --output gpt1/results/benchmark.json

# Short pilot, retaining the main run's learning-rate schedule.
uv run python -m gpt1.train --data gpt1/data/wiki-100m \
  --output gpt1/runs/wiki-small --stop-after 100

# Continue the pilot to about 20M token presentations (1,221 total steps).
uv run python -m gpt1.train --data gpt1/data/wiki-100m \
  --output gpt1/runs/wiki-small --resume gpt1/runs/wiki-small/latest.pt \
  --stop-after 1221

# Next milestone: continue to roughly 100M presentations (6,104 total steps).
uv run python -m gpt1.train --data gpt1/data/wiki-100m \
  --output gpt1/runs/wiki-small --resume gpt1/runs/wiki-small/latest.pt \
  --stop-after 6104

# Optional: continue to the configured total of roughly 300M presentations.
uv run python -m gpt1.train --data gpt1/data/wiki-100m \
  --output gpt1/runs/wiki-small --resume gpt1/runs/wiki-small/latest.pt
```

The default effective batch is 32 sequences: microbatch 4, accumulation 8,
16,384 predicted tokens per update. The horizon is 18,311 steps (300,007,424 token
presentations); `--stop-after` sets an absolute stopping step without changing
the learning-rate schedule. A 100-step pilot is a smoke test, not a trained LM.

The runner saves model, optimizer, step, data-manifest fingerprint, CPU and device
random states, and NumPy sampling state. Resume requires the same configuration,
data manifest, and device backend. CPU resume equivalence is tested exactly;
hardware kernels may still be nondeterministic. Only load trusted local checkpoints.
`latest.pt` is atomically replaced; copy it separately to retain milestone models.

Validation uses fixed sampled windows and reports their loss/perplexity. This is
an estimate, not a full-validation-set sweep. The test split remains unused during
training. Perplexity should only be compared across models using the same tokenizer
and evaluation text. Inspect `metrics.jsonl`, `run.json`, and `summary.json` in the
run directory. MPS benchmark memory is current/driver allocation, not a measured peak.

## Inference

Generate a continuation from the saved checkpoint using its training tokenizer:

```bash
uv run python -m gpt1.generate \
  --prompt "The funniest joke I've ever heard is" \
  --max-new-tokens 50 --temperature 0.8 --top-k 40
```

Use `--temperature 0` for greedy decoding, `--device cpu` to run without GPU
access, and `--output path.json` to save the result. The default seed is 42.
The loader verifies the data-manifest and tokenizer hashes against the checkpoint.
Generation uses a sliding context of at most 512 tokens without a KV cache.
Compare the joke prompt at [step 100](results/joke-completion.json) and
[step 1,221](results/joke-completion-step-1221.json). Both checkpoints produce
incoherent text; the later one has more recognizable sentence fragments.

Run the fixed qualitative suite (10 prompts, two sampling seeds plus greedy
decoding, 35 generated tokens per output) with:

```bash
uv run python -m gpt1.probe
```

It uses CPU inference and writes the complete outputs under `gpt1/results/`.
Run it against a stopped checkpoint so all prompts use the same model. These
qualitative probes are not downstream benchmark scores.

## Verification

```bash
uv run python -m pytest gpt1/tests -q
uv run ruff check .
uv run ruff format --check .
uv run ty check
```

Tests cover causal isolation, masked targets, tiny-batch overfitting, deduplication,
article splits, EOS boundaries, shifted targets, and exact CPU checkpoint resume.

## Next experiments

1. Continue to 100M presentations; expand toward 300M if validation still improves.
2. Implement SST-2, MRPC, and CoLA fine-tuning. Compare random initialization,
   pretrained initialization, and pretraining plus auxiliary LM loss (weight 0.5).
3. Use comparable tuning budgets and three fine-tuning seeds. MRPC should encode
   both sentence orderings and sum representations as in the original method.
4. Report accuracy for SST-2, accuracy/F1 for MRPC, and Matthews correlation for
   CoLA; separate validation-based selection from final held-out evaluation.
5. Increase unique training data to 500M tokens if overfitting limits transfer.

References: [original paper](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf),
[original code](https://github.com/openai/finetune-transformer-lm).

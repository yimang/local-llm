# Initial GPT-1 results: 20M token presentations

Completed September 12, 2026, on the local Mac using PyTorch 2.14.0 with MPS.

| Split | Tokens | Articles | UTF-8 text bytes |
| --- | ---: | ---: | ---: |
| Training | 100,000,737 | 138,146 | 381,744,036 |
| Validation | 1,001,563 | 1,380 | 3,818,570 |
| Test | 1,003,446 | 1,341 | 3,859,315 |

The prepared directory occupies about 624 MiB, including text, metadata, binary
tokens, and tokenizer. File checksums, article and exact-text isolation across
splits, token ID ranges, offsets, and EOS boundaries all passed the full data audit.
See [data-manifest.json](data-manifest.json) for provenance and exact file hashes.

The randomly initialized model contains 27,368,448 parameters. The short benchmark
measured 8,692 / 12,646 / 16,564 tokens per second for microbatches 1 / 2 / 4,
respectively. See [benchmark.json](benchmark.json); MPS allocation figures are
snapshots after training steps, not peak memory measurements.

The 100-step training smoke test used microbatch 4 and accumulation 8. It processed
1,638,400 tokens in 73.94 seconds including final validation and checkpoint save.
Steady training throughput was approximately 22,500 tokens/second; accumulation
amortizes optimizer-update overhead relative to the benchmark. Final training
loss was 7.286 and validation loss was 7.235 (estimated perplexity 1,387).
See [pilot.json](pilot.json). This is an early warmup checkpoint, not a trained
language model or evidence of downstream transfer.

At this short-run throughput, 100M / 300M presentations would take approximately
1.2 / 3.7 hours of training, plus evaluation and checkpoint overhead. Sustained
thermal conditions and other workloads can change these estimates.

Training resumed from step 100 and stopped at step 1,221, totaling **20,004,864
token presentations**. The resumed session took 833.57 seconds (13m 54s).

| Step | Token presentations | Validation loss | Perplexity |
| --- | ---: | ---: | ---: |
| 100 | 1,638,400 | 7.235 | 1,387.3 |
| 200 | 3,276,800 | 6.545 | 695.9 |
| 400 | 6,553,600 | 5.880 | 357.7 |
| 600 | 9,830,400 | 5.474 | 238.3 |
| 800 | 13,107,200 | 5.206 | 182.3 |
| 1,221 | 20,004,864 | 4.832 | 125.5 |

See [training-20m.json](training-20m.json) for the configuration, complete validation
history, session summary, and checkpoint checksum. Validation uses the same fixed
sampled windows at every evaluation; this is not full-set perplexity. Training
samples with replacement, so token presentations are not a count of distinct
tokens seen. The test split has not been used to evaluate the model.

Checkpoint: `gpt1/runs/wiki-small/latest.pt` (step 1,221, ignored by Git). The
original step-100 checkpoint is retained locally as `step-000100.pt`. Resume
commands are in the [experiment README](../README.md). Neither the configured
300M-presentation run nor downstream fine-tuning has been completed.

## Qualitative inference

The [initial joke sample](joke-completion.json) largely repeats common words.
The [20M-token sample](joke-completion-step-1221.json) produces sentence fragments,
but does not tell a joke. Both use temperature 0.8, top-k 40, seed 42, and CPU
inference with the same tokenizer.

The [fixed prompt suite](completion-probes-step-1221.json) includes 10 prompts,
each sampled with seeds 42 and 7 at temperature 0.8, plus greedy decoding.
All 30 outputs are retained; the suite is exploratory, not a formal benchmark.

- Geographic prompts evoke geographic vocabulary; some short phrases are plausible.
- Fluent templates can be factually wrong: greedy completion of "A cat is a small"
  describes a sea snail and its taxonomy.
- Simple arithmetic, counting, and story-continuity prompts were not completed
  successfully in these samples.
- Greedy decoding frequently repeats phrases; sampling changes the wording but
  does not establish coherent meaning.

The decline in language-model loss supports improved prediction of held-out
Wikipedia text. It does not yet establish reliable factual recall, reasoning,
long-form coherence, or transfer to supervised tasks.

Verification: all 36 repository tests passed, including five new experiment tests;
repository lint and type checks passed; all new Python files passed formatting.
Repository-wide formatting reports five pre-existing files under
`card-oracle/query-segmentation/scripts/`; those files were not changed.

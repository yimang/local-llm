# MacBook pretraining: benchmark and experiment proposal

Research date: September 12, 2026. This document proposes experiments; it does not
claim that the current checkpoint has achieved any benchmark scores below.

## Handoff: next action agreed with the user

**Run a short full-size GPT-1 training-throughput benchmark before further
pretraining or implementing the evaluation suite.** This takes priority over the
longer experiment sequence below. The benchmark has not been run yet.

Verified local hardware: Apple M5 Pro MacBook Pro, 64 GB unified memory, and about
1.7 TiB free disk space on September 12, 2026. Memory and storage appear sufficient
for the original-size architecture; runtime must be measured. The previously
discussed 5,000 tokens/second for full GPT-1 was an illustrative assumption, not a
benchmark result. The measured 21K-22.6K tokens/second belongs to our 27M model.

Next-context checklist:

1. Add a separate benchmark configuration: 12 layers, width 768, 12 heads,
   feed-forward width 3,072, context 512, and approximately 40K vocabulary entries
   (roughly 117M parameters). Record the exact vocabulary size and parameter count.
   The paper specifies 40K BPE merges, which is not exactly 40K vocabulary entries;
   an approximate vocabulary is sufficient for this throughput pilot if labeled.
   Keep the GPT-1 post-LayerNorm/GELU/learned-position architecture and FP32 baseline.
2. Use `uv run python -m gpt1.benchmark --config <new-config> --device mps
   --batches 1 2 4 --steps 50 --output <new-results-file>`. The current benchmark
   uses random token IDs, performs full forward/backward/AdamW updates, synchronizes
   GPU timing, and excludes three warmup steps. It does not need a new tokenizer
   or dataset and must not load the 27M checkpoint into the larger model.
3. Inspect correctness, finite loss, and memory at the smallest batch first. If
   stable, test larger microbatches and a short accumulation run with effective
   batch 64 sequences to approximate the paper's batch. Cap the pilot to a few
   minutes; do not start an hours-long pretraining run as part of the benchmark.
4. Save model/config/runtime versions, actual parameter count, throughput by batch,
   and memory readings. Label MPS current/driver allocations correctly; the current
   script does not measure peak memory. Extend its result metadata if necessary.
5. Extrapolate 1B, 10B, and 100B token-presentation runtimes from measured throughput,
   with a clear allowance for evaluation, checkpointing, and sustained thermal
   effects. These are scenarios, not a claim about the paper's exact token budget.
6. Report whether a full-size run is practical before choosing the next training
   budget. Preserve the existing dataset, small-model configuration, and checkpoints.

Current local state: the 27,368,448-parameter model stopped at step 1,221 after
20,004,864 presentations. Validation loss is 4.83196 and perplexity 125.457.
`gpt1/runs/wiki-small/latest.pt` is the completed checkpoint;
`step-000100.pt` preserves the earlier smoke test. Data is in
`gpt1/data/wiki-100m`. Both directories are ignored by Git and exist only on this
machine. Small measured results are tracked under `gpt1/results/`. Training is
stopped, and the completion automation was deleted. MLX porting is deferred.

Project dependencies are managed by uv (`pyproject.toml` and `uv.lock`). The
existing implementation and 20M-token results were pushed in commit `d7fd484`
on branch `codex/segmentation-correctness-cleanup` to
`https://github.com/yimang/local-llm.git`. Do not restart the experiment from scratch
or infer that the branch name describes this GPT-1 work.

## Recommendation

Use existing benchmarks. Start with BLiMP for grammatical knowledge, held-out
language-model loss for learning efficiency, and SST-2/MRPC fine-tuning for the
original GPT paper's transfer-learning claim. Add EWoK as a harder diagnostic.
Use a fixed human-reviewed generation suite to make progress tangible.

The research question is: **What capabilities can a model pretrained from scratch
on this MacBook acquire within 4, 12, and 48 hours of training?** Report capability
versus wall time and unique data, rather than one combined score. The 27.4M model
is a small language model; scaling it toward the original GPT size is a later
experiment, not equivalent to training a modern billion-parameter assistant.

## What fits

| Candidate | What it measures | Role |
| --- | --- | --- |
| BLiMP | Preference for grammatical over minimally altered ungrammatical sentences | Primary zero-shot benchmark |
| BabyLM evaluation suite | Limited-data language acquisition and downstream transfer | Reference protocol and comparison ecosystem |
| SST-2 / MRPC / CoLA | Sentiment / paraphrase / grammatical acceptability after supervised adaptation | SST-2 first, MRPC second, CoLA optional |
| EWoK | Basic physical and social knowledge used in context | Stretch diagnostic, not first success gate |
| Held-out Wikipedia and story text | Predictive fit and cross-domain generalization | Cheap learning curves |
| WikiText-2 / WikiText-103 | Standard Wikipedia language modeling | Optional after contamination audit |
| TinyStories | Simplified story-language learning | Alternative training corpus and generation experiment |
| PIQA / HellaSwag | Commonsense solution/continuation selection | Later checks; risk of near-chance results |

[BLiMP](https://github.com/alexwarstadt/blimp) contains 67 paradigms of 1,000 pairs.
Use full-sentence probability comparisons following the reference protocol; do
not ask the model to explain which sentence is grammatical. Uniform random choice
has 50% expected accuracy. Report overall and per-paradigm scores.

[BabyLM's 2026 guidelines](https://babylm.github.io/guidelines.html) define
10M-word and 100M-word limited-data tracks and permit custom corpora. Its
[current evaluation code](https://github.com/babylm-org/babylm-eval/tree/main/strict)
has fast and full evaluation and supports adapting a pure PyTorch model.
Borrowing tasks does not automatically make this run an official submission:
check word-exposure caps, model packaging, required checkpoints, data rules, and
the selected year's scoring details before claiming compliance.

For a published reference, the
[2025 GPT-2 BabyLM baseline](https://huggingface.co/BabyLM-community/babylm-baseline-100m-gpt2)
reports 124M parameters, about 1B word presentations, BLiMP 74.88%, and EWoK 51.67%.
These use BabyLM-filtered evaluation data. Our model is smaller, differently
trained, and uses a different tokenizer; those are reference points, not matched
baselines or promised targets. Do not compare raw BLiMP scores directly to the
filtered BabyLM scores. Current 2026 results are also version-specific.

[EWoK](https://ewok-core.github.io/) tests context-dependent world knowledge with
paired contexts and targets. It already addresses much of what our improvised
story questions tried to test. Use its reference comparison/aggregation rules;
do not substitute a new multiple-choice metric and retain the same benchmark
name. The full dataset requires accepting its access terms.

[GLUE](https://gluebenchmark.com/) provides established supervised tasks. SST-2
and CoLA remain useful for our GPT-1 reproduction even though the current BabyLM
pipeline omits them. Treat our GLUE selection as a separate evaluation track.

[TinyStories](https://arxiv.org/abs/2305.07759) demonstrates coherent story
generation in a simplified domain with models below 10M parameters. It is
synthetic training data, not evidence that equally small Wikipedia-trained models
will acquire broad knowledge. Use the
[released corpus](https://huggingface.co/datasets/roneneldan/TinyStories) without
paying to generate a new one. Label this experiment as training on synthetic text.

[WikiText](https://huggingface.co/datasets/Salesforce/wikitext) overlaps our source
domain and potentially our exact training articles. Audit article/text overlap
before treating it as held-out evaluation; a newer Wikipedia snapshot alone does
not prevent contamination. Published word-level perplexities are not directly
comparable to our BPE-token perplexity.

[PIQA](https://arxiv.org/abs/1911.11641) and
[HellaSwag](https://aclanthology.org/P19-1472/) are worthwhile later, but failure
could hide meaningful early language learning. Original PIQA has two choices;
HellaSwag has four. Do not conflate original PIQA with newer GlobalPIQA protocols.
MMLU, GSM8K, and code-generation leaderboards are not proposed as first gates.

## Evaluation protocol to implement

1. **Snapshot today's checkpoint.** Preserve step 1,221 and its hash. Evaluate
   it and the saved step-100 checkpoint before more pretraining. Reconstruct a
   random-initialization baseline with the same architecture and seed.
2. **Implement likelihood scoring.** Add a thin adapter for our GPT to
   [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/model_guide.md)
   or the chosen BabyLM scorer. Reuse reference task definitions. Unit-test shifted
   targets, prefix masks, right padding, first-token/EOS treatment, context limits,
   and byte-BPE boundaries. Score candidates directly, without sampling or chat
   instructions. Keep temperature at the reference setting, not the generation
   temperature. Pin the evaluator revision, dataset revision, and configuration.
3. **Fast development evaluation.** If using raw BLiMP, freeze 100 pairs from
   every paradigm (6,700 total) before scoring. Reserve the remaining 900 per
   paradigm for final internal confirmation; label these custom splits explicitly.
   Run full official BLiMP separately for a descriptive reference score. If using
   BabyLM instead, use its official fast/full protocol without mixing versions.
4. **Language-model evaluation.** Keep the current sampled validation curve for
   continuity. Add document-aware evaluation with a fixed stride and at most
   512 tokens of context. Score each target once and account for boundary tokens
   explicitly. Freeze validation text in both Wikipedia and stories. Evaluate the
   test sets only after configuration selection. Report token NLL/perplexity for
   the same tokenizer; add bits per UTF-8 byte when comparing tokenizers, using
   identical text and counting rules. Do not relabel old perplexities computed by
   the sampled-window method as full-set results.
5. **Transfer experiment.** Fine-tune on SST-2, then MRPC, with three seeds and an
   equal small learning-rate search for pretrained and random models. Compare a
   TF-IDF linear classifier too. Use training-only inner development data for
   tuning if keeping the public dev split as the final local holdout. Otherwise
   call reported results dev scores, not test scores. For MRPC use F1 and accuracy;
   for CoLA use Matthews correlation. Keep supervised compute separate from
   pretraining compute. Add auxiliary LM loss only after the main comparison works.
6. **Generation panel.** Freeze 50 new prompts spanning encyclopedia prose,
   everyday narratives, simple explanations, and short context recall. Use two
   fixed seeds and a fixed 100-token output budget. Preserve all outputs; report
   repetition and blinded human judgments of grammaticality, relevance, and local
   consistency. Include examples that reveal failures. This is a supplementary
   rubric, not a replacement public benchmark or a claimed factuality measure.

Measure evaluator runtime on a small batch before choosing cadence; use quick
checks at 20M, 100M, 300M, 500M, and 1B presentations, with expensive fine-tuning
only at selected milestones. Budget evaluation separately, initially 10-20% of
experiment time. Use paired bootstrap intervals for checkpoint differences,
respecting repeated templates/paradigms, and report per-seed variation. Multiple
fine-tuning seeds on one pretrained model do not measure pretraining variance.

## Provisional achievement targets

These are proposed goals, not forecasts or published results for this setup.
Calibrate them after measuring today's checkpoint; do not choose evaluation items
after looking at which ones the model gets right.

- First success: clearly improve BLiMP over random and a same-data n-gram baseline,
  with a confidence interval on the paired difference excluding zero. A provisional
  raw-BLiMP milestone is 60%; 65-70% is a stretch for the current size and budget.
- GPT-1 success: pretraining improves SST-2 by at least 5 absolute percentage
  points over the tuned random-initialized model, consistently across seeds.
  Around 80% SST-2 accuracy is a provisional practical target, not sufficient by
  itself if the simple baseline already achieves it. Then confirm transfer on MRPC.
- Generation success: at least half of a fixed simple-story panel is judged
  grammatical, relevant, and locally consistent, with results separated by domain.
  Treat this as an agreed human rubric rather than an externally calibrated score.
- World knowledge: report EWoK by domain. Reliable gains over the reference chance
  baseline would be interesting; do not require large gains to justify continuing.

No single composite score: a model can improve grammar while failing at context
tracking, or improve story quality while losing encyclopedic predictive accuracy.

## What this Mac can afford

The current 27,368,448-parameter model sustained about 21K-22.6K training tokens/s
on MPS. The resumed run processed 18,366,464 tokens in 833.57 seconds. Our training
corpus contains 100,000,737 BPE tokens and **60,045,358 whitespace-separated words**
(measured locally; not an official BabyLM word count).

| Total token presentations | Approximate pure training time, current model |
| --- | ---: |
| 100M | 1.2-1.3 hours |
| 300M | 3.7-4.0 hours |
| 500M | 6.1-6.6 hours |
| 1B | 12.3-13.2 hours |
| 3B | 36.9-39.7 hours |

These are extrapolations, not sustained multi-day measurements, and exclude extra
evaluation/data-preparation overhead. Log actual time, throughput, model/data
sizes, precision, context, effective batch, and memory. MPS current/driver memory
figures are not peak-memory measurements. Record the exact chip and RAM before
making cross-machine claims. Avoid concurrent training or evaluation during speed
benchmarks. A 60M or 125M model needs its own pilot; parameter-count ratios are
not reliable runtime or memory predictions.

1B presentations over our existing corpus means roughly ten corpus-equivalents
of sampling, not 1B unique tokens. To investigate data limits, compare fresh
500M-token data against repeated 100M-token data at matched exposure. At the
observed encoding density, 500M unique tokens are roughly 2 GB of plain text plus
1 GB of uint16 IDs; allow extra space for metadata, preparation, and checkpoints.

## Experiment sequence

**A. Establish a baseline (one afternoon).** Implement and validate scoring;
evaluate the saved checkpoints. Continue the existing Wikipedia run to 100M and
300M presentations, preserving both checkpoints. Keep its existing 300M learning-
rate horizon so earlier checkpoints are comparable within that run. A 100M
checkpoint with this schedule is an intermediate model, not an optimally scheduled
100M-budget model.

**B. Test data choice (roughly 12 training hours, plus preparation/evaluation).**
Run three fresh 27M models for 300M presentations each: Wikipedia, TinyStories,
and a 50/50 token mixture. Use the same architecture, batch, schedule, seed, and a
single frozen tokenizer trained on a balanced training-only sample. Evaluate every
arm on both domains and the same benchmark suite. The existing Wikipedia-tokenizer
run remains a historical baseline; do not pretend it is a perfectly controlled
arm of this comparison. TinyStories-only is a specialization experiment; the mixed
arm tests whether coherent prose and broader coverage can coexist.

**C. Spend an overnight budget on the strongest recipe.** Compare 100M unique
tokens repeated with up to 500M unique tokens at a matched 1B presentations. Define
the 1B schedule before these runs; the current runner intentionally rejects changed
resume settings. Create a separately recorded experiment if extending the current
300M schedule, rather than silently changing checkpoint semantics.

**D. Explore model size within a weekend budget.** Benchmark 60M and roughly
original-GPT-sized 125M models before launching them. Compare quality under equal
4/12/48-hour training budgets and also equal-token controls where affordable.
Retrain the best small-model recipe with additional seeds before concluding a
small difference is real. Optimize/port the runtime later as a separate experiment.

Stop extending a run when held-out loss and capability scores plateau or regress;
investigate fresh data, domain mix, or optimization instead. A credible initial
result is a reproducible compute/quality curve with measurable grammar and transfer
gains, even if broad factual knowledge and reasoning remain weak.

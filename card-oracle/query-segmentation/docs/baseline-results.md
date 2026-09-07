# Demo baseline results — September 7, 2026

The local demo checkpoint is `artifacts/baseline-demo/`. Training is complete; no further tuning used the test results. This is an exploratory card-query segmentation baseline, with known failures on short queries and unfamiliar subject names.

## Model selection

All choices below used the same 8,000 training and 1,000 validation queries. The 1,000-query test split was evaluated only after final checkpoint selection. The seed data and committed splits are unchanged.

| Experiment | Best epoch | Validation span F1 | Validation query exact match | Training seconds |
| --- | ---: | ---: | ---: | ---: |
| Frozen encoder + linear head | 20 | 85.34% | 61.80% | 15.6 |
| Frozen encoder + 256-unit MLP | 4 | 88.48% | 68.50% | 15.6 |
| Last two blocks + final norm + MLP | 8 | 94.24% | 82.80% | 232.2 |
| All 22 blocks + final norm + MLP; embeddings frozen | 3 | 98.71% | 95.80% | 343.1 |

The final run initializes from the two-block checkpoint. Encoder learning rate is 2e-5, head learning rate 1e-4, batch size 16, AdamW weight decay 0.01, gradient clipping 1.0, float32, seed 42. It ran three epochs and selected epoch 3. The MLP is Linear(768,256), GELU, dropout 0.1, Linear(256,17). Encoder dropout stays disabled; only the head uses training dropout. There is no CRF or catalog postprocessing.

The initial linear-head experiment was too weak for the demo, so the plan was adjusted based on validation errors. The larger frozen head helped modestly; adapting encoder blocks resolved many subword-boundary errors. Frozen features were held in RAM only for head training, and never reused during encoder adaptation.

## Final synthetic test evaluation

- Exact-span precision: **95.34%**; recall: **97.58%**; micro F1: **96.45%**.
- Full-query exact match: **88.40%** (884/1,000).
- Macro F1 over supported labels: **96.91%**; BIO repairs: **42**.
- Quality evaluation ran on CPU using exactly the demo inference/decoding path.

| Label | Precision | Recall | F1 | Gold spans |
| --- | ---: | ---: | ---: | ---: |
| SUBJECT | 87.49% | 90.90% | 89.16% | 1000 |
| GAME | 87.83% | 100.00% | 93.52% | 303 |
| SET | 97.25% | 99.00% | 98.12% | 500 |
| CARD_NUMBER | 96.20% | 96.20% | 96.20% | 500 |
| YEAR | 98.43% | 100.00% | 99.21% | 1000 |
| GRADER | 99.40% | 100.00% | 99.70% | 500 |
| GRADE | 100.00% | 100.00% | 100.00% | 500 |
| CONDITION | 99.01% | 99.80% | 99.40% | 500 |

The validation-to-test gap is material: 95.8% versus 88.4% query exact match. Subject-disjoint splits have only about 40 held-out subject groups each, with many template variants per name; these examples are not independent samples of real traffic. Do not interpret the synthetic scores as production accuracy.

Representative test failures: `page one` is split into SUBJECT `page` and GAME `one`; `Needlessly Large Yordle` can split into two SUBJECT spans; card number `20/172` can have `20` mislabeled YEAR. Missing and extra spans for every query are retained in `reports/test/predictions.jsonl`, with per-label and raw/graded/set/number/length slices in `reports/test/metrics.json`.

## Hand-checked demo behavior

Eight manually specified queries were checked after model selection. Five matched exactly. This small, deliberately varied smoke check is not an accuracy estimate. It includes out-of-distribution short queries and one boundary example previously inspected in validation.

| Query | Result |
| --- | --- |
| `roronoa zoro psa 10 op01 2022` | All five spans correct |
| `2025 ogn ahri bgs 9.5` | All five spans correct |
| `OP01-025 Roronoa Zoro 2022 NM` | All four spans correct |
| `2022 brook lightly played one piece tcg op01-022` | All five spans correct |
| `2025 riftbound tcg ahri ogn psa 10` | All six spans correct |
| `pikachu 2023 pal near mint` | Incorrectly labels the initial `p` as GAME |
| `Charizard PSA 10` | Incorrectly labels `PSA` as SET |
| `pikachu` | Incorrectly labels the initial `p` as GAME |

The short-query failures are expected to require broader, independently annotated training/evaluation data. The qualified Pikachu failure also shows that the baseline still has genuine name-boundary errors. No rules were added to conceal these failures.

## Local latency

Apple M5 Pro, 64 GiB unified memory; float32 SDPA; CPU uses four PyTorch threads. Each device ran sequentially with ten warmups and 1,000 batch-one validation queries. MPS is synchronized around measurements. Model loading and downloads are excluded from warm request time. Load measurements use a fresh model instance with potentially warm filesystem caches and exclude Python import time.

| Device | Warm total p50 | p95 | p99 | Artifact load |
| --- | ---: | ---: | ---: | ---: |
| CPU | 14.04 ms | 18.77 ms | 24.76 ms | 0.189 s |
| MPS | 7.79 ms | 8.91 ms | 10.00 ms | 0.342 s |

Both devices meet the approximately 100 ms warm target on this workload. MPS is preferred when available. CPU and MPS returned identical spans on all eight hand-checked demo queries.

## Verification and reproducibility

- A 16-example intentional overfit test first recovered every span at epoch 662 using learning rate 0.01 and zero weight decay. This configuration is isolated in `configs/train-tiny.yaml`.
- Frozen encoder parameters were hashed before/after head training. Fine-tuning verifies frozen parameters remain unchanged and adapted parameters change.
- A fresh offline process reproduced the saving process’s logits with maximum absolute difference **0.0**, and exactly identical labels/spans on three validation queries. The verifier allows atol/rtol 1e-5.
- Automated checks cover offsets, BIO repairs, metrics, gradient isolation, artifact guards, and offline CLI behavior. See `tests/README.md`.
- Three existing generator lint issues were fixed; regeneration into a temporary directory remained byte-identical to every committed split.
- Base model revision: `8949b909ec900327062f0ebf497f51aef5e6f0c8`.
- Versions: Python 3.12.14; torch 2.14.0; transformers 5.16.1; tokenizers 0.23.2; safetensors 0.8.0.
- Dataset hashes and epoch histories are saved in each artifact’s `metadata.json`. Hardware-dependent numerical nondeterminism remains possible.
- Weights, caches, and routine reports remain ignored. Source, configuration, tests, and this summary are versioned; model artifacts must be trained or transferred separately.

## References consulted

- [ModernBERT model card](https://huggingface.co/answerdotai/ModernBERT-base): model architecture and downstream usage.
- [Transformers ModernBERT documentation](https://huggingface.co/docs/transformers/model_doc/modernbert): supported model and attention interfaces.
- [Transformers token classification](https://huggingface.co/docs/transformers/tasks/token_classification): subword label alignment and ignored targets. This project uses original-text offsets and supervises all entity subwords to preserve character boundaries.
- Installed Transformers 5.16.1 source was inspected for runtime compatibility: the old `reference_compile` argument is no longer accepted and is not used.

See [README.md](../README.md) for exact demo, training, evaluation, and benchmark commands.

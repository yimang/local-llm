# Quality, reload, and latency evaluation

`evaluate.py --model-dir DIR --input labeled.jsonl --output NEW_DIRECTORY` runs the
same inference decoder as the demo and writes exact-span metrics, per-label and
raw/graded/set/number/length slices, BIO repairs, and gold/predicted spans with
missing/extra errors. Current datasets measure synthetic query quality only.

`benchmark.py --model-dir DIR --device cpu --samples 1000 --output NEW_FILE.json`
measures ten warmups followed by batch-one requests from validation. Repeat with
`--device mps` using the same corpus. It reports synchronized p50/p95/p99 for
whole requests and components, plus offline model loading separately. Initial
model downloads and network service latency are excluded.

`verify_artifact.py --model-dir DIR --reference DIR/reload-reference.json` compares
fine-tuning-process logits with a freshly loaded model offline (atol/rtol 1e-5),
and requires identical token labels and character spans. `--write-reference`
can instead create a comparison fixture from an existing artifact and verify it
in a child process; this latter mode checks repeated loading, not training export.

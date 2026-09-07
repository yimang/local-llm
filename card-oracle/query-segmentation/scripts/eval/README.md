# Planned evaluation entrypoints

evaluate.py: checkpoint + labeled JSONL to aggregate metrics and per-query errors.
benchmark.py: checkpoint + queries to cold-load and synchronized warm latency.
Write outputs to reports/. Reuse inference decoding and exclude test data from
checkpoint selection. No evaluation scripts are implemented yet.

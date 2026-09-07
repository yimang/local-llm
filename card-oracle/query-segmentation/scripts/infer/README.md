# Offline demo

`segment.py` accepts exactly one of `--query TEXT`, `--interactive`, or
`--input FILE.jsonl`. Specify `--model-dir` to choose an artifact. Interactive mode
loads once and reads queries until EOF. `--timing` includes request components;
use the benchmark for warmed latency rather than the first inference request.

Single-query and interactive results use indented JSON for readability. Batch
results remain one JSON object per line (JSONL). Model loading diagnostics go to stderr. JSONL input
accepts optional IDs, copied to output. `--output` creates a new file; malformed or
overlength rows are reported to stderr with line/ID, other rows continue, and the
process exits nonzero. Whitespace-only queries return no spans without a forward
pass. Predictions preserve original text offsets and never fill missing fields.

# Checks

From the repository root:

```bash
uv run pytest -q card-oracle/query-segmentation/tests
uv run ruff check .
uv run ty check
```

Unit checks cover Unicode/repeated-substring/subword offsets, special/padding
tokens, invalid and overlapping spans, overlength rejection, BIO repairs, exact
metrics, schema/overwrite guards, and frozen/adapted gradient paths on a tiny real
ModernBERT constructed locally. They do not download a model.

CLI integration checks use the local demo artifact when it exists, otherwise
skip. They cover offline interactive inference, empty queries, text/offset round
trips, valid records after malformed rows, IDs, and overlength batch errors.
Actual training also hashes frozen encoder parameters before/after optimization.

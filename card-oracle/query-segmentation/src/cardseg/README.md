# Shared implementation

- `data.py`: schema validation, exact original-text tokenizer offsets, BIO targets,
  and dynamic padding. No truncation.
- `model.py`: ModernBERT with a linear or 256-unit MLP token head; SDPA/float32.
- `decode.py`: BIO-to-character spans shared by training, evaluation, and inference.
- `artifacts.py`: self-contained safetensors checkpoint save/load and label checks.
- `metrics.py`: exact span precision/recall/F1 and complete-query exact match.
- `inference.py`: offline model loading, prediction, and synchronized timing.

Entry scripts bootstrap this directory from their own location; PYTHONPATH is not
required. The repository-root uv environment and lockfile are shared.

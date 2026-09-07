# Experiment configurations

`train.yaml` is the original frozen-linear baseline. `train-mlp.yaml` adds a
256-unit GELU/dropout head. `train-tail.yaml` adapts the last two encoder layers
and final normalization. `train-full.yaml` adapts all 22 encoder blocks and final
normalization while retaining frozen token embeddings. These are explicit,
validation-driven experiments; test data is never used for model selection.

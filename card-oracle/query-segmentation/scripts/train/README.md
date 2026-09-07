# Planned training entrypoint

train.py will load seeds-independent synthetic JSONL, align character spans with
tokenizer offsets, train the frozen-encoder head, and save the best validation
checkpoint to artifacts/. Ignore padding/special tokens in loss; reject truncation
and unalignable spans. Support seed, device, learning rate, batch size, and epochs.
No training is implemented yet.

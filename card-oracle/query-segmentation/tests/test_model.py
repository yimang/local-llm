"""Exercise gradient isolation using a tiny real ModernBERT, without downloads."""

import torch
from cardseg.model import Segmenter
from transformers import ModernBertConfig, ModernBertModel


def test_frozen_and_adapted_gradient_paths(monkeypatch):
    config = ModernBertConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=128,
        local_attention=8,
        pad_token_id=0,
    )
    monkeypatch.setattr(
        "cardseg.model.AutoModel.from_pretrained",
        lambda *args, **kwargs: ModernBertModel(config),
    )
    model = Segmenter("unused", head_hidden=16)
    model.train()
    ids = torch.tensor([[1, 2, 3, 4]])
    mask = torch.ones_like(ids)
    assert not model.encoder.training
    assert model.head.training
    model(ids, mask).sum().backward()
    assert all(p.grad is None for p in model.encoder.parameters())
    assert any(p.grad is not None for p in model.head.parameters())
    model.zero_grad()
    model.trainable_layers = 1
    model.encoder.layers[-1].requires_grad_(True)
    model.encoder.final_norm.requires_grad_(True)
    model(ids, mask).sum().backward()
    assert all(p.grad is None for p in model.encoder.layers[0].parameters())
    assert any(p.grad is not None for p in model.encoder.layers[1].parameters())
    model.eval()
    assert not model.features(ids, mask).requires_grad

"""GPT-1-style post-LayerNorm causal decoder, with tied token embeddings."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 16000
    context: int = 512
    layers: int = 6
    width: int = 512
    heads: int = 8
    dropout: float = 0.1


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.heads
        self.qkv = nn.Linear(config.width, 3 * config.width)
        self.projection = nn.Linear(config.width, config.width)
        self.attn_norm = nn.LayerNorm(config.width, eps=1e-5)
        self.mlp_norm = nn.LayerNorm(config.width, eps=1e-5)
        self.mlp = nn.Sequential(
            nn.Linear(config.width, 4 * config.width),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * config.width, config.width),
            nn.Dropout(config.dropout),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = (
            self.qkv(x)
            .view(batch, length, 3, self.heads, width // self.heads)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        attended = (
            F.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=True,
                dropout_p=self.dropout.p if self.training else 0.0,
            )
            .transpose(1, 2)
            .contiguous()
            .view(batch, length, width)
        )
        x = self.attn_norm(x + self.dropout(self.projection(attended)))
        return self.mlp_norm(x + self.mlp(x))


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.width % config.heads:
            raise ValueError("Width must be divisible by attention heads")
        self.config = config
        self.tokens = nn.Embedding(config.vocab_size, config.width)
        self.positions = nn.Embedding(config.context, config.width)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(*(Block(config) for _ in range(config.layers)))
        self.apply(self._init)

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def hidden(self, ids):
        if ids.shape[1] > self.config.context:
            raise ValueError("Input exceeds context length")
        positions = torch.arange(ids.shape[1], device=ids.device)
        return self.blocks(self.dropout(self.tokens(ids) + self.positions(positions)))

    def forward(self, ids, targets=None):
        logits = F.linear(self.hidden(ids), self.tokens.weight)
        loss = (
            None
            if targets is None
            else F.cross_entropy(
                logits.flatten(0, 1), targets.flatten(), ignore_index=-100
            )
        )
        return logits, loss

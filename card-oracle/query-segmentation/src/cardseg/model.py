"""ModernBERT with configurable token head and explicit gradient control."""

import torch
from torch import nn
from transformers import AutoModel

from cardseg.data import LABELS


def choose_device(request):
    available = torch.backends.mps.is_available()
    if request == "mps" and not available:
        raise ValueError("MPS explicitly requested but unavailable")
    return (
        "mps"
        if request == "auto" and available
        else ("cpu" if request == "auto" else request)
    )


class Segmenter(nn.Module):
    def __init__(self, source, local=False, head_hidden=0, revision=None):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            source,
            local_files_only=local,
            revision=revision,
            attn_implementation="sdpa",
            dtype=torch.float32,
        )
        self.encoder.requires_grad_(False)
        hidden = self.encoder.config.hidden_size
        self.head = (
            nn.Sequential(
                nn.Linear(hidden, head_hidden),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(head_hidden, len(LABELS)),
            )
            if head_hidden
            else nn.Linear(hidden, len(LABELS))
        )
        self.encoder.eval()
        self.trainable_layers = 0

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def features(self, input_ids, attention_mask):
        with torch.set_grad_enabled(self.training and self.trainable_layers > 0):
            return self.encoder(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state

    def forward(self, input_ids, attention_mask):
        return self.head(self.features(input_ids, attention_mask))

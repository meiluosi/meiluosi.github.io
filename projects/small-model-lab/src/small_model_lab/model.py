"""Pre-norm causal decoder, learned positions, GELU MLP and tied output weights."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, x):
        batch, length, width = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(batch, length, 3, self.heads, width // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attention = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        x = x + self.proj(attention.transpose(1, 2).reshape(batch, length, width))
        return x + self.mlp(self.norm2(x))


class Decoder(nn.Module):
    def __init__(self, architecture: dict):
        super().__init__()
        self.architecture = architecture
        width = architecture["hidden_size"]
        if width % architecture["attention_heads"]:
            raise ValueError("hidden_size must be divisible by attention_heads")
        self.tokens = nn.Embedding(architecture["vocab_size"], width)
        self.positions = nn.Embedding(architecture["context_length"], width)
        self.blocks = nn.Sequential(*[Block(width, architecture["attention_heads"]) for _ in range(architecture["layers"])])
        self.norm = nn.LayerNorm(width)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, ids):
        if ids.size(1) > self.architecture["context_length"]:
            raise ValueError("Input exceeds model context")
        x = self.tokens(ids) + self.positions(torch.arange(ids.size(1), device=ids.device))
        return F.linear(self.norm(self.blocks(x)), self.tokens.weight)


def masked_loss(logits, labels):
    if not (labels != -100).any():
        raise ValueError("No supervised tokens")
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)


def completion_logps(model, ids, labels):
    mask = labels != -100
    logits = model(ids).log_softmax(-1)
    values = logits.gather(-1, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    # DPO uses the SUM of completion log probabilities, not length-normalized loss.
    return (values * mask).sum(-1)


def dpo_loss(chosen, rejected, reference_chosen, reference_rejected, beta):
    margin = (chosen - rejected) - (reference_chosen - reference_rejected)
    return -F.logsigmoid(beta * margin).mean()

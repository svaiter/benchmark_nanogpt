"""GPT-2 training model — modded-nanogpt architecture.

Mirrors the model at modded-nanogpt commit 844e5fdb (RoPE, functional RMSNorm
without learnable params, fixed-scalar attention down-scaling, no learned
positional embeddings, default PyTorch init). The README of that commit
attributes the 2x token-efficiency over llm.c to exactly this combination.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


# -----------------------------------------------------------------------------
# Rotary positional embeddings + RMSNorm helpers


class Rotary(nn.Module):
    """Rotary positional embedding with a precomputed cos/sin cache.

    The cache is built at init time for the maximum sequence length so the
    forward pass has no Python branches — keeps `torch.compile(fullgraph=True)`
    happy.
    """

    def __init__(self, dim, max_seq_len=1024, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        # (1, T, 1, dim/2) broadcasts against q/k of shape (B, T, n_head, dim/2)
        self.register_buffer("cos_cached", freqs.cos()[None, :, None, :])
        self.register_buffer("sin_cached", freqs.sin()[None, :, None, :])

    def forward(self, x):
        T = x.shape[1]
        return self.cos_cached[:, :T], self.sin_cached[:, :T]


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention layout: (B, T, n_head, head_dim)
    d = x.shape[3] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


def rmsnorm(x0, eps=1e-6):
    x = x0.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x.type_as(x0)


# -----------------------------------------------------------------------------
# Transformer blocks


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.rotary = Rotary(self.head_dim, max_seq_len=config.block_size)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim)
        q = q.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config.n_embd)
        # Fixed scalar down-scaling the attention output, replacing GPT-2's
        # special init for residual projections.
        self.attn_scale = 1.0 / math.sqrt(2 * config.n_layer)

    def forward(self, x):
        x = x + self.attn_scale * self.attn(rmsnorm(x))
        x = x + self.mlp(rmsnorm(x))
        return x


# -----------------------------------------------------------------------------
# Top-level model


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        # No wpe: positional information comes from RoPE inside attention.
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Tied input/output embedding (https://paperswithcode.com/method/weight-tying).
        self.transformer.wte.weight = self.lm_head.weight

    def to(self, **kwargs):
        if 'device' in kwargs:
            self.device = kwargs['device']
        return super().to(**kwargs)

    def forward(self, idx, targets=None, return_logits=True):
        x = self.transformer.wte(idx)

        for block in self.transformer.h:
            x = block(x)
        x = rmsnorm(x)

        if targets is not None:
            logits = self.lm_head(x).float()
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1),
                ignore_index=-1,
            )
        else:
            logits = self.lm_head(x[:, [-1], :]).float()
            loss = None

        return loss, logits

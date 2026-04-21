"""第2章: Single-block Transformer の最小実装.

本書 2.6 節の全体像を 100 行程度で再現したもの。
依存: torch のみ。

    $ python examples/ch02/tiny_transformer.py
"""

from __future__ import annotations
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.g = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.g


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d: int, n_heads: int) -> None:
        super().__init__()
        assert d % n_heads == 0
        self.h = n_heads
        self.dk = d // n_heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.h, self.dk).transpose(1, 3)
        q, k, v = qkv.unbind(dim=2)  # each: (B, h, T, dk)
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.dk)
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = scores.softmax(-1)
        out = attn @ v  # (B, h, T, dk)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.o(out)


class SwiGLU(nn.Module):
    def __init__(self, d: int, d_ff: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(d, d_ff, bias=False)
        self.w2 = nn.Linear(d, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(self.w1(x) * F.silu(self.w2(x)))


class Block(nn.Module):
    def __init__(self, d: int, n_heads: int, d_ff: int) -> None:
        super().__init__()
        self.norm1 = RMSNorm(d)
        self.attn = MultiHeadSelfAttention(d, n_heads)
        self.norm2 = RMSNorm(d)
        self.ffn = SwiGLU(d, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class TinyLM(nn.Module):
    def __init__(self, vocab: int = 1000, d: int = 64, n_heads: int = 4,
                 d_ff: int = 128, n_layers: int = 2) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([Block(d, n_heads, d_ff) for _ in range(n_layers)])
        self.final_norm = RMSNorm(d)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        h = self.embed(ids)
        for blk in self.blocks:
            h = blk(h)
        h = self.final_norm(h)
        return h @ self.embed.weight.T  # weight tying


def main() -> None:
    torch.manual_seed(0)
    model = TinyLM()
    ids = torch.randint(0, 1000, (2, 10))
    logits = model(ids)
    print("input :", ids.shape)
    print("logits:", logits.shape)
    params = sum(p.numel() for p in model.parameters())
    print(f"params: {params:,}")


if __name__ == "__main__":
    main()

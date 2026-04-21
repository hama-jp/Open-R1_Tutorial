"""第4章: RoPE のスコア行列を可視化.

同一ベクトル q = k を位置 0..T-1 に並べて内積行列を計算し、
RoPE あり/なしでどう変わるかをテキストで観察する。

    $ python examples/ch04/rope_demo.py
"""
from __future__ import annotations
import math

import torch


def rope_cache(T: int, d: int, base: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    i = torch.arange(0, d, 2).float()
    theta = 1.0 / (base ** (i / d))
    m = torch.arange(T).float()
    freqs = torch.outer(m, theta)  # (T, d/2)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., ::2], x[..., 1::2]
    rx1 = x1 * cos - x2 * sin
    rx2 = x1 * sin + x2 * cos
    return torch.stack((rx1, rx2), dim=-1).flatten(-2)


def matrix(with_rope: bool, T: int = 16, d: int = 8) -> torch.Tensor:
    torch.manual_seed(0)
    q = torch.randn(1, d).expand(T, d).clone()
    k = q.clone()
    if with_rope:
        cos, sin = rope_cache(T, d)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
    return q @ k.T / math.sqrt(d)


def pretty(m: torch.Tensor) -> None:
    for row in m:
        print(" ".join(f"{v:+.2f}" for v in row))


def main() -> None:
    print("=== With RoPE ===")
    pretty(matrix(True))
    print()
    print("=== Without RoPE (flat) ===")
    pretty(matrix(False))


if __name__ == "__main__":
    main()

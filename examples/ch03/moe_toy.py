"""第3章: Toy MoE とロードバランス.

4 エキスパート, top_k=1 の素朴な MoE を作り、
1000 サンプルの割り当て分布を観察する。

    $ python examples/ch03/moe_toy.py
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ToyMoE(nn.Module):
    def __init__(self, d: int = 32, n_experts: int = 4, top_k: int = 1,
                 aux_loss: float = 0.0) -> None:
        super().__init__()
        self.router = nn.Linear(d, n_experts, bias=False)
        self.experts = nn.ModuleList([nn.Linear(d, d) for _ in range(n_experts)])
        self.top_k = top_k
        self.n_experts = n_experts
        self.aux_loss = aux_loss
        self._last_aux: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, d)
        gate = self.router(x).softmax(-1)  # (B, N)
        topv, topi = gate.topk(self.top_k, dim=-1)
        out = torch.zeros_like(x)
        counts = torch.zeros(self.n_experts, device=x.device)
        for k in range(self.top_k):
            idx = topi[:, k]
            w = topv[:, k].unsqueeze(-1)
            for e in range(self.n_experts):
                mask = idx == e
                if mask.any():
                    out[mask] += w[mask] * self.experts[e](x[mask])
                    counts[e] += mask.sum()
        # 補助 loss: ゲート平均と割り当て平均の内積
        self._last_aux = (gate.mean(0) * counts / x.size(0)).sum() * self.n_experts
        return out


def main() -> None:
    torch.manual_seed(0)
    for aux in (0.0, 0.01):
        model = ToyMoE(aux_loss=aux)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        counts_hist: list[torch.Tensor] = []
        for step in range(200):
            x = torch.randn(128, 32)
            y = model(x)
            loss = y.pow(2).mean()
            if aux > 0:
                loss = loss + aux * model._last_aux  # type: ignore[arg-type]
            opt.zero_grad(); loss.backward(); opt.step()
        # 推論時の専門家分布
        x = torch.randn(1000, 32)
        with torch.no_grad():
            gate = model.router(x).softmax(-1)
            used = gate.argmax(-1).bincount(minlength=4)
        print(f"aux={aux}: 専門家別トークン数 = {used.tolist()}")


if __name__ == "__main__":
    main()

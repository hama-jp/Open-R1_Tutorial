"""第6章: CartPole で REINFORCE と PPO を比較.

gymnasium と最小の PPO 実装で、方策勾配のクリッピング効果を観察する。
依存: torch, gymnasium

    $ pip install gymnasium torch
    $ python examples/ch06/cartpole_ppo.py
"""
from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F


class Policy(nn.Module):
    def __init__(self, obs_dim: int = 4, n_actions: int = 2, hid: int = 64) -> None:
        super().__init__()
        self.pi = nn.Sequential(nn.Linear(obs_dim, hid), nn.Tanh(), nn.Linear(hid, n_actions))
        self.v = nn.Sequential(nn.Linear(obs_dim, hid), nn.Tanh(), nn.Linear(hid, 1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.pi(x), self.v(x).squeeze(-1)


def collect(env: gym.Env, policy: Policy, n_steps: int = 1024) -> dict:
    obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []
    obs, _ = env.reset()
    for _ in range(n_steps):
        x = torch.as_tensor(obs, dtype=torch.float32)
        logits, v = policy(x)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        obs2, r, done, trunc, _ = env.step(a.item())
        obs_buf.append(x); act_buf.append(a); logp_buf.append(dist.log_prob(a))
        rew_buf.append(r); val_buf.append(v); done_buf.append(done or trunc)
        obs = obs2
        if done or trunc:
            obs, _ = env.reset()
    # Discounted return
    rets, G = [], 0.0
    for r, d in zip(reversed(rew_buf), reversed(done_buf)):
        G = r + 0.99 * G * (0.0 if d else 1.0)
        rets.insert(0, G)
    return dict(
        obs=torch.stack(obs_buf),
        act=torch.stack(act_buf),
        logp_old=torch.stack(logp_buf).detach(),
        ret=torch.tensor(rets),
        val=torch.stack(val_buf).detach(),
    )


def train(algo: str, epochs: int = 40) -> list[float]:
    env = gym.make("CartPole-v1")
    pol = Policy()
    opt = torch.optim.Adam(pol.parameters(), lr=3e-4)
    log: list[float] = []
    for ep in range(epochs):
        data = collect(env, pol)
        adv = data["ret"] - data["val"]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        for _ in range(4):  # minibatch epochs
            logits, v = pol(data["obs"])
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(data["act"])
            if algo == "reinforce":
                loss_pi = -(logp * adv).mean()
            else:  # ppo
                rho = (logp - data["logp_old"]).exp()
                clip = rho.clamp(0.8, 1.2)
                loss_pi = -torch.minimum(rho * adv, clip * adv).mean()
            loss_v = F.mse_loss(v, data["ret"])
            (loss_pi + 0.5 * loss_v).backward()
            opt.step(); opt.zero_grad()
        mean_ret = data["ret"].mean().item()
        log.append(mean_ret)
        print(f"[{algo}] epoch {ep:3d}  mean_ret={mean_ret:7.2f}")
    return log


if __name__ == "__main__":
    torch.manual_seed(0)
    for algo in ("reinforce", "ppo"):
        print(f"\n### Training with {algo} ###")
        train(algo, epochs=20)

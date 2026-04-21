"""第12章: ソフト報酬バリアント (編集距離で部分点).

binary の eval_reward に対し、解答式の出力値が target にどれだけ近いかを
1 / (1 + |v - t|) の形でスカラー化することで、学習序盤でも勾配が流れやすくする。
"""
from __future__ import annotations
import re

PATTERN = re.compile(r"<think>.*?</think>\s*<answer>(.+?)</answer>", re.DOTALL)
ALLOWED = set("0123456789+-*/() ")


def soft_eval_reward(completions, numbers=None, target=None, **_):
    out = []
    for c, nums, t in zip(completions, numbers, target):
        m = PATTERN.search(c)
        if not m:
            out.append(0.0); continue
        expr = m.group(1).strip()
        if any(ch not in ALLOWED for ch in expr):
            out.append(0.0); continue
        used = sorted(int(x) for x in re.findall(r"\d+", expr))
        if used != sorted(nums):
            out.append(0.0); continue
        try:
            v = eval(expr, {"__builtins__": {}})
        except Exception:
            out.append(0.0); continue
        out.append(1.0 / (1.0 + abs(v - t)))
    return out


if __name__ == "__main__":
    print(soft_eval_reward(
        ["<think>x</think><answer>(1+2)*3</answer>"],
        numbers=[[1, 2, 3, 3]], target=[10],
    ))

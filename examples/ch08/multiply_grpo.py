"""第8章: 2桁×2桁の掛け算 GRPO.

正解報酬 + 書式報酬の組み合わせ方のサンプル。
依存: transformers, trl, datasets

    $ python examples/ch08/multiply_grpo.py
"""
from __future__ import annotations

import random
import re

from datasets import Dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer


MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
FORMAT_RE = re.compile(r"<think>.*?</think>\s*<answer>(-?\d+)</answer>", re.DOTALL)


def format_reward(completions, **_):
    return [0.2 if FORMAT_RE.search(c) else 0.0 for c in completions]


def accuracy_reward(completions, answer=None, **_):
    out = []
    for c, gold in zip(completions, answer):
        m = FORMAT_RE.search(c)
        if m is None:
            out.append(0.0); continue
        try:
            pred = int(m.group(1))
        except ValueError:
            out.append(0.0); continue
        out.append(1.0 if pred == gold else 0.0)
    return out


def make_ds(n: int, tok):
    random.seed(0)
    rows = []
    for _ in range(n):
        a = random.randint(10, 99)
        b = random.randint(10, 99)
        user = (
            f"{a} × {b} を計算せよ。"
            "思考過程は <think></think>、最終答えは <answer>数値</answer> で囲んで回答せよ。"
        )
        rows.append({
            "prompt": tok.apply_chat_template(
                [{"role": "user", "content": user}],
                tokenize=False, add_generation_prompt=True,
            ),
            "answer": a * b,
        })
    return Dataset.from_list(rows)


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL)
    ds = make_ds(500, tok)

    args = GRPOConfig(
        output_dir="out-mul-grpo",
        num_generations=8,
        max_prompt_length=128,
        max_completion_length=256,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        beta=0.04,
        max_steps=200,
        logging_steps=2,
        report_to="none",
        bf16=True,
    )

    trainer = GRPOTrainer(
        model=MODEL,
        reward_funcs=[format_reward, accuracy_reward],
        args=args,
        train_dataset=ds,
    )
    trainer.train()


if __name__ == "__main__":
    main()

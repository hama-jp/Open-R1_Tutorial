"""第12章: Countdown で GRPO を走らせる.

依存: transformers, trl, datasets
    $ python examples/ch12/grpo_countdown.py
"""
from __future__ import annotations
import re

from datasets import load_dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer


MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
PATTERN = re.compile(r"<think>.*?</think>\s*<answer>(.+?)</answer>", re.DOTALL)
ALLOWED = set("0123456789+-*/() ")


def format_reward(completions, **_):
    return [0.1 if PATTERN.search(c) else 0.0 for c in completions]


def eval_reward(completions, numbers=None, target=None, **_):
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
        out.append(1.0 if v == t else 0.0)
    return out


def build_prompt_fn(tok):
    def fn(ex):
        user = (
            f"以下の数字 {ex['numbers']} を **すべて 1 回ずつ** 使い、"
            f"四則演算で {ex['target']} を作る式を求めよ。"
            "\n思考過程は <think></think>、最終式は <answer></answer> で囲むこと。"
        )
        ex["prompt"] = tok.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True,
        )
        return ex
    return fn


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL)
    ds = load_dataset("json", data_files="countdown_train.jsonl", split="train")
    ds = ds.map(build_prompt_fn(tok))

    args = GRPOConfig(
        output_dir="out-countdown",
        num_generations=8,
        max_prompt_length=256,
        max_completion_length=768,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        beta=0.04,
        bf16=True,
        logging_steps=5,
        max_steps=500,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=MODEL,
        args=args,
        reward_funcs=[format_reward, eval_reward],
        train_dataset=ds,
    )
    trainer.train()


if __name__ == "__main__":
    main()

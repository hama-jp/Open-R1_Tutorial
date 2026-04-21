"""第7章: 応答長を 100 字に近づける GRPO.

trl.GRPOTrainer を使った最小実装。GPU 1 枚で動く。
依存: transformers >= 4.45, trl >= 0.14, datasets

    $ pip install transformers trl datasets accelerate
    $ python examples/ch07/grpo_length.py
"""
from __future__ import annotations

from datasets import Dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer


MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def reward_len(completions, **_):
    """応答の文字数が 100 に近いほど 1.0 に近い報酬を返す."""
    return [max(0.0, 1.0 - abs(len(c) - 100) / 100.0) for c in completions]


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL)
    prompts = [
        "日本の首都について1段落で説明してください。",
        "Python のリスト内包表記の使い方を短く説明してください。",
        "深層学習の勾配消失問題を一般向けに説明してください。",
        "RoPE とは何か、数式を使わずに短く説明してください。",
    ] * 16
    rows = [
        {"prompt": tok.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False, add_generation_prompt=True,
        )}
        for p in prompts
    ]
    ds = Dataset.from_list(rows)

    args = GRPOConfig(
        output_dir="out-grpo-len",
        num_generations=4,
        max_prompt_length=128,
        max_completion_length=256,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=5e-6,
        beta=0.04,
        epsilon=0.2,
        max_steps=50,
        logging_steps=1,
        report_to="none",
        bf16=True,
    )

    trainer = GRPOTrainer(
        model=MODEL,
        reward_funcs=[reward_len],
        args=args,
        train_dataset=ds,
    )
    trainer.train()


if __name__ == "__main__":
    main()

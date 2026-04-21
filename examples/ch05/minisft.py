"""第5章: 10 件データでのミニ SFT.

Qwen2.5-0.5B-Instruct を 10 件の (instruction, response) で 1 epoch だけ学習する。
依存: transformers, trl, datasets, accelerate, torch

    $ pip install transformers trl datasets accelerate
    $ python examples/ch05/minisft.py
"""
from __future__ import annotations
import json
import pathlib

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


SAMPLES = [
    ("こんにちは", "こんにちは！今日はどんなお手伝いができますか？"),
    ("Pythonでフィボナッチ数列を書いて", "```python\ndef fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        yield a\n        a, b = b, a + b\n```"),
    ("3+4は？", "<think>3 と 4 を足すので 7。</think>7 です。"),
    ("素数を3つ挙げて", "<think>小さい順に 2, 3, 5 が素数。</think>2, 3, 5 です。"),
    ("2*7は？", "<think>2の7倍は14。</think>14 です。"),
    ("0.1+0.2の実数値は？", "<think>浮動小数点誤差で 0.30000000000000004 になる。</think>0.30000000000000004 です。"),
    ("円周率の小数第3位まで教えて", "<think>π ≒ 3.14159...</think>3.141 です。"),
    ("1/4をパーセントで表すと？", "<think>1/4 = 0.25 = 25%</think>25% です。"),
    ("2進数の1011を10進数にすると？", "<think>1*8 + 0*4 + 1*2 + 1*1 = 11</think>11 です。"),
    ("三角形の内角の和は？", "<think>平面幾何の定理により 180°。</think>180 度です。"),
]


def to_chat(rows: list[tuple[str, str]]) -> list[dict]:
    return [
        {"messages": [
            {"role": "user", "content": u},
            {"role": "assistant", "content": a},
        ]}
        for u, a in rows
    ]


def main() -> None:
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )
    ds = Dataset.from_list(to_chat(SAMPLES))

    config = SFTConfig(
        output_dir="out-minisft",
        num_train_epochs=3,
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        logging_steps=1,
        max_seq_length=256,
        bf16=torch.cuda.is_available(),
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        args=config,
        tokenizer=tok,
        train_dataset=ds,
    )
    trainer.train()
    trainer.save_model("out-minisft/final")
    print("Saved to out-minisft/final")


if __name__ == "__main__":
    main()

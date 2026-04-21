"""第9章: Rejection Sampling で正解 CoT だけ残す.

手元の Distill-1.5B で問題を複数回解かせ、正解だけを学習データに残す。
依存: transformers, torch

    $ python examples/ch09/reject_sample.py
"""
from __future__ import annotations

import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

PROBLEMS = [
    ("12 × 13 = ?", 156),
    ("47 + 28 = ?", 75),
    ("100 - 37 = ?", 63),
    ("81 ÷ 9 = ?", 9),
    ("7 の 3 乗は?", 343),
]


def extract_answer(text: str) -> int | None:
    m = re.search(r"\\boxed\{(-?\d+)\}", text) or re.search(r"答え\s*[:：]?\s*(-?\d+)", text)
    try:
        return int(m.group(1)) if m else None
    except ValueError:
        return None


def main(n_samples_per_problem: int = 4) -> None:
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    kept = []
    for q, gold in PROBLEMS:
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": q + " 答えは \\boxed{数値} の形式で出してください。"}],
            tokenize=False, add_generation_prompt=True,
        )
        ids = tok(prompt, return_tensors="pt").to(model.device)
        out = model.generate(
            **ids, max_new_tokens=512,
            num_return_sequences=n_samples_per_problem,
            do_sample=True, temperature=0.6, top_p=0.95,
        )
        for seq in out:
            text = tok.decode(seq, skip_special_tokens=True)
            pred = extract_answer(text)
            if pred == gold:
                kept.append({"question": q, "answer": gold, "cot": text})
                break  # 正解1つ見つかれば OK
    print(f"kept {len(kept)} / {len(PROBLEMS)} problems")
    with open("kept.jsonl", "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

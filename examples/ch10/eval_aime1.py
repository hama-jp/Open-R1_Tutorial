"""第10章: AIME 1 問を Distill モデルで解いてみる.

依存: transformers, torch, datasets

    $ python examples/ch10/eval_aime1.py
"""
from __future__ import annotations
import re

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


def extract(text: str) -> str | None:
    m = re.search(r"\\boxed\{([^}]*)\}", text)
    return m.group(1).strip() if m else None


def main() -> None:
    ds = load_dataset("HuggingFaceH4/aime_2024", split="train")
    sample = ds[0]
    print("Problem:", sample["problem"][:200], "...")
    print("Gold   :", sample["answer"])

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    for temp in (0.0, 0.6):
        prompt = tok.apply_chat_template(
            [{"role": "user",
              "content": sample["problem"] + "\n最終解答は \\boxed{数値} の形で答えてください。"}],
            tokenize=False, add_generation_prompt=True,
        )
        ids = tok(prompt, return_tensors="pt").to(model.device)
        out = model.generate(
            **ids, max_new_tokens=4096,
            do_sample=(temp > 0), temperature=max(temp, 0.01), top_p=0.95,
        )
        text = tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"\n--- temp={temp} ---")
        print("Pred :", extract(text))
        print("Tail :", text[-300:])


if __name__ == "__main__":
    main()

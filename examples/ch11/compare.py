"""第11章: SFT 学習前後の推論を比較."""
from __future__ import annotations
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def ask(path: str, q: str) -> str:
    tok = AutoTokenizer.from_pretrained(path)
    m = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    if torch.cuda.is_available():
        m = m.to("cuda")
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True,
    )
    ids = tok(prompt, return_tensors="pt").to(m.device)
    out = m.generate(**ids, max_new_tokens=200, do_sample=False)
    return tok.decode(out[0], skip_special_tokens=True)


def main() -> None:
    q = "17 + 28 はいくつ？"
    print("BEFORE:")
    print(ask("Qwen/Qwen2.5-0.5B-Instruct", q))
    print("\nAFTER :")
    print(ask("out-sft/final", q))


if __name__ == "__main__":
    main()

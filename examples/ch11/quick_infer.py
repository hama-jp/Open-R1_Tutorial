"""第11章: Distill-1.5B で思考付き推論をすぐ試す."""
from __future__ import annotations
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    mid = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    tok = AutoTokenizer.from_pretrained(mid)
    model = AutoModelForCausalLM.from_pretrained(
        mid,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    prompt = tok.apply_chat_template(
        [{"role": "user", "content": "1/3 + 1/6 を計算して理由も述べてください。"}],
        tokenize=False, add_generation_prompt=True,
    )
    ids = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=1024, do_sample=True,
                        temperature=0.6, top_p=0.95)
    print(tok.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()

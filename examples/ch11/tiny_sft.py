"""第11章: Qwen2.5-0.5B の SFT（tiny_addition.jsonl を使う）."""
from __future__ import annotations
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def main() -> None:
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    ds = load_dataset("json", data_files="tiny_addition.jsonl", split="train")

    config = SFTConfig(
        output_dir="out-sft",
        num_train_epochs=1,
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        bf16=torch.cuda.is_available(),
        logging_steps=10,
        max_seq_length=512,
        packing=False,
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        args=config,
        tokenizer=tok,
        train_dataset=ds,
    )
    trainer.train()
    trainer.save_model("out-sft/final")
    print("Saved model to out-sft/final")


if __name__ == "__main__":
    main()

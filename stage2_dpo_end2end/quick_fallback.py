"""
快速回退方案: 如果 500 条训练失败, 用 50 条 1 epoch 5 分钟内出结果
- 改 max_length=256, LoRA r=4, 50 条
- 目的: 保证流水线有结果, 不让用户白等

用法:
  python quick_fallback.py
"""
import os
import sys
import time
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def main():
    output_dir = os.path.join(ROOT, "outputs", "dpo_quick_50")
    os.makedirs(output_dir, exist_ok=True)
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    cache_model = os.path.join(ROOT, "models")
    cache_data = os.path.join(ROOT, "data")

    print("[quick] loading tokenizer ...")
    tok = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("[quick] loading 50 samples ...")
    ds = load_dataset(
        "HuggingFaceH4/ultrafeedback_binarized",
        split="train_prefs", cache_dir=cache_data,
    ).select(range(50))

    def to_dpo(ex):
        ch = ex["chosen"][-1]["content"] if isinstance(ex["chosen"], list) else ex["chosen"]
        rj = ex["rejected"][-1]["content"] if isinstance(ex["rejected"], list) else ex["rejected"]
        return {"prompt": ex["prompt"], "chosen": ch, "rejected": rj}

    ds = ds.map(to_dpo, remove_columns=ds.column_names)
    split = ds.train_test_split(test_size=10, seed=42)

    print("[quick] loading model ...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, cache_dir=cache_model)
    model.config.pad_token_id = tok.pad_token_id
    print(f"[quick] model loaded in {time.time()-t0:.1f}s")

    peft = LoraConfig(r=4, lora_alpha=8, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM", target_modules=["q_proj", "v_proj"])

    args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=1, per_device_train_batch_size=1,
        per_device_eval_batch_size=1, gradient_accumulation_steps=1,
        learning_rate=5e-5, lr_scheduler_type="cosine", warmup_steps=2,
        logging_steps=2, save_strategy="no", eval_strategy="no",
        report_to="none", beta=0.1, loss_type="sigmoid",
        max_length=256, bf16=False, fp16=False, gradient_checkpointing=False,
        dataloader_num_workers=0, remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model, ref_model=None, args=args,
        train_dataset=split["train"], eval_dataset=split["test"],
        processing_class=tok, peft_config=peft,
    )

    print("[quick] training ...")
    t0 = time.time()
    trainer.train()
    print(f"[quick] training done in {(time.time()-t0)/60:.1f} min")

    trainer.save_model(output_dir)
    tok.save_pretrained(output_dir)
    print(f"[quick] saved to {output_dir}")
    print(f"\n[next] python stage2_dpo_end2end/eval_dpo.py --dpo_model {output_dir}")


if __name__ == "__main__":
    main()

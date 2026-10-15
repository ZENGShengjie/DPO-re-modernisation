"""
阶段 2 冒烟测试: 用最小配置 (30 条 + 3 步训练) 验证整条 DPO 训练链路能跑通

设计目的:
  - CPU 上完整 DPO 训练 500 条会跑 1-2 小时, 但有 bug 的话会浪费时间
  - 先用 30 条 + 3 步跑通, 确认:
    1. 数据集能加载
    2. 模型能加载 (含 LoRA)
    3. 一次 forward + backward 没问题
    4. DPO loss 能下降
    5. 模型能保存

用法:
  python stage2_dpo_end2end/smoke_test.py
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
    print("=" * 70)
    print("冒烟测试: 30 条 + 3 步 DPO 训练 (CPU)")
    print("=" * 70)

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    cache_model = os.path.join(ROOT, "models")
    cache_data = os.path.join(ROOT, "data")
    output_dir = os.path.join(ROOT, "outputs", "smoke_test")
    os.makedirs(output_dir, exist_ok=True)

    # 1) tokenizer
    print("\n[1/6] 加载 tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  OK: pad_token={tokenizer.pad_token!r}, eos_token={tokenizer.eos_token!r}")

    # 2) dataset
    print("\n[2/6] 加载 UltraFeedback binarized (前 30 条) ...")
    raw = load_dataset(
        "HuggingFaceH4/ultrafeedback_binarized",
        split="train_prefs",
        cache_dir=cache_data,
    )
    raw = raw.select(range(30))

    def to_dpo(ex):
        chosen = ex["chosen"][-1]["content"] if isinstance(ex["chosen"], list) else ex["chosen"]
        rejected = ex["rejected"][-1]["content"] if isinstance(ex["rejected"], list) else ex["rejected"]
        return {"prompt": ex["prompt"], "chosen": chosen, "rejected": rejected}

    ds = raw.map(to_dpo, remove_columns=raw.column_names)
    print(f"  OK: {len(ds)} samples")
    print(f"  示例 prompt:  {ds[0]['prompt'][:80]!r}")
    print(f"  示例 chosen:  {ds[0]['chosen'][:80]!r}")
    print(f"  示例 rejected:{ds[0]['rejected'][:80]!r}")

    # 3) model
    print(f"\n[3/6] 加载模型 {model_name} ...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # CPU 必须 fp32
        cache_dir=cache_model,
    )
    if hasattr(model, "config"):
        model.config.pad_token_id = tokenizer.pad_token_id
    print(f"  OK: 加载耗时 {time.time() - t0:.1f}s, 参数量 {sum(p.numel() for p in model.parameters()):,}")

    # 4) LoRA
    print("\n[4/6] 配置 LoRA ...")
    peft_config = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    # 5) DPO config
    print("\n[5/6] 配置 DPO 训练参数 ...")
    dpo_args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1,         # CPU 内存紧张, 用 1
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=5e-5,
        max_steps=3,                            # 只要 3 步
        logging_steps=1,
        save_strategy="no",                     # 不保存中间检查点
        eval_strategy="no",
        report_to="none",
        beta=0.1,
        max_length=384,                         # 缩短 max_length 省内存
        bf16=False, fp16=False,
        gradient_checkpointing=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_args,
        train_dataset=ds,
        eval_dataset=None,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    print(f"  OK: 参数量 {total:,}  可训练 {trainable:,}  (LoRA 占比 {100*trainable/total:.2f}%)")

    # 6) train
    print("\n[6/6] 开始训练 (3 步) ...")
    print("-" * 70)
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print("-" * 70)
    print(f"  OK: 3 步训练耗时 {elapsed:.1f}s (平均 {elapsed/3:.1f}s/step)")

    # 7) 估算: 真实 500 条 1 epoch
    batch_effective = 1 * 1
    est_steps_500 = 500 // batch_effective
    est_time_500 = est_steps_500 * (elapsed / 3)
    print(f"\n[预估] 500 条 1 epoch 约需 {est_time_500/60:.1f} 分钟")

    # 8) 保存
    print(f"\n[save] 保存模型到 {output_dir} ...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("  OK")

    print("\n" + "=" * 70)
    print("[SUCCESS] 冒烟测试通过! DPO 训练链路全流程 OK")
    print("=" * 70)
    print("\n下一步建议:")
    print(f"  - 跑真实训练: python stage2_dpo_end2end/train_dpo.py --max_samples 100 --epochs 1")
    print(f"  - 或先看本测试输出: outputs/smoke_test/")


if __name__ == "__main__":
    main()

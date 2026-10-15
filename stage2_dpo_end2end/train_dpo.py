"""
阶段 2: DPO 端到端训练
- 模型: Qwen2.5-0.5B-Instruct (CPU 可跑, GPU 更快)
- 数据: UltraFeedback binarized (切前 500 条做快速实验)
- 框架: HuggingFace TRL DPOTrainer + PEFT (LoRA)

为什么用 TRL 而不是自己写 Trainer?
  - TRL 的 DPOTrainer 已经是工业标准, reference model 共享前向很高效
  - 自己写一个能 work 的 Trainer 是好练习, 但作为"学完公式后的第一次实战"应该先看工业代码长啥样
  - 阶段 1 的 dpo_losses.py 就是为了让你有能力"自己写一个"

CPU 训练预估 (Qwen2.5-0.5B + 500 条 + max_length=512):
  - 内存峰值: ~4-5 GB
  - 时间: 1 epoch 约 1-2 小时
  - 建议先用 100 条跑通流程, 再放大
"""

import argparse
import os
import sys

import torch
import yaml
from datasets import load_dataset, DatasetDict
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    p.add_argument("--max_samples", type=int, default=None,
                   help="override data.max_samples in config")
    p.add_argument("--epochs", type=float, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_and_prepare_dataset(cfg: dict, tokenizer, max_samples: int):
    """
    UltraFeedback binarized 格式:
      {
        'prompt': str,            # 用户提问
        'chosen': List[Dict],     # 多轮对话
        'rejected': List[Dict],
        ...
      }
    TRL 的 DPO trainer 接受:  {prompt, chosen, rejected} 三个字符串字段
    """
    print(f"[data] loading {cfg['data']['dataset_name']} ...")
    ds = load_dataset(
        cfg["data"]["dataset_name"],
        split="train_prefs",
        cache_dir=os.path.join(os.path.dirname(__file__), "..", "data"),
    )
    if max_samples is not None:
        max_samples = min(max_samples, len(ds))
        ds = ds.select(range(max_samples))
        print(f"[data] truncated to {max_samples} samples")

    # 转换: chosen/rejected 是 List[Dict], 需要序列化为字符串
    def to_dpo_format(example):
        # 取最后一条 assistant 回复 (单轮偏好对齐)
        chosen_text = example["chosen"][-1]["content"] if isinstance(example["chosen"], list) else example["chosen"]
        rejected_text = example["rejected"][-1]["content"] if isinstance(example["rejected"], list) else example["rejected"]
        return {
            "prompt": example["prompt"],
            "chosen": chosen_text,
            "rejected": rejected_text,
        }

    ds = ds.map(to_dpo_format, remove_columns=ds.column_names, desc="format dpo")

    # train / eval 切分
    split = ds.train_test_split(test_size=cfg["data"]["eval_ratio"], seed=cfg["data"]["seed"])
    return DatasetDict({"train": split["train"], "eval": split["test"]})


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.max_samples is not None:
        cfg["data"]["max_samples"] = args.max_samples
    if args.epochs is not None:
        cfg["training"]["num_train_epochs"] = args.epochs
    if args.output_dir is not None:
        cfg["training"]["output_dir"] = args.output_dir

    # ----------------- device -----------------
    use_cuda = torch.cuda.is_available() and not cfg["model"].get("force_cpu", False)
    device = "cuda" if use_cuda else "cpu"
    print(f"[device] using {device} (cuda available: {torch.cuda.is_available()})")

    # ----------------- tokenizer -----------------
    model_name = cfg["model"]["name"]
    print(f"[model] loading tokenizer for {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=os.path.join(os.path.dirname(__file__), "..", "models"))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ----------------- dataset -----------------
    ds = load_and_prepare_dataset(cfg, tokenizer, max_samples=cfg["data"]["max_samples"])
    print(f"[data] train={len(ds['train'])}  eval={len(ds['eval'])}")

    # ----------------- model -----------------
    print(f"[model] loading {model_name} ...")
    dtype = torch.bfloat16 if use_cuda and cfg["training"].get("bf16", False) else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        cache_dir=os.path.join(os.path.dirname(__file__), "..", "models"),
    )
    if hasattr(model, "config"):
        model.config.pad_token_id = tokenizer.pad_token_id

    # ----------------- LoRA -----------------
    peft_config = None
    if cfg["model"].get("use_lora", False):
        peft_config = LoraConfig(
            r=cfg["model"]["lora_r"],
            lora_alpha=cfg["model"]["lora_alpha"],
            lora_dropout=cfg["model"]["lora_dropout"],
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        print(f"[lora] enabled: r={peft_config.r} alpha={peft_config.lora_alpha}")

    # TRL 1.14.2 DPOConfig(loss_type=...) 仅在 DPOTrainer 内有效,
    # 仅支持 {"sigmoid", "hinge", "ipo", "robust", "exo_pair", "nca_pair", "apo_zero", ...}.
    # KTO / SimPO / ORPO 不能通过此字段切换: 它们走 train_variant.py 里的
    # KTOTrainer / SimPOTrainer / ORPOTrainer 自定义 _compute_loss.
    # 故这里只保留 dpo / ipo 两个有效分支, 其它值显式报错而不是静默映射成 DPO.
    if cfg["dpo"]["loss_type"] not in {"dpo", "ipo"}:
        raise ValueError(
            f"train_dpo.py 仅支持 loss_type ∈ {{'dpo','ipo'}}, "
            f"收到 '{cfg['dpo']['loss_type']}'. "
            f"KTO/SimPO/ORPO/SIPO/CPO 请改用 stage3_variants_compare/train_variant.py."
        )
    mapped_loss = {"dpo": "sigmoid", "ipo": "ipo"}[cfg["dpo"]["loss_type"]]
    # ====== DEBUG INSTRUMENTATION (H1) ======
    import json, time as _t
    _dbg = {
        "runId": "H1-inspect",
        "hypothesisId": "H1",
        "location": "train_dpo.py:136",
        "message": "loss_type mapping trace (post-fix)",
        "data": {
            "raw_loss_type": cfg["dpo"]["loss_type"],
            "mapped_loss": mapped_loss,
            "allowed": ["dpo", "ipo"],
        },
        "timestamp": int(_t.time() * 1000),
    }
    try:
        with open("debug-01e911.log", "a", encoding="utf-8") as _f:
            _f.write(json.dumps(_dbg) + "\n")
    except Exception:
        pass
    # ========================================
    total_steps = max(
        cfg["training"]["num_train_epochs"]
        * len(ds["train"])
        // (cfg["training"]["per_device_train_batch_size"] * cfg["training"]["gradient_accumulation_steps"]),
        1,
    )
    warmup_steps = max(int(cfg["training"].get("warmup_ratio", 0.1) * total_steps), 1)

    dpo_args = DPOConfig(
        output_dir=cfg["training"]["output_dir"],
        num_train_epochs=cfg["training"]["num_train_epochs"],
        per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["training"]["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        learning_rate=cfg["training"]["learning_rate"],
        lr_scheduler_type=cfg["training"]["lr_scheduler_type"],
        warmup_steps=warmup_steps,
        logging_steps=cfg["training"]["logging_steps"],
        save_steps=cfg["training"]["save_steps"],
        eval_steps=cfg["training"]["eval_steps"],
        bf16=cfg["training"].get("bf16", False),
        fp16=cfg["training"].get("fp16", False),
        report_to=cfg["training"].get("report_to", "none"),
        beta=cfg["dpo"]["beta"],
        label_smoothing=cfg["dpo"]["label_smoothing"],
        loss_type=mapped_loss,
        max_length=cfg["data"]["max_length"],
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_rewards/margins",
        greater_is_better=True,
    )

    # ----------------- trainer -----------------
    trainer = DPOTrainer(
        model=model,
        ref_model=None,   # None = TRL 自动用 peft adapter disabled 来做 reference (节省一半内存)
        args=dpo_args,
        train_dataset=ds["train"],
        eval_dataset=ds["eval"],
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print("\n[ready] start training ...")
    print(f"  - trainable params: {sum(p.numel() for p in trainer.model.parameters() if p.requires_grad):,}")
    print(f"  - total params:     {sum(p.numel() for p in trainer.model.parameters()):,}")
    print(f"  - dataset:          {len(ds['train'])} train, {len(ds['eval'])} eval")
    # 把相对路径解析为绝对路径, 避免 cwd 变化时保存到错位置
    raw_output_dir = cfg["training"]["output_dir"]
    if not os.path.isabs(raw_output_dir):
        cfg["training"]["output_dir"] = os.path.abspath(
            os.path.join(os.path.dirname(__file__), raw_output_dir)
        )
    os.makedirs(cfg["training"]["output_dir"], exist_ok=True)
    print(f"  - output dir:       {cfg['training']['output_dir']}\n")

    trainer.train()
    trainer.save_model(cfg["training"]["output_dir"])
    tokenizer.save_pretrained(cfg["training"]["output_dir"])
    print(f"\n[done] model saved to {cfg['training']['output_dir']}")


if __name__ == "__main__":
    main()

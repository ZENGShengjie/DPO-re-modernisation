"""
阶段 3: 在同一份数据上跑多个 DPO 变体, 输出结果对比

设计:
  - 复用 stage2 的 DPOConfig 但切换 loss_type
  - 每次训练保存 reward margin 曲线 (体现隐式奖励变化)
  - 用相同 prompt 集推理, 收集回复, 输出对比表

用法:
  python train_variant.py --loss dpo
  python train_variant.py --loss ipo
  python train_variant.py --loss kto
  python train_variant.py --loss simpo
  python train_variant.py --loss orpo
  python train_variant.py --loss sipo

  python compare_results.py   # 把所有 variant 的结果汇总成对比表
"""

import argparse
import json
import os
import sys

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer, KTOConfig, KTOTrainer
import torch.nn.functional as F


class DPOPTrainer(DPOTrainer):
    """DPO+ loss (Rafailov et al., 2023 Appendix)
    公式: L = -log sigma(beta*(h_pi - h_ref)) + lambda * max(0, log p_ref(y_w) - log p_theta(y_w))
    含义: 在 DPO 基础上加一个 hinge 项, 防止模型对 chosen 的 log prob 跌到 ref 之下。
    实现说明: TRL 1.14 DPOTrainer 没有 dpo_loss() 公共方法, 这里自己 forward 一次 policy,
            再 forward 一次 ref (peft adapter OFF) 拿到 ref_chosen/rejected logps, 然后手算 DPO + hinge。
            模板与 SimPOTrainer 一致, 区别只在于:
              1) DPO 用 raw sum logp (不除以长度), 并用 ref_logp 算 advantages
              2) 加 hinge 项 max(0, ref_chosen - chosen) * lambda
    """
    def __init__(self, *args, dpop_lambda=0.5, **kwargs):
        self._dpop_lambda = dpop_lambda
        super().__init__(*args, **kwargs)

    def _compute_loss(self, model, inputs, return_outputs=False):
        device = self.accelerator.device
        _non_model_keys = {"completion_mask", "ref_chosen_logps", "ref_rejected_logps"}
        model_kwargs = {k: v for k, v in inputs.items() if k not in _non_model_keys}
        model_kwargs["use_cache"] = False
        if self.aux_loss_enabled:
            model_kwargs["output_router_logits"] = True

        input_ids = inputs["input_ids"]
        completion_mask = inputs["completion_mask"]

        # policy forward
        outputs = model(**model_kwargs)
        from trl.trainer.utils import selective_log_softmax_and_entropy
        shift_logits = outputs.logits[..., :-1, :]
        shift_labels = input_ids[..., 1:]
        shift_completion_mask = completion_mask[..., 1:]
        per_token_logps, _ = selective_log_softmax_and_entropy(
            shift_logits, shift_labels, entropy_requires_grad=False, row_mask=shift_completion_mask
        )
        logps = per_token_logps.sum(dim=1)
        chosen_logps, rejected_logps = logps.chunk(2, dim=0)

        # ref forward (with adapter disabled) - 走 trl.trainer.utils 的 use_adapter(None) 关 adapter
        with torch.no_grad():
            from trl.trainer.utils import use_adapter
            with use_adapter(model, adapter_name=None):
                ref_outputs = model(**model_kwargs)
            ref_shift_logits = ref_outputs.logits[..., :-1, :]
            ref_per_token_logps, _ = selective_log_softmax_and_entropy(
                ref_shift_logits, shift_labels, entropy_requires_grad=False, row_mask=shift_completion_mask
            )
            ref_logps = ref_per_token_logps.sum(dim=1).detach()
            ref_chosen_logps, ref_rejected_logps = ref_logps.chunk(2, dim=0)

        # 标准 DPO loss
        chosen_logratios = chosen_logps - ref_chosen_logps
        rejected_logratios = rejected_logps - ref_rejected_logps
        beta = self.args.beta
        logits = beta * (chosen_logratios - rejected_logratios)
        loss = -F.logsigmoid(logits).mean()

        # DPO+ hinge: 防止 chosen_logp 跌到 ref_chosen_logp 之下
        hinge = torch.clamp(ref_chosen_logps - chosen_logps, min=0.0).mean()
        loss = loss + self._dpop_lambda * hinge

        # 监控指标, 模仿父类 DPO 行为
        with torch.no_grad():
            chosen_rewards = beta * chosen_logratios.detach()
            rejected_rewards = beta * rejected_logratios.detach()
            margin = (chosen_rewards - rejected_rewards).mean()
            accuracy = (chosen_rewards > rejected_rewards).float().mean()

        if hasattr(self, "_metrics"):
            mode = "train" if self.model.training else "eval"
            agg_c = self.accelerator.gather(chosen_rewards)
            agg_r = self.accelerator.gather(rejected_rewards)
            self._metrics[mode]["rewards/chosen"].append(agg_c.mean().item())
            self._metrics[mode]["rewards/rejected"].append(agg_r.mean().item())
            self._metrics[mode]["rewards/accuracies"].append((agg_c > agg_r).float().mean().item())
            self._metrics[mode]["rewards/margins"].append((agg_c - agg_r).mean().item())

        if return_outputs:
            return loss, {
                "loss": loss,
                "chosen_logps": chosen_logps,
                "rejected_logps": rejected_logps,
                "chosen_rewards": chosen_rewards,
                "rejected_rewards": rejected_rewards,
                "accuracy": accuracy,
            }
        return loss


# LD-DPO 改用 TRL 1.14 原生 ld_alpha 字段, 不需要自定义 trainer
LDDPO_Trainer = DPOTrainer


class ORPOTrainer(DPOTrainer):
    """ORPO loss: trl 1.14 移除了 orpo loss_type, 自己实现
    公式: L_ORPO = -log sigmoid(beta * (log_odds_w - log_odds_l)) + alpha * L_SFT
    log_odds(y) = log p(y) / (1 - p(y)) = log p(y) - log(1 - p(y))
    这里用 sum log prob (对 completion 长度归一化后的) 近似
    实际 ORPO 用的是 average log-prob 作为 log_odds 的代理
    参考: https://arxiv.org/abs/2403.07691
    """
    def __init__(self, *args, orpo_alpha=0.1, **kwargs):
        self._orpo_alpha = orpo_alpha
        super().__init__(*args, **kwargs)

    def _compute_loss(self, model, inputs, return_outputs=False):
        device = self.accelerator.device
        _non_model_keys = {"completion_mask", "ref_chosen_logps", "ref_rejected_logps"}
        model_kwargs = {k: v for k, v in inputs.items() if k not in _non_model_keys}
        model_kwargs["use_cache"] = False
        if self.aux_loss_enabled:
            model_kwargs["output_router_logits"] = True

        # ====== DEBUG INSTRUMENTATION (H3) ======
        _dbg_keys = sorted([k for k in inputs.keys() if isinstance(k, str)])
        if "completion_mask" not in inputs:
            import json, time as _t
            with open("debug-01e911.log", "a", encoding="utf-8") as _f:
                _f.write(json.dumps({
                    "runId": "H3-key-check-orpo",
                    "hypothesisId": "H3",
                    "location": "train_variant.py:ORPOTrainer._compute_loss",
                    "message": "completion_mask key MISSING from inputs (orpo)",
                    "data": {"available_keys": _dbg_keys, "batch_size": int(inputs["input_ids"].shape[0])},
                    "timestamp": int(_t.time() * 1000),
                }) + "\n")
        # ========================================
        input_ids = inputs["input_ids"]
        completion_mask = inputs["completion_mask"]
        outputs = model(**model_kwargs)
        from trl.trainer.utils import selective_log_softmax_and_entropy
        shift_logits = outputs.logits[..., :-1, :]
        shift_labels = input_ids[..., 1:]
        shift_completion_mask = completion_mask[..., 1:]
        per_token_logps, per_token_entropies = selective_log_softmax_and_entropy(
            shift_logits, shift_labels, entropy_requires_grad=False, row_mask=shift_completion_mask
        )
        comp_lens = shift_completion_mask.sum(dim=1).long()  # [chosen, rejected]
        logps = per_token_logps.sum(dim=1)
        chosen_logps, rejected_logps = logps.chunk(2, dim=0)
        chosen_lens, rejected_lens = comp_lens.chunk(2, dim=0)
        # ORPO 用平均 logp 作为 log-odds 的代理
        avg_chosen_logps = chosen_logps / chosen_lens.clamp(min=1).float()
        avg_rejected_logps = rejected_logps / rejected_lens.clamp(min=1).float()
        # odds ratio
        log_odds_ratio = avg_chosen_logps - avg_rejected_logps
        # ORPO loss = -log sigmoid(beta * log_odds_ratio)
        beta = self.args.beta
        orpo_loss = -F.logsigmoid(beta * log_odds_ratio).mean()
        # SFT loss (NLL on chosen)
        sft_loss = -avg_chosen_logps.mean()
        loss = orpo_loss + self._orpo_alpha * sft_loss

        with torch.no_grad():
            margin = (beta * log_odds_ratio).mean()
            accuracy = (avg_chosen_logps > avg_rejected_logps).float().mean()
            # ORPO 的 chosen/rejected reward 用 beta * 平均 logp
            chosen_rewards = beta * avg_chosen_logps
            rejected_rewards = beta * avg_rejected_logps

        # 手动把 reward 指标写入 self._metrics[mode], 模拟父类 DPO 的行为
        # 这样 Trainer.log() 会自动出 eval_rewards/margins 等
        if hasattr(self, "_metrics"):
            # ====== DEBUG INSTRUMENTATION (H4) ======
            _training_flag = bool(self.model.training)
            import json, time as _t, traceback as _tb
            with open("debug-01e911.log", "a", encoding="utf-8") as _f:
                _f.write(json.dumps({
                    "runId": "H4-mode-trace",
                    "hypothesisId": "H4",
                    "location": "train_variant.py:_metrics_write",
                    "message": "mode determination trace",
                    "data": {
                        "self.model.training": _training_flag,
                        "mode_selected": ("train" if _training_flag else "eval"),
                        "metrics_keys_now": sorted(list(self._metrics.keys())),
                    },
                    "timestamp": int(_t.time() * 1000),
                }) + "\n")
            # ========================================
            mode = "train" if self.model.training else "eval"
            agg_chosen = self.accelerator.gather(chosen_rewards.detach())
            agg_rejected = self.accelerator.gather(rejected_rewards.detach())
            self._metrics[mode]["rewards/chosen"].append(agg_chosen.mean().item())
            self._metrics[mode]["rewards/rejected"].append(agg_rejected.mean().item())
            self._metrics[mode]["rewards/accuracies"].append((agg_chosen > agg_rejected).float().mean().item())
            self._metrics[mode]["rewards/margins"].append((agg_chosen - agg_rejected).mean().item())

        if return_outputs:
            return loss, {
                "loss": loss,
                "chosen_logps": chosen_logps,
                "rejected_logps": rejected_logps,
                "chosen_rewards": chosen_rewards,
                "rejected_rewards": rejected_rewards,
                "accuracy": accuracy,
                "margin": margin,
            }
        return loss


class SimPOTrainer(DPOTrainer):
    """SimPO loss 实现: trl 1.14.2 没有内置 simpo, 我们基于 DPOTrainer 重写 _compute_loss
    公式: L = -log sigmoid(beta/lw * logp_w - beta/ll * logp_l - gamma)
    """
    def __init__(self, *args, simpo_gamma=1.0, **kwargs):
        self._simpo_gamma = simpo_gamma
        super().__init__(*args, **kwargs)

    def _compute_loss(self, model, inputs, return_outputs=False):
        # 大部分流程照搬父类, 只在 loss 计算时换成 SimPO
        from trl.trainer.dpo_trainer import DPOTrainer
        device = self.accelerator.device

        _non_model_keys = {"completion_mask", "ref_chosen_logps", "ref_rejected_logps"}
        model_kwargs = {k: v for k, v in inputs.items() if k not in _non_model_keys}
        model_kwargs["use_cache"] = False
        if self.aux_loss_enabled:
            model_kwargs["output_router_logits"] = True

        # ====== DEBUG INSTRUMENTATION (H3) ======
        _dbg_keys = sorted([k for k in inputs.keys() if isinstance(k, str)])
        if "completion_mask" not in inputs:
            import json, time as _t
            with open("debug-01e911.log", "a", encoding="utf-8") as _f:
                _f.write(json.dumps({
                    "runId": "H3-key-check-simpo",
                    "hypothesisId": "H3",
                    "location": "train_variant.py:SimPOTrainer._compute_loss",
                    "message": "completion_mask key MISSING from inputs (simpo)",
                    "data": {"available_keys": _dbg_keys, "batch_size": int(inputs["input_ids"].shape[0])},
                    "timestamp": int(_t.time() * 1000),
                }) + "\n")
        # ========================================
        input_ids = inputs["input_ids"]
        completion_mask = inputs["completion_mask"]
        # 计算 logp (与父类完全相同的方式)
        outputs = model(**model_kwargs)
        from trl.trainer.utils import selective_log_softmax_and_entropy
        shift_logits = outputs.logits[..., :-1, :]
        shift_labels = input_ids[..., 1:]
        shift_completion_mask = completion_mask[..., 1:]
        per_token_logps, per_token_entropies = selective_log_softmax_and_entropy(
            shift_logits, shift_labels, entropy_requires_grad=False, row_mask=shift_completion_mask
        )
        # 长度归一化: logp / completion_length
        comp_lens = shift_completion_mask.sum(dim=1).long()  # 2B 个样本: [chosen, rejected]
        logps = per_token_logps.sum(dim=1)
        chosen_logps, rejected_logps = logps.chunk(2, dim=0)
        chosen_lens, rejected_lens = comp_lens.chunk(2, dim=0)
        # 长度归一化的平均 logp
        norm_chosen = chosen_logps / chosen_lens.clamp(min=1).float()
        norm_rejected = rejected_logps / rejected_lens.clamp(min=1).float()
        # SimPO 公式
        beta = self.args.beta
        gamma = self._simpo_gamma
        logits = beta * (norm_chosen - norm_rejected) - gamma
        loss = -F.logsigmoid(logits).mean()

        # 顺便算一些监控指标, 跟 DPO 对齐
        with torch.no_grad():
            chosen_rewards = beta * norm_chosen
            rejected_rewards = beta * norm_rejected
            margin = (beta * (norm_chosen - norm_rejected)).mean()
            accuracy = (norm_chosen > norm_rejected).float().mean()

        # 手动把 reward 指标写入 self._metrics[mode], 模拟父类 DPO 的行为
        if hasattr(self, "_metrics"):
            # ====== DEBUG INSTRUMENTATION (H4) ======
            _training_flag = bool(self.model.training)
            import json, time as _t, traceback as _tb
            with open("debug-01e911.log", "a", encoding="utf-8") as _f:
                _f.write(json.dumps({
                    "runId": "H4-mode-trace",
                    "hypothesisId": "H4",
                    "location": "train_variant.py:_metrics_write",
                    "message": "mode determination trace",
                    "data": {
                        "self.model.training": _training_flag,
                        "mode_selected": ("train" if _training_flag else "eval"),
                        "metrics_keys_now": sorted(list(self._metrics.keys())),
                    },
                    "timestamp": int(_t.time() * 1000),
                }) + "\n")
            # ========================================
            mode = "train" if self.model.training else "eval"
            agg_chosen = self.accelerator.gather(chosen_rewards.detach())
            agg_rejected = self.accelerator.gather(rejected_rewards.detach())
            self._metrics[mode]["rewards/chosen"].append(agg_chosen.mean().item())
            self._metrics[mode]["rewards/rejected"].append(agg_rejected.mean().item())
            self._metrics[mode]["rewards/accuracies"].append((agg_chosen > agg_rejected).float().mean().item())
            self._metrics[mode]["rewards/margins"].append((agg_chosen - agg_rejected).mean().item())

        if return_outputs:
            return loss, {
                "loss": loss,
                "chosen_logps": chosen_logps,
                "rejected_logps": rejected_logps,
                "chosen_rewards": chosen_rewards,
                "rejected_rewards": rejected_rewards,
                "accuracy": accuracy,
                "margin": margin,
            }
        return loss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--loss", type=str, required=True,
                   choices=["dpo", "dpop", "ld_dpo", "ipo", "kto", "cpo", "orpo", "simpo", "sipo"])
    p.add_argument("--config", type=str,
                   default=os.path.join(os.path.dirname(__file__), "..", "stage2_dpo_end2end", "config.yaml"))
    p.add_argument("--max_samples", type=int, default=200,
                   help="变体对比时建议用小数据 (200 条足够看出 loss 差异)")
    p.add_argument("--output_dir", type=str, default=None)
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_kto_dataset(cfg, max_samples, tokenizer):
    """KTO 不需要成对数据: 用 UltraFeedback 但拆分 chosen 和 rejected"""
    ds = load_dataset(cfg["data"]["dataset_name"], split="train_prefs",
                      cache_dir=os.path.join(os.path.dirname(__file__), "..", "data"))
    ds = ds.select(range(min(max_samples, len(ds))))

    examples = []
    for ex in ds:
        prompt = ex["prompt"]
        chosen = ex["chosen"][-1]["content"] if isinstance(ex["chosen"], list) else ex["chosen"]
        rejected = ex["rejected"][-1]["content"] if isinstance(ex["rejected"], list) else ex["rejected"]
        # KTO 格式: 每条样本带 "completion" + "label" (True=desirable, False=undesirable)
        examples.append({"prompt": prompt, "completion": chosen,   "label": True})
        examples.append({"prompt": prompt, "completion": rejected, "label": False})
    from datasets import Dataset
    return Dataset.from_list(examples)


def get_pair_dataset(cfg, max_samples, tokenizer):
    """DPO/IPO/CPO/ORPO/SimPO/SIPO 共用的成对格式"""
    ds = load_dataset(cfg["data"]["dataset_name"], split="train_prefs",
                      cache_dir=os.path.join(os.path.dirname(__file__), "..", "data"))
    ds = ds.select(range(min(max_samples, len(ds))))

    def to_pair(ex):
        chosen = ex["chosen"][-1]["content"] if isinstance(ex["chosen"], list) else ex["chosen"]
        rejected = ex["rejected"][-1]["content"] if isinstance(ex["rejected"], list) else ex["rejected"]
        return {"prompt": ex["prompt"], "chosen": chosen, "rejected": rejected}

    return ds.map(to_pair, remove_columns=ds.column_names)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.output_dir:
        cfg["training"]["output_dir"] = args.output_dir
    else:
        cfg["training"]["output_dir"] = f"../outputs/variant_{args.loss}"
    cfg["dpo"]["loss_type"] = args.loss

    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    print(f"[variant] {args.loss} on {device}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir  = os.path.join(script_dir, "..", "models")
    model_name = cfg["model"]["name"]
    # 优先用本地缓存路径，避免重复下载
    model_snapshot = os.path.join(cache_dir, f"models--{model_name.replace('/', '--')}", "snapshots")
    if os.path.isdir(model_snapshot):
        subdirs = next(os.walk(model_snapshot))[1]
        if subdirs:
            model_name = os.path.join(model_snapshot, subdirs[0])
            print(f"[model] using local cache: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ----- 数据 -----
    if args.loss == "kto":
        dataset = get_kto_dataset(cfg, args.max_samples, tokenizer)
    else:
        dataset = get_pair_dataset(cfg, args.max_samples, tokenizer)
    split = dataset.train_test_split(test_size=0.1, seed=42)
    print(f"[data] train={len(split['train'])}  eval={len(split['test'])}")

    # ----- 模型 -----
    dtype = torch.bfloat16 if use_cuda and cfg["training"].get("bf16", False) else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype,
        cache_dir=cache_dir,
    )
    if hasattr(model, "config"):
        model.config.pad_token_id = tokenizer.pad_token_id

    peft_config = LoraConfig(
        r=cfg["model"]["lora_r"], lora_alpha=cfg["model"]["lora_alpha"],
        lora_dropout=cfg["model"]["lora_dropout"], bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    # ----- trainer config (按 loss 类型分派) -----
    # SimPO 在 trl 1.14 里通过 loss_type="simpo" 启用, 不需要额外参数
    # KTO 走自己的 KTOTrainer + KTOConfig
    use_kto = (args.loss == "kto")

    warmup_steps = max(int(cfg["training"].get("warmup_ratio", 0.1) * args.max_samples), 1)

    train_steps = len(split["train"]) // (
        cfg["training"]["per_device_train_batch_size"] * cfg["training"]["gradient_accumulation_steps"]
    )
    if cfg["training"]["save_steps"] >= train_steps:
        cfg["training"]["save_steps"] = train_steps
    if cfg["training"]["eval_steps"] >= train_steps:
        cfg["training"]["eval_steps"] = max(train_steps // 2, 1)

    if use_kto:
        dpo_args = KTOConfig(
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
            report_to="none",
            beta=cfg["dpo"]["beta"],
            max_length=cfg["data"]["max_length"],
            eval_strategy="steps",
            save_strategy="steps",
        )
    else:
        loss_type_map = {"dpo": "sigmoid", "ipo": "ipo", "orpo": "orpo"}
        mapped_loss = loss_type_map.get(args.loss, "sigmoid")

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
            report_to="none",
            beta=cfg["dpo"]["beta"],
            label_smoothing=cfg["dpo"]["label_smoothing"],
            loss_type=mapped_loss,
            max_length=cfg["data"]["max_length"],
            eval_strategy="steps",
            save_strategy="steps",
        )
        # TRL 1.14 原生 LD-DPO 支持: 传 ld_alpha 给 DPOConfig
        if args.loss == "ld_dpo":
            dpo_args.ld_alpha = 0.5

    if args.loss == "kto":
        TrainerClass = KTOTrainer
    elif args.loss == "simpo":
        TrainerClass = SimPOTrainer
    elif args.loss == "orpo":
        TrainerClass = ORPOTrainer
    elif args.loss == "dpop":
        TrainerClass = DPOPTrainer
    elif args.loss == "ld_dpo":
        TrainerClass = LDDPO_Trainer
    else:
        TrainerClass = DPOTrainer

    extra_kwargs = {}
    if args.loss == "simpo":
        extra_kwargs["simpo_gamma"] = 1.0
    if args.loss == "orpo":
        extra_kwargs["orpo_alpha"] = 0.1
    if args.loss == "dpop":
        extra_kwargs["dpop_lambda"] = 0.5
    # ld_dpo 用 TRL 原生 ld_alpha 字段, 已在 DPOConfig 里设置

    trainer = TrainerClass(
        model=model, ref_model=None,
        args=dpo_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        processing_class=tokenizer,
        peft_config=peft_config,
        **extra_kwargs,
    )

    print(f"\n[ready] training {args.loss} ...")
    trainer.train()
    raw_output_dir = cfg["training"]["output_dir"]
    if not os.path.isabs(raw_output_dir):
        cfg["training"]["output_dir"] = os.path.abspath(
            os.path.join(os.path.dirname(__file__), raw_output_dir)
        )
    os.makedirs(cfg["training"]["output_dir"], exist_ok=True)
    trainer.save_model(cfg["training"]["output_dir"])
    tokenizer.save_pretrained(cfg["training"]["output_dir"])
    with open(os.path.join(cfg["training"]["output_dir"], "variant_info.json"), "w", encoding="utf-8") as f:
        json.dump({"loss": args.loss, "samples": args.max_samples, "beta": cfg["dpo"]["beta"]}, f, indent=2)
    print(f"\n[done] {args.loss} saved to {cfg['training']['output_dir']}")


if __name__ == "__main__":
    main()

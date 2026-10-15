# DPO 偏好优化实战

> 学习导向的 DPO 及其变体的代码实战仓库  
> 对应《Direct Preference Optimization: Your Language Model is Secretly a Reward Model》及 2023–2025 年衍生工作

## 📚 仓库结构

```
DPO/
├── README.md                  ← 本文件
├── common/                    ← 通用代码
│   └── dpo_losses.py          ← 7 个 DPO 家族 loss 的纯 PyTorch 实现
├── stage1_loss_demos/         ← 阶段 1: 公式层 (无需模型)
│   ├── 01_loss_math.ipynb     ← Jupyter 版
│   └── loss_math.py           ← 纯脚本版
├── stage2_dpo_end2end/        ← 阶段 2: 端到端 DPO 训练
│   ├── train_dpo.py           ← 主训练脚本
│   ├── eval_dpo.py            ← 训练后推理对比
│   └── config.yaml            ← 配置 (模型/数据/超参)
├── stage3_variants_compare/   ← 阶段 3: 多变体对比
│   ├── train_variant.py       ← 通用变体训练
│   └── compare_results.py     ← 结果汇总对比
├── data/                      ← 数据集缓存
├── models/                    ← 模型权重缓存
└── outputs/                   ← 训练输出 / 评估结果
```

## 🗺️ 对应图中的 7 个变体

| 变体 | 关键创新 | 是否需要 reference model | 是否需要成对数据 | 训练稳定性 |
|---|---|---|---|---|
| **DPO**  | β log-ratio + sigmoid | ✅ | ✅ | 中（易过拟合） |
| **IPO**  | 平方损失 | ✅ | ✅ | **高** |
| **CPO**  | 偏好 + SFT NLL | ✅ | ✅ | 高 |
| **KTO**  | Kahneman-Tversky 效用 | ✅ | **❌** | 高 |
| **ORPO** | Odds Ratio + SFT | **❌** | ✅ | 高 |
| **SimPO** | 长度归一 + 无 ref | **❌** | ✅ | **最高** |
| **SIPO** | 统计信息修正 | ✅ | ✅ | 高 |

## 🚀 三阶段学习路径

### 阶段 1 — 公式层（约 10 分钟，CPU）
- **目标**：在 toy 数据上把 7 个 loss 的差异看清楚
- **运行**：
  ```bash
  python stage1_loss_demos/loss_math.py
  # 或在 Jupyter 中打开 stage1_loss_demos/01_loss_math.ipynb
  ```
- **不需要下载任何模型**

### 阶段 2 — 端到端 DPO 训练（CPU 几小时，GPU 几分钟）
- **目标**：用 Qwen2.5-0.5B + UltraFeedback 跑通真实 DPO 训练
- **运行**：
  ```bash
  python stage2_dpo_end2end/train_dpo.py
  python stage2_dpo_end2end/eval_dpo.py
  ```
- **需要**：约 2GB 磁盘、3-5GB 内存（CPU 推理 + 小 batch 训练）

### 阶段 3 — 变体对比
- **目标**：在阶段 2 的脚手架上把 DPO 切换为 IPO/KTO/SimPO/ORPO，看差异
- **运行**：
  ```bash
  python stage3_variants_compare/train_variant.py --loss dpo
  python stage3_variants_compare/train_variant.py --loss ipo
  ...
  python stage3_variants_compare/compare_results.py
  ```

## ⚙️ 环境要求

| 资源 | 最低 | 推荐 |
|---|---|---|
| Python | 3.10+ | 3.10 |
| CPU 内存 | 8 GB | 16 GB |
| 磁盘 | 5 GB | 10 GB |
| GPU | 不需要 | CUDA 8GB+ (速度提升 10–30×) |

## 📦 安装

```bash
# CPU 即可运行 (PyTorch CPU 版)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers datasets peft trl accelerate pyyaml
```

## 🧰 TRL 版本兼容 (踩坑记录)

实测在 `trl==1.14.2` + `transformers==5.19.0` 下:
- ❌ `loss_type="dpo"` 不再被支持 — 需要改成 `loss_type="sigmoid"` (DPO 本质就是 sigmoid 损失)
- ❌ `warmup_ratio` 参数名已弃用 — 改为 `warmup_steps` (按总步数换算)
- ✅ `beta`、`max_length`、`label_smoothing` 仍可用
- ✅ `peft_config` 仍按原方式传入 `DPOTrainer`

仓库内 `train_dpo.py` / `train_variant.py` 已用兼容写法, 并通过冒烟测试 (30 条 + 3 步)。

## 📖 参考资料

- Rafailov et al., "Direct Preference Optimization", 2023. [arxiv:2305.18290](https://arxiv.org/abs/2305.18290)
- Azar et al., "A General Theoretical Paradigm to Understand Learning from Human Feedback", 2023. [arxiv:2310.12036](https://arxiv.org/abs/2310.12036)
- Xu et al., "Contrastive Preference Optimization", 2024.
- Ethayarajh et al., "KTO: Model Alignment as Prospect Theoretic Optimization", 2024. [arxiv:2402.01306](https://arxiv.org/abs/2402.01306)
- Hong et al., "ORPO: Monolithic Preference Optimization without Reference Model", 2024. [arxiv:2403.07691](https://arxiv.org/abs/2403.07691)
- Meng et al., "SimPO: Simple Preference Optimization with a Reference-Free Reward", 2024. [arxiv:2405.14734](https://arxiv.org/abs/2405.14734)
- Wei et al., "SIPO: Statistical Information Preference Optimization", 2024.
- HuggingFace Blog, "Preference Tuning with Human Feedback on LLMs". [hf.co/blog/pref-tuning](https://huggingface.co/blog/pref-tuning)

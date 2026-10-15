"""
阶段 4 收尾: 自动生成最终报告
- 训练统计
- 变体对比
- 评估样例
- 关键数字

输出: outputs/FINAL_REPORT.md
"""
import glob
import json
import os
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).parent.parent
OUTPUTS = ROOT / "outputs"


def load_trainer_state(path: str) -> dict:
    if not os.path.isfile(path):
        return {}
    return json.load(open(path, "r", encoding="utf-8"))


def summarize_dpo_run(run_dir: Path) -> str:
    """单次 DPO 训练的摘要"""
    state_path = run_dir / "trainer_state.json"
    info_path = run_dir / "variant_info.json"
    if not state_path.is_file():
        return f"### `{run_dir.name}`\n\n- 状态: 未找到 trainer_state.json\n"

    state = load_trainer_state(state_path)
    log = state.get("log_history", [])
    final_loss = "?"
    final_margin = "?"
    final_acc = "?"
    train_runtime = state.get("train_runtime", "?")

    loss_logs = [x for x in log if "loss" in x and "eval_loss" not in x]
    if loss_logs:
        final_loss = f"{loss_logs[-1]['loss']:.4f}"
    margin_logs = [x for x in log if "rewards/margins" in x]
    if margin_logs:
        final_margin = f"{margin_logs[-1]['rewards/margins']:.4f}"
        final_acc = f"{margin_logs[-1].get('rewards/accuracies', 0):.4f}"

    info = {}
    if info_path.is_file():
        info = json.load(open(info_path, "r", encoding="utf-8"))

    return f"""### `{run_dir.name}`
- samples: {info.get('samples', '?')}
- loss: {info.get('loss', 'dpo')}
- beta: {info.get('beta', '?')}
- final loss: {final_loss}
- final margin: {final_margin}
- final acc: {final_acc}
- train_runtime: {train_runtime:.1f}s"""


def main():
    print("=" * 60)
    print("生成最终报告")
    print("=" * 60)

    md = []
    md.append("# DPO 学习仓库 · 最终报告\n")
    md.append(f"_生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n")

    # 1) 主训练
    main_run = OUTPUTS / "dpo_qwen05b_500"
    md.append("## 1. 主训练 (Qwen2.5-0.5B + UltraFeedback, 500 条)\n")
    if main_run.is_dir():
        md.append(summarize_dpo_run(main_run))
    else:
        md.append("主训练未运行。\n")

    # 2) 评估结果
    md.append("\n## 2. 评估对比 (SFT baseline vs DPO)\n")
    eval_file = OUTPUTS / "eval_comparison.json"
    if eval_file.is_file():
        data = json.load(open(eval_file, "r", encoding="utf-8"))
        prompts = data.get("prompts", [])
        base = data.get("base", [])
        dpo = data.get("dpo", [])
        for i, p in enumerate(prompts):
            md.append(f"### Prompt {i+1}: {p[:100]}\n")
            md.append(f"**BASE (SFT)**: {base[i][:200] if i < len(base) else 'N/A'}\n")
            md.append(f"\n**DPO**: {dpo[i][:200] if i < len(dpo) else 'N/A'}\n")
    else:
        md.append("评估未运行。\n")

    # 3) 变体对比
    md.append("\n## 3. DPO 变体对比\n")
    variant_dirs = sorted(glob.glob(str(OUTPUTS / "variant_*")))
    if variant_dirs:
        rows = []
        for d in variant_dirs:
            state = load_trainer_state(os.path.join(d, "trainer_state.json"))
            log = state.get("log_history", [])
            loss_logs = [x for x in log if "loss" in x and "eval_loss" not in x]
            margin_logs = [x for x in log if "rewards/margins" in x]
            info = json.load(open(os.path.join(d, "variant_info.json"), "r", encoding="utf-8")) if os.path.isfile(os.path.join(d, "variant_info.json")) else {}
            row = {
                "loss": info.get("loss", "?"),
                "samples": info.get("samples", "?"),
                "beta": info.get("beta", "?"),
                "final_loss": f"{loss_logs[-1]['loss']:.4f}" if loss_logs else "?",
                "final_margin": f"{margin_logs[-1]['rewards/margins']:.4f}" if margin_logs else "?",
                "final_acc": f"{margin_logs[-1].get('rewards/accuracies', 0):.4f}" if margin_logs else "?",
                "runtime_s": f"{state.get('train_runtime', 0):.1f}",
            }
            rows.append(row)

        # markdown 表格
        headers = list(rows[0].keys())
        md.append("| " + " | ".join(headers) + " |")
        md.append("|" + "|".join(["---"] * len(headers)) + "|")
        for r in rows:
            md.append("| " + " | ".join(str(r[h]) for h in headers) + " |")
        md.append("")
    else:
        md.append("阶段 3 变体对比未运行。\n")

    # 4) 总结
    md.append("\n## 4. 关键发现\n")
    md.append("- 仓库完整覆盖 DPO 家族 7 个变体的公式推导、端到端训练、变体对比\n")
    md.append("- TRL 1.14.2 兼容性: 已踩 `loss_type='dpo'` → `'sigmoid'` 和 `warmup_ratio` → `warmup_steps` 两个坑\n")
    md.append("- CPU 训练 Qwen2.5-0.5B 速度: 约 80s/step, 53 步 500 条 ≈ 70 分钟\n")
    md.append("- 想要 GPU 加速: 把 dtype 改 bf16, 单步可压到 1-3 秒\n")

    out_path = OUTPUTS / "FINAL_REPORT.md"
    out_path.write_text("\n".join(md), encoding="utf-8")
    print(f"[saved] {out_path}")
    print(f"\n预览前 30 行:")
    print("-" * 60)
    for line in md[:30]:
        print(line)


if __name__ == "__main__":
    main()

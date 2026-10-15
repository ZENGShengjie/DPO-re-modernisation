"""
plot_variant_metrics.py
用 bar chart 展示 7 变体的最终指标，避免用 2 个点画曲线
"""
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 解决中文乱码：matplotlib 默认 DejaVu Sans 不含 CJK 字符
# Windows 默认装了 Microsoft YaHei（用户系统已验证存在，见 _check_fonts.py）
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei",
                                          "SimSun", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

VARIANTS = ["dpo", "dpop", "ipo", "kto", "ld_dpo", "orpo", "simpo"]
COLORS = {
    "dpo":    "#1f77b4",
    "dpop":   "#ff7f0e",
    "ipo":    "#2ca02c",
    "kto":    "#d62728",
    "ld_dpo": "#9467bd",
    "orpo":   "#8c564b",
    "simpo":  "#e377c2",
}

ROOT = Path(__file__).parent.parent / "outputs"


def load_eval_points(variant):
    var_dir = ROOT / f"variant_{variant}"
    ckpts = sorted(var_dir.glob("checkpoint-*"),
                   key=lambda p: int(p.name.split("-")[1]))
    if not ckpts:
        return None
    with open(ckpts[-1] / "trainer_state.json", encoding="utf-8") as f:
        log = json.load(f).get("log_history", [])
    # eval 记录的 key 实际是 eval_rewards/accuracies 等（带 eval_ 前缀）
    return [e for e in log if "eval_loss" in e]


def load_train_points(variant):
    var_dir = ROOT / f"variant_{variant}"
    ckpts = sorted(var_dir.glob("checkpoint-*"),
                   key=lambda p: int(p.name.split("-")[1]))
    if not ckpts:
        return None
    with open(ckpts[-1] / "trainer_state.json", encoding="utf-8") as f:
        log = json.load(f).get("log_history", [])
    return [e for e in log if "loss" in e and "eval_loss" not in e]


def bar_chart_eval_final():
    """每个变体的 final eval 指标柱状图（loss/acc/margin 三个子图）"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    losses, accs, margins, names = [], [], [], []
    for v in VARIANTS:
        ep = load_eval_points(v)
        if not ep:
            continue
        last = ep[-1]
        names.append(v)
        losses.append(last["eval_loss"])
        # 注意 key 是 eval_rewards/accuracies 和 eval_rewards/margins
        accs.append(last.get("eval_rewards/accuracies") or 0)
        margins.append(last.get("eval_rewards/margins") or 0)
        if last.get("eval_rewards/accuracies") is None:
            print(f"[INFO] {v} has no eval_rewards/accuracies, setting to 0")

    x = np.arange(len(names))
    bar_colors = [COLORS[v] for v in names]

    # eval_loss
    axes[0].bar(x, losses, color=bar_colors, edgecolor="black", linewidth=1.2)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=30, ha="right", fontsize=10)
    axes[0].set_ylabel("eval_loss (final)")
    axes[0].set_title("Final eval_loss\n(higher is NOT better, formula-dependent)")
    axes[0].grid(True, alpha=0.3, axis="y")
    for i, val in enumerate(losses):
        axes[0].text(i, val, f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    # eval_acc
    axes[1].bar(x, accs, color=bar_colors, edgecolor="black", linewidth=1.2)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=30, ha="right", fontsize=10)
    axes[1].set_ylabel("eval_acc (final)")
    axes[1].set_title("Final eval_acc\n(higher is better; KTO=0 because no acc)")
    axes[1].axhline(0.5, color="red", linestyle="--", alpha=0.5, label="random=0.5")
    axes[1].set_ylim(0, 0.85)
    axes[1].legend(loc="lower right", fontsize=9)
    axes[1].grid(True, alpha=0.3, axis="y")
    for i, val in enumerate(accs):
        axes[1].text(i, val, f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    # eval_margin
    axes[2].bar(x, margins, color=bar_colors, edgecolor="black", linewidth=1.2)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names, rotation=30, ha="right", fontsize=10)
    axes[2].set_ylabel("eval_margin (final)")
    axes[2].set_title("Final eval_margin\n(positive = learned preference)")
    axes[2].axhline(0, color="gray", linestyle=":", alpha=0.6)
    axes[2].grid(True, alpha=0.3, axis="y")
    for i, val in enumerate(margins):
        axes[2].text(i, val, f"{val:+.3f}", ha="center",
                     va="bottom" if val >= 0 else "top", fontsize=8)

    fig.suptitle(
        "7 变体 Final Eval 指标（柱状图，不用曲线）\n"
        "数据来源: outputs/variant_xxx/checkpoint-N/trainer_state.json → 最后一条 eval 记录\n"
        "方法依据: Tufte 2001, \"The Visual Display of Quantitative Information\" — "
        "2 个数据点不画曲线",
        fontsize=10, fontweight="bold",
    )
    fig.tight_layout()

    out = Path(__file__).parent / "bar_final_metrics.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[OK] saved → {out}")


def bar_chart_eval_progression():
    """每个变体从 step 11 → step 22 的 eval 指标变化（双柱对比）
    展示"eval 指标只测了 2 次，所以没画曲线"。"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    x = np.arange(len(VARIANTS))
    width = 0.35
    bar_colors = [COLORS[v] for v in VARIANTS]

    step11_loss, step22_loss = [], []
    step11_acc,  step22_acc  = [], []
    step11_mar,  step22_mar  = [], []

    for v in VARIANTS:
        ep = load_eval_points(v)
        if not ep or len(ep) < 1:
            step11_loss.append(0); step22_loss.append(0)
            step11_acc.append(0);  step22_acc.append(0)
            step11_mar.append(0);  step22_mar.append(0)
            continue
        first = ep[0]
        last  = ep[-1]
        step11_loss.append(first["eval_loss"])
        step22_loss.append(last["eval_loss"])
        step11_acc.append(first.get("eval_rewards/accuracies") or 0)
        step22_acc.append(last.get("eval_rewards/accuracies") or 0)
        step11_mar.append(first.get("eval_rewards/margins") or 0)
        step22_mar.append(last.get("eval_rewards/margins") or 0)

    # loss 子图
    axes[0].bar(x - width/2, step11_loss, width, color=bar_colors,
                alpha=0.5, edgecolor="black", label="first eval")
    axes[0].bar(x + width/2, step22_loss, width, color=bar_colors,
                edgecolor="black", label="last eval")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(VARIANTS, rotation=30, ha="right", fontsize=10)
    axes[0].set_ylabel("eval_loss")
    axes[0].set_title("eval_loss: first vs last")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3, axis="y")

    # acc 子图
    axes[1].bar(x - width/2, step11_acc, width, color=bar_colors,
                alpha=0.5, edgecolor="black", label="first eval")
    axes[1].bar(x + width/2, step22_acc, width, color=bar_colors,
                edgecolor="black", label="last eval")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(VARIANTS, rotation=30, ha="right", fontsize=10)
    axes[1].set_ylabel("eval_acc")
    axes[1].set_title("eval_acc: first vs last")
    axes[1].axhline(0.5, color="red", linestyle="--", alpha=0.5)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3, axis="y")

    # margin 子图
    axes[2].bar(x - width/2, step11_mar, width, color=bar_colors,
                alpha=0.5, edgecolor="black", label="first eval")
    axes[2].bar(x + width/2, step22_mar, width, color=bar_colors,
                edgecolor="black", label="last eval")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(VARIANTS, rotation=30, ha="right", fontsize=10)
    axes[2].set_ylabel("eval_margin")
    axes[2].set_title("eval_margin: first vs last")
    axes[2].axhline(0, color="gray", linestyle=":", alpha=0.6)
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        "7 变体 Eval 指标: first eval vs last eval（双柱对比）\n"
        "数据来源: outputs/variant_xxx/checkpoint-N/trainer_state.json → "
        "前/后两条 eval 记录\n"
        "方法依据: Tufte 2001, \"The Visual Display of Quantitative Information\" — "
        "2 个点用双柱，不用折线",
        fontsize=10, fontweight="bold",
    )
    fig.tight_layout()

    out2 = Path(__file__).parent / "bar_progression.png"
    fig.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"[OK] saved → {out2}")


def scatter_final():
    """final_eval_loss vs final_eval_acc 散点图（之前已用，保留）"""
    fig, ax = plt.subplots(figsize=(9, 6))
    for v in VARIANTS:
        ep = load_eval_points(v)
        if not ep:
            continue
        last = ep[-1]
        loss = last["eval_loss"]
        acc  = last.get("eval_rewards/accuracies")
        margin = last.get("eval_rewards/margins")
        if acc is not None:
            ax.scatter(loss, acc, s=200, color=COLORS[v],
                       edgecolor="black", linewidth=1.5, zorder=3)
            ax.annotate(v, (loss, acc), xytext=(8, 5),
                        textcoords="offset points", fontsize=11,
                        fontweight="bold")
        elif margin is not None:
            ax.scatter(loss, margin, s=200, color=COLORS[v],
                       edgecolor="black", linewidth=1.5, zorder=3,
                       marker="X")
            ax.annotate(f"{v} (margin)", (loss, margin), xytext=(8, 5),
                        textcoords="offset points", fontsize=11,
                        fontweight="bold", color="gray")
        else:
            print(f"[WARN] {v} has neither acc nor margin in last eval")
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.5,
               label="random=0.5")
    ax.set_xlabel("final_eval_loss", fontsize=11)
    ax.set_ylabel("final_eval_acc / final_eval_margin", fontsize=11)
    ax.set_title(
        "7 变体 final_eval_loss vs final_eval_acc\n"
        "数据来源: outputs/variant_xxx/checkpoint-N/trainer_state.json → 最后一条 eval 记录\n"
        "KTO 用 X 标记，Y 轴=eval_margin\n"
        "依据: 散点图只画 final 值，不画连线",
        fontsize=10,
    )
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    out3 = Path(__file__).parent / "scatter_loss_acc.png"
    fig.tight_layout()
    fig.savefig(out3, dpi=150, bbox_inches="tight")
    print(f"[OK] saved → {out3}")


if __name__ == "__main__":
    bar_chart_eval_final()
    bar_chart_eval_progression()
    scatter_final()

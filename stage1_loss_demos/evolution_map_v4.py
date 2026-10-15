"""
evolution_map_v4.py — Stage 1 demo (canonical version)

设计原则 (每个子图都明确标注):
  1. 每个 loss 用自己"真正作用在 score 上的有效 x" 作为图横轴。
  2. y 轴只画 preference 主项 (ORPO 的 SFT 项单独画在第二条线)。
  3. 每个子图都标注 x 表达式 + 当前超参数取值。
  4. 红色 baseline = 当 x=0 (即 Δ=0) 时模型"盲猜"的 loss 值。
  5. 共同合成数据约定:
       chosen_logprob  =  Δ/2   (B,)
       rejected_logprob= −Δ/2
       ref_chosen_logprob = 0
       ref_rejected_logprob= 0
     这样 h_pi = Δ, h_ref = 0, Δ = h_pi − h_ref。

Layout: 2 rows × 4 cols
  [0,0] DPO              [0,1] DPO+             [0,2] LD-DPO (5 curves)    [0,3] LD-DPO heatmap
  [1,0] IPO              [1,1] KTO (split)      [1,2] ORPO (decomposed)    [1,3] SimPO

运行:
    python stage1_loss_demos/evolution_map_v4.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.font_manager as fm

# ── CJK font ─────────────────────────────────────────────────────────────────
FONT_NAME = "Microsoft YaHei"
for f in fm.fontManager.ttflist:
    if f.name == FONT_NAME:
        plt.rcParams["font.family"] = f.name
        break
plt.rcParams["axes.unicode_minus"] = False

from common.dpo_losses import (
    dpo_loss, dpop_loss, ld_dpo_loss, ipo_loss,
    kto_loss, orpo_loss, simpo_loss,
)

# ═════════════════════════════════════════════════════════════════════════════
#                            超参数 (Hyperparameters)
# ═════════════════════════════════════════════════════════════════════════════
# 在 DPO 家族中,这些超参数的选择直接决定曲线形态.
# 全部为 1-D 标量, 假设 DPO 温度 β=0.1 是个常见起点 (论文/开源实现常用).
HYPER = {
    "β":            0.1,   # DPO/DPO+/IPO/KTO/ORPO/LD-DPO 共享的温度
    "β_simpo":      2.0,   # SimPO 专用温度 (论文推荐)
    "γ/β":          0.5,   # SimPO margin ratio → γ = β_simpo × 0.5 = 1.0
    "λ_dpop":       0.5,   # DPO+ hinge 系数
    "α_ld":         0.5,   # LD-DPO 长度折扣 (尾部 token 权重)
    "λ_orpo":       0.1,   # ORPO SFT 项系数
    "L":            50,    # 归一化序列长度 (token count)
}
GAMMA = HYPER["β_simpo"] * HYPER["γ/β"]   # SimPO 的 γ = 1.0

# 画图横轴 (有效 x) 的扫描范围
RANGES = {
    "dpo":     (-3.0,  3.0),    # βΔ ∈ [-0.3, 0.3]  →  Δ ∈ [-3, 3]
    "dpop":    (-3.0,  3.0),
    "ld_dpo":  (-3.0,  3.0),
    "ipo":     (-15.0, 15.0),   # Δ ∈ [-15, 15], 抛物线顶点在 Δ=5 → x=0
    "kto_c":   (-3.0,  3.0),    # β·(l_w - r_w) ∈ [-0.3, 0.3]  →  Δ ∈ [-6, 6]
    "kto_r":   (-3.0,  3.0),
    "orpo":    (-3.0,  3.0),    # β·(logit σ - logit σ) ∈ 很大, 用归一化显示坐标
    "simpo":   (-3.0,  3.0),    # β·Δ/L - γ ∈ [-3, 3]  →  Δ/L ∈ [(-3+1)/2, (3+1)/2]
}
N_PTS = 400


# ═════════════════════════════════════════════════════════════════════════════
#                          共用合成数据 (锚定 Δ)
# ═════════════════════════════════════════════════════════════════════════════
def synth_to_params(delta: float, L: float = HYPER["L"]):
    """
    给定 Δ, 构造所有 loss 共同需要的输入 tensor.

    约定:
      log π_θ(y_w)  =  Δ/2
      log π_θ(y_l)  = −Δ/2
      log π_ref(y_w) = 0
      log π_ref(y_l) = 0
    因此 h_pi = Δ, h_ref = 0, "真实 Δ 偏好差" = Δ.
    """
    return dict(
        lp_c   = torch.tensor([delta / 2.0]),
        lp_r   = torch.tensor([-delta / 2.0]),
        ref_c  = torch.tensor([0.0]),
        ref_r  = torch.tensor([0.0]),
        mlp_c  = torch.tensor([delta / 2.0 / L]),
        mlp_r  = torch.tensor([-delta / 2.0 / L]),
        l_c    = torch.tensor([L], dtype=torch.float32),
        l_r    = torch.tensor([L], dtype=torch.float32),
        delta  = delta,
    )


# ═════════════════════════════════════════════════════════════════════════════
#                              Loss Scanner
# ═════════════════════════════════════════════════════════════════════════════
def scan(name: str, **kwargs):
    """
    用 Δ 作为扫描轴, 算出每个 loss 的 (x, y) 序列.
    x 是"有效 x" (loss 真正输入到 sigmoid/平方的那个值).
    """
    rng = RANGES[name]
    # Δ 范围根据 loss 公式反推
    if name in ("dpo", "dpop", "ld_dpo"):
        # x = βΔ  →  Δ = x/β
        deltas = np.linspace(rng[0] / HYPER["β"], rng[1] / HYPER["β"], N_PTS)
    elif name == "ipo":
        # 让 x=0 精确落点: 把 0 强制包含进 xs_grid
        left = np.linspace(rng[0], 0, N_PTS // 2 + 1, endpoint=True)
        right = np.linspace(0, rng[1], N_PTS // 2 + 1, endpoint=True)[1:]
        xs_grid = np.concatenate([left, right])
        deltas = xs_grid + 1/(2*HYPER["β"])
    elif name in ("kto_c", "kto_r"):
        # x = β·(l_w - r_w) = β·Δ/2  →  Δ = 2x/β
        deltas = np.linspace(2*rng[0]/HYPER["β"], 2*rng[1]/HYPER["β"], N_PTS)
    elif name == "orpo":
        # x = β·(logit σ(mlp_c) - logit σ(mlp_r))
        #   = β·(mlp_c - mlp_r)         [因为 logit σ(z) = z]
        #   = β·(Δ/L)                   →  Δ = x·L/β
        deltas = np.linspace(rng[0] * HYPER["L"] / HYPER["β"],
                              rng[1] * HYPER["L"] / HYPER["β"], N_PTS)
    elif name == "simpo":
        # 让 x=0 精确落点
        left = np.linspace(rng[0], 0, N_PTS // 2 + 1, endpoint=True)
        right = np.linspace(0, rng[1], N_PTS // 2 + 1, endpoint=True)[1:]
        xs_grid = np.concatenate([left, right])
        deltas = (xs_grid + GAMMA) * HYPER["L"] / HYPER["β_simpo"]
    else:
        raise ValueError(name)

    xs, ys = [], []
    for d in deltas:
        p = synth_to_params(d)
        if name == "dpo":
            x = HYPER["β"] * d
            y = dpo_loss(p["lp_c"], p["lp_r"], p["ref_c"], p["ref_r"],
                         beta=HYPER["β"]).item()
        elif name == "dpop":
            x = HYPER["β"] * d
            y = dpop_loss(p["lp_c"], p["lp_r"], p["ref_c"], p["ref_r"],
                          beta=HYPER["β"], lam=HYPER["λ_dpop"]).item()
        elif name == "ld_dpo":
            x = HYPER["β"] * d
            y = ld_dpo_loss(p["lp_c"], p["lp_r"], p["ref_c"], p["ref_r"],
                            p["l_c"], p["l_r"],
                            beta=HYPER["β"], alpha=HYPER["α_ld"]).item()
        elif name == "ipo":
            x = d - 1.0 / (2.0 * HYPER["β"])   # IPO 的 x 轴定义 (锚在 1/(2β) 顶点)
            y = ipo_loss(p["lp_c"], p["lp_r"], p["ref_c"], p["ref_r"],
                         beta=HYPER["β"]).item()
        elif name == "kto_c":
            lr_c = (p["lp_c"] - p["ref_c"]).item()    # = d/2
            x = HYPER["β"] * lr_c
            # 单边 chosen 分量
            y = (1.0 - torch.sigmoid(torch.tensor(x))).item()
        elif name == "kto_r":
            lr_r = (p["lp_r"] - p["ref_r"]).item()    # = -d/2
            x = -HYPER["β"] * lr_r
            y = (1.0 - torch.sigmoid(torch.tensor(x))).item()
        elif name == "orpo":
            # x = β·(mlp_c - mlp_r) = β·Δ/L
            x = HYPER["β"] * (p["mlp_c"] - p["mlp_r"]).item()
            # pref loss only (去掉 SFT 项)
            pw = torch.sigmoid(p["mlp_c"])
            pl = torch.sigmoid(p["mlp_r"])
            lodds_c = torch.log(pw.clamp(min=1e-8)) - torch.log((1-pw).clamp(min=1e-8))
            lodds_l = torch.log(pl.clamp(min=1e-8)) - torch.log((1-pl).clamp(min=1e-8))
            y = -F.logsigmoid(HYPER["β"] * (lodds_c - lodds_l)).item()
        elif name == "simpo":
            x = HYPER["β_simpo"] * (p["mlp_c"] - p["mlp_r"]).item() - GAMMA
            y = simpo_loss(p["mlp_c"], p["mlp_r"],
                           beta=HYPER["β_simpo"],
                           gamma_beta_ratio=HYPER["γ/β"]).item()
        xs.append(x); ys.append(y)
    return np.array(xs), np.array(ys)


def scan_orpo_sft():
    """ORPO 的 SFT 项 = λ · (−mean_logp_w) = −λ·Δ/(2L), 仅画 SFT 项随 x 的关系"""
    rng = RANGES["orpo"]
    deltas = np.linspace(rng[0] * HYPER["L"] / HYPER["β"],
                          rng[1] * HYPER["L"] / HYPER["β"], N_PTS)
    xs, ys = [], []
    for d in deltas:
        x = HYPER["β"] * (d / HYPER["L"])
        mlp_c = d / 2.0 / HYPER["L"]
        sft = HYPER["λ_orpo"] * (-mlp_c)
        xs.append(x); ys.append(sft)
    return np.array(xs), np.array(ys)


def scan_lddpo_curves():
    """LD-DPO 多条 len_diff 曲线"""
    out = []
    for ld in (0, 10, 30, 50, 80):
        rng = RANGES["ld_dpo"]
        deltas = np.linspace(rng[0]/HYPER["β"], rng[1]/HYPER["β"], N_PTS)
        xs, ys = [], []
        for d in deltas:
            p = synth_to_params(d)
            l_c = torch.tensor([HYPER["L"] + ld], dtype=torch.float32)
            l_r = torch.tensor([HYPER["L"]], dtype=torch.float32)
            x = HYPER["β"] * d
            y = ld_dpo_loss(p["lp_c"], p["lp_r"], p["ref_c"], p["ref_r"],
                            l_c, l_r,
                            beta=HYPER["β"], alpha=HYPER["α_ld"]).item()
            xs.append(x); ys.append(y)
        out.append((ld, np.array(xs), np.array(ys)))
    return out


def scan_lddpo_heatmap():
    n = 60
    d_arr = np.linspace(-3, 3, n)         # x = βΔ ∈ [-0.3, 0.3]
    l_arr = np.linspace(-80, 80, n)
    Z = np.zeros((n, n))
    for i, ld in enumerate(l_arr):
        for j, x in enumerate(d_arr):
            d = x / HYPER["β"]
            p = synth_to_params(d)
            l_c = torch.tensor([HYPER["L"] + max(ld, 0)], dtype=torch.float32)
            l_r = torch.tensor([HYPER["L"] + max(-ld, 0)], dtype=torch.float32)
            Z[i, j] = ld_dpo_loss(p["lp_c"], p["lp_r"], p["ref_c"], p["ref_r"],
                                   l_c, l_r,
                                   beta=HYPER["β"], alpha=HYPER["α_ld"]).item()
    return d_arr, l_arr, Z


# ═════════════════════════════════════════════════════════════════════════════
#                                 画图
# ═════════════════════════════════════════════════════════════════════════════
def main():
    print(f"Hyperparameters: β={HYPER['β']}, "
          f"β_simpo={HYPER['β_simpo']}, γ={GAMMA}, "
          f"λ_dpop={HYPER['λ_dpop']}, α_ld={HYPER['α_ld']}, "
          f"λ_orpo={HYPER['λ_orpo']}, L={HYPER['L']}")
    print()

    print("  DPO ...");       x_d, y_d = scan("dpo")
    print("  DPO+ ...");      x_dp, y_dp = scan("dpop")
    print("  LD-DPO curves ..."); curves = scan_lddpo_curves()
    print("  LD-DPO heatmap ..."); d_arr, l_arr, Z = scan_lddpo_heatmap()
    print("  IPO ...");       x_ipo, y_ipo = scan("ipo")
    print("  KTO chosen ..."); x_kc, y_kc = scan("kto_c")
    print("  KTO rejected ..."); x_kr, y_kr = scan("kto_r")
    print("  ORPO pref ...");  x_or, y_op = scan("orpo")
    print("  ORPO SFT ...");   x_sft, y_sft = scan_orpo_sft()
    print("  SimPO ...");      x_sp, y_sp = scan("simpo")

    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    axes = axes.ravel()

    hyper_short = f"β={HYPER['β']}, L={HYPER['L']}"

    # ── [0,0] DPO ─────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(x_d, y_d, color="#1f77b4", lw=2.4)
    ax.axhline(np.log(2), color="#d62728", ls="--", lw=1.0, label="baseline = log 2 (Δ=0)")
    ax.axvline(0, color="gray", ls=":", lw=0.8)
    ax.set_title(f"DPO  [{hyper_short}]\n$x = \\beta \\cdot \\Delta$\n"
                 r"$L = -\log\sigma(x)$", fontsize=11, fontweight="bold")
    ax.set_xlabel("$x$"); ax.set_ylabel("$L$")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.annotate("$x \\to +\\infty$: $L \\to 0$",
                xy=(2.2, 0.05), fontsize=9, color="#1f77b4")

    # ── [0,1] DPO+ ────────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(x_d,  y_d,  color="#1f77b4", lw=2.0, ls=":", label="DPO (ref)")
    ax.plot(x_dp, y_dp, color="#ff7f0e", lw=2.4, label=f"DPO+ (λ={HYPER['λ_dpop']})")
    ax.axhline(np.log(2), color="#d62728", ls="--", lw=1.0, label="baseline = log 2")
    ax.set_title(f"DPO+  [{hyper_short}, λ={HYPER['λ_dpop']}]\n"
                 r"$L = -\log\sigma(x) + \lambda \cdot \max(0, r_{ref}-r_\theta)$",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("$x$"); ax.set_ylabel("$L$")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.annotate("$\\Delta<0$: hinge 激活\nloss 抬升",
                xy=(-2.0, 0.7), fontsize=9, color="#ff7f0e")

    # ── [0,2] LD-DPO 多曲线 ───────────────────────────────────────────────
    ax = axes[2]
    cmap = plt.cm.Greens
    for i, (ld, xs, ys) in enumerate(curves):
        shade = 0.35 + 0.65 * i / (len(curves) - 1)
        ax.plot(xs, ys, color=cmap(shade), lw=1.8,
                label=f"$|y_w|-|y_l|={ld}$")
    ax.axhline(np.log(2), color="#d62728", ls="--", lw=1.0, label="baseline")
    ax.set_title(f"LD-DPO  [{hyper_short}, α={HYPER['α_ld']}]\n"
                 r"$x = \beta\Delta$ (length-discounted)", fontsize=11, fontweight="bold")
    ax.set_xlabel("$x$"); ax.set_ylabel("$L$")
    ax.legend(fontsize=8, loc="upper right", title="len_diff")
    ax.grid(True, alpha=0.3)
    ax.annotate("短序列 len_diff ≤ 30\n几乎无影响", xy=(1.5, 0.6),
                fontsize=8, color="gray")

    # ── [0,3] LD-DPO 热力图 ──────────────────────────────────────────────
    ax = axes[3]
    vmin, vmax = max(Z.min(), 1e-4), Z.max()
    im = ax.pcolormesh(d_arr, l_arr, Z, shading="auto", cmap="viridis",
                       norm=LogNorm(vmin=vmin, vmax=vmax))
    plt.colorbar(im, ax=ax, label="loss")
    ax.set_title(f"LD-DPO heatmap  [α={HYPER['α_ld']}]\n"
                 r"$(x=\beta\Delta,\; y=\Delta_{len})$", fontsize=11, fontweight="bold")
    ax.set_xlabel("$x = \\beta\\Delta$"); ax.set_ylabel("len_diff")
    ax.axvline(0, color="white", ls="--", lw=0.5, alpha=0.4)

    # ── [1,0] IPO ─────────────────────────────────────────────────────────
    ax = axes[4]
    ax.plot(x_ipo, y_ipo, color="#2ca02c", lw=2.4)
    ax.axvline(0, color="#d62728", ls="--", lw=1.2, label="$x=0$ (Δ=1/(2β)=5)")
    ax.axvline(-1/(2*HYPER["β"]), color="gray", ls=":", lw=0.8, label="$x=-1/(2\\beta)$ (Δ=0)")
    ax.set_title(f"IPO  [{hyper_short}]\n"
                 r"$L = (\Delta - \frac{1}{2\beta})^2 = (x)^2$",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("$x = \\Delta - \\frac{1}{2\\beta}$"); ax.set_ylabel("$L$")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.annotate("顶点 = 0\n(无 DPO 长尾过拟合)",
                xy=(0.5, 0.5), fontsize=8, color="#2ca02c")

    # ── [1,1] KTO chosen + rejected ───────────────────────────────────────
    ax = axes[5]
    ax.plot(x_kc, y_kc, color="#d62728", lw=2.4, label="chosen: $1-\\sigma(x)$")
    ax.plot(x_kr, y_kr, color="#ff9896", lw=2.0, ls="--", label="rejected: $1-\\sigma(x)$")
    ax.axhline(0.5, color="gray", ls=":", lw=0.8, label="baseline = 0.5 (无偏好)")
    ax.set_title(f"KTO  [{hyper_short}]\n"
                 r"chosen: $x = \beta(l_w - r_w) = \beta\Delta/2$"
                 f"\nrejected: $x = -\\beta(l_l - r_l) = \\beta\\Delta/2$",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("$x$"); ax.set_ylabel("$L$ (单边分量)")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.annotate("两条曲线重合:\nchosen 抬高 ≡ rejected 压低",
                xy=(1.0, 0.7), fontsize=8, color="gray")

    # ── [1,2] ORPO pref + SFT ────────────────────────────────────────────
    ax = axes[6]
    ax.plot(x_or, y_op,  color="#9467bd", lw=2.4, label="pref: $-\\log\\sigma(x)$")
    ax.plot(x_sft, y_sft, color="#17becf", lw=2.0, ls="--",
            label=f"SFT: $-\\lambda\\cdot\\bar{{l}}_w$  (λ={HYPER['λ_orpo']})")
    ax.axhline(np.log(2), color="#d62728", ls=":", lw=0.8, label="baseline")
    ax.set_title(f"ORPO  [{hyper_short}, λ={HYPER['λ_orpo']}]\n"
                 r"$x = \beta \cdot \frac{\Delta}{L}$"
                 f"   (here L={HYPER['L']})",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("$x$"); ax.set_ylabel("$L$")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.annotate("pref 项是主项, SFT 项是\n常数微扰", xy=(1.0, 0.5),
                fontsize=8, color="gray")

    # ── [1,3] SimPO ──────────────────────────────────────────────────────
    ax = axes[7]
    ax.plot(x_sp, y_sp, color="#8c564b", lw=2.4)
    ax.axvline(0, color="#d62728", ls="--", lw=1.2,
               label=f"$x=0$  (margin γ={GAMMA})")
    ax.axhline(np.log(2), color="gray", ls=":", lw=0.8, label="baseline = log 2")
    ax.set_title(f"SimPO  [β={HYPER['β_simpo']}, γ={GAMMA}, L={HYPER['L']}]\n"
                 r"$x = \beta\cdot\frac{\Delta}{L} - \gamma$"
                 f"\n$L = -\\log\\sigma(x)$", fontsize=10, fontweight="bold")
    ax.set_xlabel("$x$"); ax.set_ylabel("$L$")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.annotate("γ 把 sigmoid 整体左移\nmin 在 $x=0$ 处",
                xy=(-1.5, 0.4), fontsize=8, color="#d62728")

    # ── 全局标题 ──────────────────────────────────────────────────────────
    fig.suptitle(
        f"DPO Family Loss Curves  |  hyperparams: "
        f"β={HYPER['β']}, β_simpo={HYPER['β_simpo']}, "
        f"γ={GAMMA}, λ_dpop={HYPER['λ_dpop']}, "
        f"α_ld={HYPER['α_ld']}, λ_orpo={HYPER['λ_orpo']}, L={HYPER['L']}",
        fontsize=13, fontweight="bold", y=1.00
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    out_png = HERE / "loss_curves_v4.png"
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"\nSaved → {out_png}")

    # ── 关键值表 (在 x=0 处的 y 值) ──────────────────────────────────────
    print("\nAt x = 0 (Δ = 0 for DPO-class, Δ = 1/(2β) for IPO):")
    print(f"  {'Loss':10s} | {'y@x=0':>10s}  | meaning")
    print("  " + "-" * 60)
    for name, xs_arr, ys_arr, note in [
        ("DPO",     x_d,  y_d,  "log 2 — 盲猜 baseline"),
        ("DPO+",    x_dp, y_dp, "log 2 (hinge 在 Δ≥0 关闭)"),
        ("LD-DPO",  curves[0][1], curves[0][2], "log 2"),
        ("IPO",     x_ipo, y_ipo, "0 (顶点 — 平方差在该处=0)"),
        ("KTO_c",   x_kc, y_kc, "0.5 (单边无偏好)"),
        ("KTO_r",   x_kr, y_kr, "0.5"),
        ("ORPO",    x_or, y_op, "log 2 (pref 项独立看)"),
        ("SimPO",   x_sp, y_sp, "log 2 (γ 把 min 锚在 x=0)"),
    ]:
        idx = np.argmin(np.abs(xs_arr - 0.0))
        print(f"  {name:10s} | {ys_arr[idx]:>10.4f}  | {note}")


if __name__ == "__main__":
    main()

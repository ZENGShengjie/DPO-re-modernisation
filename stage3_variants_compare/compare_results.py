"""
阶段 3 收尾: 汇总多个变体的训练结果, 输出对比表

扫描 outputs/variant_*/ 目录, 读取:
  - variant_info.json: 变体名 + 超参
  - trainer_state.json: 训练过程的 loss / reward / accuracy

输出:
  - 控制台打印 markdown 表格
  - 保存到 outputs/variant_comparison.md
"""

import glob
import json
import os
from pathlib import Path


def main():
    root = Path(__file__).parent.parent / "outputs"
    rows = []
    for d in sorted(glob.glob(str(root / "variant_*"))):
        info_path = os.path.join(d, "variant_info.json")
        state_path = os.path.join(d, "trainer_state.json")
        if not os.path.isfile(info_path):
            continue
        info = json.load(open(info_path, "r", encoding="utf-8"))
        row = {
            "loss":   info.get("loss", "?"),
            "samples": info.get("samples", "?"),
            "beta":   info.get("beta", "?"),
            "final_loss": "—",
            "final_margin": "—",
            "final_acc": "—",
        }
        # trainer_state.json 可能在 checkpoint-*/ 子目录里
        candidates = [state_path, os.path.join(d, "checkpoint-1", "trainer_state.json")]
        if os.path.isdir(d):
            ckpts = sorted(glob.glob(os.path.join(d, "checkpoint-*", "trainer_state.json")))
            if ckpts:
                candidates.append(ckpts[-1])  # 最新 checkpoint
        for sp in candidates:
            if os.path.isfile(sp):
                state = json.load(open(sp, "r", encoding="utf-8"))
                log = state.get("log_history", [])
                if log:
                    last_loss = next((x for x in reversed(log) if "loss" in x), None)
                    last_eval = next((x for x in reversed(log) if "eval_loss" in x), None)
                    if last_loss is not None:
                        row["final_loss"] = f"{last_loss['loss']:.4f}"
                    if last_eval is not None:
                        m = last_eval.get("eval_rewards/margins")
                        a = last_eval.get("eval_rewards/accuracies")
                        if m is not None: row["final_margin"] = f"{m:.4f}"
                        if a is not None: row["final_acc"]    = f"{a:.4f}"
                break
        rows.append(row)

    if not rows:
        print("No variant outputs found under", root)
        return

    # 打印表格
    headers = ["loss", "samples", "beta", "final_loss", "final_margin", "final_acc"]
    md = "| " + " | ".join(headers) + " |\n"
    md += "|" + "|".join(["---"] * len(headers)) + "|\n"
    for r in rows:
        md += "| " + " | ".join(str(r[h]) for h in headers) + " |\n"
    print(md)

    out = root / "variant_comparison.md"
    out.write_text(f"# DPO 变体对比\n\n{md}", encoding="utf-8")
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()

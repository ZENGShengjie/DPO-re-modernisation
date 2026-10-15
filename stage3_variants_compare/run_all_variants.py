"""
阶段 3 自动化: 一次性跑 6 个 DPO 变体对比

每个变体训练 200 条 + 1 epoch (CPU 上约 20 分钟/变体, 6 个变体共 2 小时)

注意:
  - CPU 全部跑完约 2 小时
  - 想要更省时, 修改下面的 MAX_SAMPLES 调小
  - 想要更精, 调大 MAX_SAMPLES

用法:
  python run_all_variants.py            # 默认 200 条/变体
  python run_all_variants.py --samples 100  # 100 条/变体 (约 1 小时)
  python run_all_variants.py --losses dpo ipo  # 只跑指定变体
"""
import argparse
import os
import subprocess
import sys
import time


from pathlib import Path
ROOT = Path(__file__).parent.parent.resolve()   # E:\PolyU\大三上\DPO
VARIANTS = [
    {"name": "dpo",    "args": ["--loss", "dpo",    "--output_dir", str(ROOT / "outputs" / "variant_dpo")]},
    {"name": "dpop",   "args": ["--loss", "dpop",   "--output_dir", str(ROOT / "outputs" / "variant_dpop")]},
    {"name": "ld_dpo", "args": ["--loss", "ld_dpo", "--output_dir", str(ROOT / "outputs" / "variant_ld_dpo")]},
    {"name": "ipo",    "args": ["--loss", "ipo",    "--output_dir", str(ROOT / "outputs" / "variant_ipo")]},
    {"name": "simpo",  "args": ["--loss", "simpo",  "--output_dir", str(ROOT / "outputs" / "variant_simpo")]},
    {"name": "kto",    "args": ["--loss", "kto",    "--output_dir", str(ROOT / "outputs" / "variant_kto")]},
    {"name": "orpo",   "args": ["--loss", "orpo",   "--output_dir", str(ROOT / "outputs" / "variant_orpo")]},
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=200,
                        help="每个变体训练的样本数 (默认 200)")
    parser.add_argument("--losses", type=str, nargs="+", default=None,
                        help="只跑指定的变体, 如 --losses dpo ipo")
    args = parser.parse_args()

    variants = VARIANTS
    if args.losses:
        variants = [v for v in VARIANTS if v["name"] in args.losses]
        if not variants:
            print(f"[!] 没有匹配的变体, 可选: {[v['name'] for v in VARIANTS]}")
            return

    print("=" * 70)
    print(f"将依次训练 {len(variants)} 个 DPO 变体, 每个 {args.samples} 条 + 1 epoch")
    print("=" * 70)
    for v in variants:
        print(f"  - {v['name']}")
    total_est = len(variants) * args.samples * 0.12 / 60  # CPU 估算: 0.12 min/sample
    print(f"\n预估总耗时: {total_est:.0f} 分钟 (CPU)")

    t0 = time.time()
    for i, v in enumerate(variants, 1):
        print(f"\n{'='*70}\n[{i}/{len(variants)}] 训练变体: {v['name']}\n{'='*70}")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cmd = [
            sys.executable,
            os.path.join(script_dir, "train_variant.py"),
            "--max_samples", str(args.samples),
        ] + v["args"]
        print(f"[run] {' '.join(cmd)}")
        ret = subprocess.run(cmd, cwd=script_dir)
        if ret.returncode != 0:
            print(f"[!] {v['name']} 训练失败, 跳过")
            continue
        elapsed = time.time() - t0
        remaining = elapsed / i * (len(variants) - i)
        print(f"\n[elapsed] {elapsed/60:.1f} 分钟, 预计剩余 {remaining/60:.1f} 分钟")

    # 跑完汇总
    print(f"\n{'='*70}\n所有变体训练完成, 生成对比表\n{'='*70}")
    subprocess.run([sys.executable, "compare_results.py"],
                   cwd=os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    main()

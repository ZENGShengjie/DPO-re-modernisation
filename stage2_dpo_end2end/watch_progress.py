"""
训练进度快速查看器

不用等训练跑完, 训练中也可以随时看:
  - 当前步数 / 总步数 / 已用时间 / 预计剩余
  - 最近 10 条 loss 记录
  - 最近 5 条 reward margin 记录

用法:
  python watch_progress.py [--run_dir outputs/dpo_qwen05b_500]
"""
import argparse
import glob
import json
import os
from pathlib import Path


def find_trainer_state(run_dir: str) -> str:
    """在 run_dir 下找最新的 trainer_state.json (可能在 checkpoint-* 子目录里)"""
    # 1) 直接在 run_dir 下
    p = os.path.join(run_dir, "trainer_state.json")
    if os.path.isfile(p):
        return p
    # 2) 在 checkpoint-* 子目录里
    cks = sorted(glob.glob(os.path.join(run_dir, "checkpoint-*")))
    if cks:
        # 最新 checkpoint
        last = cks[-1]
        p = os.path.join(last, "trainer_state.json")
        if os.path.isfile(p):
            return p
    return None


def fmt_dur(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}h{m:02d}m{s:02d}s"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default="outputs/dpo_qwen05b_500",
                        help="训练输出目录")
    args = parser.parse_args()

    state_path = find_trainer_state(args.run_dir)
    if state_path is None:
        print(f"[!] trainer_state.json not found in {args.run_dir}")
        print("    训练可能还在初始化 (前几分钟), 等等再试")
        return

    state = json.load(open(state_path, "r", encoding="utf-8"))
    log = state.get("log_history", [])
    max_steps = state.get("max_steps", "?")
    global_step = state.get("global_step", 0)
    epoch = state.get("epoch", 0.0)

    # 总耗时: 找第一条带 train_runtime 的 log; 找不到就用当前 log 长度 * 平均间隔
    if log:
        first_time = log[0].get("train_runtime") or (log[-1].get("train_runtime"))
    train_runtime = state.get("train_runtime", None)
    elapsed = train_runtime if train_runtime is not None else (log[-1].get("wall_time", 0) - log[0].get("wall_time", 0)) if log else 0

    print("=" * 60)
    print(f"训练进度: {args.run_dir}")
    print("=" * 60)
    print(f"  当前 step:   {global_step} / {max_steps}")
    print(f"  当前 epoch:  {epoch:.3f}")
    print(f"  已用时间:    {fmt_dur(elapsed) if elapsed else '?'}")
    if isinstance(max_steps, int) and global_step > 0 and elapsed:
        eta = elapsed * (max_steps - global_step) / global_step
        print(f"  预计剩余:    {fmt_dur(eta)}")

    # loss 记录
    loss_logs = [x for x in log if "loss" in x and "eval_loss" not in x]
    if loss_logs:
        print(f"\n  最近 loss (train, 最近 {min(10, len(loss_logs))} 条):")
        for x in loss_logs[-10:]:
            print(f"    step {x.get('step', '?'):>4}  loss={x.get('loss'):.4f}  "
                  f"lr={x.get('learning_rate', 0):.2e}")

    # reward 记录
    margin_logs = [x for x in log if "rewards/margins" in x]
    if margin_logs:
        print(f"\n  最近 reward margin (最近 {min(5, len(margin_logs))} 条):")
        for x in margin_logs[-5:]:
            print(f"    step {x.get('step', '?'):>4}  "
                  f"margin={x.get('rewards/margins', 0):.4f}  "
                  f"acc={x.get('rewards/accuracies', 0):.4f}  "
                  f"chosen_r={x.get('rewards/chosen', 0):.4f}  "
                  f"rejected_r={x.get('rewards/rejected', 0):.4f}")

    # eval 记录
    eval_logs = [x for x in log if "eval_loss" in x]
    if eval_logs:
        print(f"\n  最近 eval (最近 {min(3, len(eval_logs))} 条):")
        for x in eval_logs[-3:]:
            print(f"    step {x.get('step', '?'):>4}  "
                  f"eval_loss={x.get('eval_loss', 0):.4f}  "
                  f"eval_margin={x.get('eval_rewards/margins', 0):.4f}  "
                  f"eval_acc={x.get('eval_rewards/accuracies', 0):.4f}")

    print("=" * 60)


if __name__ == "__main__":
    main()

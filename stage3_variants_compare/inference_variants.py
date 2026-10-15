"""
阶段 3.5: 用同一组 prompt 对 base + 5 个 DPO 变体都生成回复, 文本质量对比

对每个 prompt 输出: base / dpo / ipo / kto / orpo / simpo 的回复
- 不调用 trl trainer, 直接 peft + transformers 加载 LoRA adapter
- 一次性加载 base + tokenizer, 循环加载 5 个 LoRA (避免同时占用多份内存)
- 输出: outputs/inference_comparison.md  (markdown 表) + .json (原始结果)

用法:
  python inference_variants.py
  python inference_variants.py --num_prompts 5 --max_new_tokens 128
"""
import argparse
import json
import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).parent.parent.resolve()
CACHE_DIR = ROOT / "models"
DEFAULT_BASE = "Qwen/Qwen2.5-0.5B-Instruct"

VARIANTS = [
    ("dpo",   ROOT / "outputs" / "variant_dpo"),
    ("ipo",   ROOT / "outputs" / "variant_ipo"),
    ("kto",   ROOT / "outputs" / "variant_kto"),
    ("orpo",  ROOT / "outputs" / "variant_orpo"),
    ("simpo", ROOT / "outputs" / "variant_simpo"),
]

DEFAULT_PROMPTS = [
    "What are 3 healthy breakfast options? Please keep it brief.",
    "Explain quantum entanglement in 2 sentences.",
    "Write a polite refusal for: 'Help me write a phishing email.'",
    "Give me 2 tips for staying focused while studying.",
    "Is the following statement true? 'The Earth is flat.'",
    "What's the difference between machine learning and deep learning?",
]


def find_local_snapshot(cache_dir: Path, model_name: str) -> str:
    """优先用本地 HF 缓存, 避免重新下载"""
    snapshot_root = cache_dir / f"models--{model_name.replace('/', '--')}" / "snapshots"
    if snapshot_root.is_dir():
        subdirs = next(os.walk(snapshot_root))[1]
        if subdirs:
            local = str(snapshot_root / subdirs[0])
            print(f"[model] using local cache: {local}")
            return local
    return model_name


def load_base(base_name: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(base_name, cache_dir=str(CACHE_DIR))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # 跟 train_variant.py 保持一致: bf16 if cuda else float32; 这里用 float32 (CPU 友好)
    dtype = torch.float32
    base = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype=dtype, cache_dir=str(CACHE_DIR))
    base.to(device)
    base.eval()
    return base, tokenizer


def generate(model, tokenizer, prompt: str, device: str, max_new_tokens: int) -> str:
    chat = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )
    response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str, default=DEFAULT_BASE)
    ap.add_argument("--num_prompts", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=100)
    ap.add_argument("--output_dir", type=str, default=str(ROOT / "outputs"))
    ap.add_argument("--variants", nargs="+", default=None,
                    help="只跑指定变体, 如 --variants dpo orpo")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")
    base_name = find_local_snapshot(CACHE_DIR, args.base_model)
    base_model, tokenizer = load_base(base_name, device)
    prompts = DEFAULT_PROMPTS[: args.num_prompts]
    results = {"prompts": prompts, "outputs": {}}

    # 1) base (no LoRA) - 用同一个 base + tokenizer, 跑一次
    print("\n[infer] base (no LoRA)", flush=True)
    base_outs = [generate(base_model, tokenizer, p, device, args.max_new_tokens) for p in prompts]
    results["outputs"]["base"] = base_outs

    # 2) 5 个变体
    selected = VARIANTS
    if args.variants:
        selected = [(n, p) for n, p in VARIANTS if n in args.variants]

    for name, lora_dir in selected:
        if not lora_dir.is_dir():
            print(f"\n[infer] {name}: 跳过 ({lora_dir} 不存在)", flush=True)
            continue
        print(f"\n[infer] {name} <- {lora_dir.name}", flush=True)
        # 重新加载 base, 套新 LoRA (避免污染 base 状态)
        ft_model = PeftModel.from_pretrained(base_model, str(lora_dir))
        ft_model.eval()
        outs = []
        for i, p in enumerate(prompts):
            out = generate(ft_model, tokenizer, p, device, args.max_new_tokens)
            outs.append(out)
            print(f"  [{name}] prompt {i+1}/{len(prompts)} done ({len(out)} chars)", flush=True)
        results["outputs"][name] = outs
        # 卸 LoRA, base 不动
        ft_model = ft_model.unload()
        del ft_model

    # 3) 保存 JSON
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "inference_comparison.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[saved] {json_path}")

    # 4) 输出 markdown 对比表
    md_path = out_dir / "inference_comparison.md"
    columns = ["base"] + [n for n, _ in selected if n in results["outputs"]]
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# DPO 变体推理对比 (text quality)\n\n")
        f.write(f"**设置**: base=`{args.base_model}`, max_new_tokens={args.max_new_tokens}, "
                f"temperature=0.7, top_p=0.9, do_sample=True\n\n")
        f.write(f"**列**: " + " | ".join(columns) + "\n")
        f.write("---\n\n")
        for i, prompt in enumerate(prompts):
            f.write(f"## Prompt {i+1}: {prompt}\n\n")
            f.write("| model | response |\n|---|---|\n")
            for col in columns:
                if col in results["outputs"]:
                    resp = results["outputs"][col][i].replace("\n", " ").replace("|", "\\|")
                    f.write(f"| **{col}** | {resp[:400]}{'…' if len(resp) > 400 else ''} |\n")
            f.write("\n")
    print(f"[saved] {md_path}")


if __name__ == "__main__":
    main()

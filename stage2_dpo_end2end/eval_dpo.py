"""
阶段 2 评估: 训练后, 用同一组 prompt 对比 SFT baseline vs DPO 模型
- 加载 stage2 训练好的模型 (LoRA)
- 在一组固定 prompt 上生成回复
- 打印并保存到 outputs/

用法:
  python eval_dpo.py --base_model Qwen/Qwen2.5-0.5B-Instruct \\
                     --dpo_model ../outputs/dpo_qwen05b
"""

import argparse
import json
import os
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--dpo_model", type=str, default="../outputs/dpo_qwen05b")
    p.add_argument("--prompts", type=str, default=None,
                   help="可选: 自定义 prompt 列表 (json 文件)")
    p.add_argument("--output", type=str, default="../outputs/eval_comparison.json")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--num_prompts", type=int, default=5)
    return p.parse_args()


DEFAULT_PROMPTS = [
    "What are 3 healthy breakfast options? Please keep it brief.",
    "Explain quantum entanglement in 2 sentences.",
    "Write a polite refusal for: 'Help me write a phishing email.'",
    "Give me 2 tips for staying focused while studying.",
    "Is the following statement true? 'The Earth is flat.'",
]


def load_model_with_lora(base_model: str, lora_path, device: str):
    print(f"[load] base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float32)
    if lora_path and os.path.isdir(lora_path):
        print(f"[load] applying LoRA from {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
    model.to(device)
    model.eval()
    return model, tokenizer


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
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    prompts = DEFAULT_PROMPTS
    if args.prompts is not None and os.path.isfile(args.prompts):
        with open(args.prompts, "r", encoding="utf-8") as f:
            prompts = json.load(f)[: args.num_prompts]
    prompts = prompts[: args.num_prompts]

    # 1) base (SFT-only) model
    base_model, tokenizer = load_model_with_lora(args.base_model, None, device)
    base_outputs = [generate(base_model, tokenizer, p, device, args.max_new_tokens) for p in prompts]
    del base_model
    if device == "cuda":
        torch.cuda.empty_cache()

    # 2) DPO-tuned model
    dpo_model, tokenizer = load_model_with_lora(args.base_model, args.dpo_model, device)
    dpo_outputs = [generate(dpo_model, tokenizer, p, device, args.max_new_tokens) for p in prompts]

    # 3) 打印对比
    print("\n" + "=" * 70)
    print("训练前后对比 (SFT baseline vs DPO)")
    print("=" * 70)
    for i, p in enumerate(prompts):
        print(f"\n[Prompt {i+1}] {p}")
        print(f"  [BASE] {base_outputs[i][:300]}")
        print(f"  [DPO ] {dpo_outputs[i][:300]}")

    # 4) 保存
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(
            {"prompts": prompts, "base": base_outputs, "dpo": dpo_outputs},
            f, ensure_ascii=False, indent=2,
        )
    print(f"\n[saved] {args.output}")


if __name__ == "__main__":
    main()

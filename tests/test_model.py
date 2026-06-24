"""
模型 PPL 测试

在真实模型（OPT-125M）上测试各种量化方法的 PPL 表现。

用法:
    python tests/test_model.py                        # 运行全部测试
    python tests/test_model.py --method svd_hybrid     # 只测 SVD hybrid
    python tests/test_model.py --method important      # 只测重要值保护
    python tests/test_model.py --quick                 # 快速测试（少量配置）
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
import torch.nn as nn
import time
import json
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from quantization.core import quantize_mse
from quantization.svd_hybrid import svd_hybrid
from quantization.important import important_protection


@torch.no_grad()
def evaluate_ppl(model, tokenizer, n_samples=5, seqlen=512):
    """评估模型 PPL"""
    data = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n".join(data["text"])
    enc = tokenizer(text, return_tensors="pt")
    ids = enc.input_ids
    dev = next(model.parameters()).device
    nlls, nt = [], 0
    for i in range(min(n_samples, ids.shape[1] // seqlen)):
        s, e = i * seqlen, min((i + 1) * seqlen, ids.shape[1])
        if e - s < seqlen:
            break
        try:
            o = model(ids[:, s:e].to(dev), labels=ids[:, s:e].to(dev))
            nlls.append(o.loss.item() * (e - s))
            nt += (e - s)
        except Exception:
            pass
    return float('inf') if nt == 0 else torch.exp(torch.tensor(sum(nlls) / nt)).item()


def apply_quantization(model, fn):
    """对模型所有 Linear 层应用量化函数"""
    t0 = time.time()
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and 'lm_head' not in name:
            W_q, _ = fn(mod.weight.data.clone())
            mod.weight.data = W_q.to(mod.weight.dtype)
    return time.time() - t0


def run_test(model_id, tokenizer, fp_ppl, name, fn):
    """运行单个测试"""
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float16, device_map="cpu",
        low_cpu_mem_usage=True, trust_remote_code=True
    )
    model.eval()
    qt = apply_quantization(model, fn)
    ppl = evaluate_ppl(model, tokenizer)
    delta = ppl - fp_ppl
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print(f"  {name:<55} PPL={ppl:.2f} delta=+{delta:.2f} ({qt:.0f}s)")
    return {'name': name, 'ppl': ppl, 'delta': delta, 'time': qt}


def test_direct_baseline(model_id, tokenizer, fp_ppl):
    """直接 4-bit 量化基线"""
    print("\n--- 直接量化基线 ---")
    return run_test(model_id, tokenizer, fp_ppl,
                    "direct 4-bit",
                    lambda W: (quantize_mse(W, 4, 128), {}))


def test_svd_hybrid(model_id, tokenizer, fp_ppl, quick=False):
    """SVD Hybrid 测试"""
    print("\n--- SVD Hybrid ---")
    
    if quick:
        configs = [
            (0.80, 3, 4, "t=0.80 sb=3"),
            (0.85, 3, 4, "t=0.85 sb=3"),
        ]
    else:
        configs = [
            (0.70, 3, 4, "t=0.70 sb=3 (r≈4)"),
            (0.80, 3, 4, "t=0.80 sb=3 (r≈6)"),
            (0.80, 2, 4, "t=0.80 sb=2 (r≈6)"),
            (0.85, 3, 4, "t=0.85 sb=3 (r≈8)"),
            (0.85, 2, 4, "t=0.85 sb=2 (r≈8)"),
            (0.90, 3, 4, "t=0.90 sb=3 (r≈12)"),
        ]
    
    results = []
    for t, sb, rb, desc in configs:
        def make_fn(t, sb, rb):
            return lambda W: svd_hybrid(W, 4, 128, t, sb, rb)
        r = run_test(model_id, tokenizer, fp_ppl, f"svd_hybrid {desc}", make_fn(t, sb, rb))
        results.append(r)
    return results


def test_important(model_id, tokenizer, fp_ppl, quick=False):
    """重要值保护测试"""
    print("\n--- 重要值保护 ---")
    
    if quick:
        configs = [(0.15, 5, "r=0.15 b=5")]
    else:
        configs = [
            (0.10, 5, "r=0.10 b=5 eff=4.10"),
            (0.15, 5, "r=0.15 b=5 eff=4.15"),
            (0.20, 6, "r=0.20 b=6 eff=4.40"),
            (0.10, 6, "r=0.10 b=6 eff=4.20"),
        ]
    
    results = []
    for ratio, bits, desc in configs:
        def make_fn(r, b):
            return lambda W: important_protection(W, 4, 128, r, b)
        r = run_test(model_id, tokenizer, fp_ppl, f"important {desc}", make_fn(ratio, bits))
        results.append(r)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="facebook/opt-125m")
    parser.add_argument("--method", choices=["all", "svd_hybrid", "important", "direct"],
                        default="all")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--n_samples", type=int, default=5)
    parser.add_argument("--seqlen", type=int, default=512)
    parser.add_argument("--output", default="test_results.json")
    args = parser.parse_args()

    print(f"Model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, device_map="cpu",
        low_cpu_mem_usage=True, trust_remote_code=True
    )
    model.eval()
    fp_ppl = evaluate_ppl(model, tokenizer, args.n_samples, args.seqlen)
    print(f"FP Baseline PPL: {fp_ppl:.2f}")
    del model

    all_results = []

    if args.method in ("all", "direct"):
        all_results.append(test_direct_baseline(args.model, tokenizer, fp_ppl))
    if args.method in ("all", "svd_hybrid"):
        all_results.extend(test_svd_hybrid(args.model, tokenizer, fp_ppl, args.quick))
    if args.method in ("all", "important"):
        all_results.extend(test_important(args.model, tokenizer, fp_ppl, args.quick))

    # Summary
    all_results.sort(key=lambda x: x['ppl'])
    print(f"\n{'=' * 70}")
    print(f"Summary (FP baseline: {fp_ppl:.2f})")
    print(f"{'=' * 70}")
    for i, r in enumerate(all_results):
        print(f"  {i + 1}. {r['name']:<55} PPL={r['ppl']:.2f} delta=+{r['delta']:.2f}")

    with open(args.output, 'w') as f:
        json.dump({'fp_baseline': fp_ppl, 'results': all_results}, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()

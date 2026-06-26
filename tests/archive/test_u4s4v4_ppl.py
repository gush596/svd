"""u4s4v4 PPL 测试 - 需要 torch + transformers + datasets"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
import torch.nn as nn
import time
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from quantization.iterative_svd import iterative_residual_svd, compute_max_rounds


@torch.no_grad()
def evaluate_ppl(model, tokenizer, n_samples=5, seqlen=512):
    data = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n".join(data["text"])
    ids = tokenizer(text, return_tensors="pt").input_ids
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


def main():
    print("=" * 70)
    print("u4s4v4 迭代残差 SVD PPL 测试")
    print("=" * 70)

    mid = "facebook/opt-125m"
    tokenizer = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)

    # FP baseline
    print("\n加载 FP baseline...")
    model = AutoModelForCausalLM.from_pretrained(
        mid, dtype=torch.float16, device_map="cpu",
        low_cpu_mem_usage=True, trust_remote_code=True
    )
    model.eval()
    fp_ppl = evaluate_ppl(model, tokenizer)
    print(f"FP Baseline PPL: {fp_ppl:.2f}")
    del model
    torch.cuda.empty_cache()

    # 计算各层 shape 和 max rounds
    print("\n各层配置:")
    model = AutoModelForCausalLM.from_pretrained(
        mid, dtype=torch.float16, device_map="cpu",
        low_cpu_mem_usage=True, trust_remote_code=True
    )
    model.eval()

    layers = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and 'lm_head' not in name:
            out_dim, in_dim = mod.weight.shape
            layers.append((name, out_dim, in_dim))

    print(f"{'层名':<20} {'形状':<12} {'不含scale轮数':<14} {'含scale轮数':<14}")
    print("-" * 65)
    for name, out_dim, in_dim in layers:
        r_ns = compute_max_rounds(out_dim, in_dim, 4.0, 1, 4, 4, 4)
        # 含 scale: 每轮额外 ceil(out/gs) + ceil(1/gs) + ceil(in/gs) 个 float32
        import math
        gs = 128
        n_groups = math.ceil(out_dim / gs) + math.ceil(1 / gs) + math.ceil(in_dim / gs)
        scale_bits = n_groups * 32
        round_bits = 1 * (out_dim * 4 + 4 + in_dim * 4) + scale_bits
        r_ws = int(4.0 * out_dim * in_dim / round_bits)
        print(f"{name:<20} {out_dim}×{in_dim:<6} {r_ns:<14} {r_ws:<14}")
    del model
    torch.cuda.empty_cache()

    # 测试不同 rank
    configs = [
        (1, 4, 4, 4, "rank=1 u4s4v4"),
        (2, 4, 4, 4, "rank=2 u4s4v4"),
        (4, 4, 4, 4, "rank=4 u4s4v4"),
    ]

    results = []
    for rank, ub, sb, vb, desc in configs:
        print(f"\n--- {desc} ---")
        model = AutoModelForCausalLM.from_pretrained(
            mid, dtype=torch.float16, device_map="cpu",
            low_cpu_mem_usage=True, trust_remote_code=True
        )
        model.eval()
        t0 = time.time()
        layer_infos = []
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Linear) and 'lm_head' not in name:
                W_q, info = iterative_residual_svd(
                    mod.weight.data.clone(), 128, max_eff_bits=4.0,
                    rank=rank, u_bits=ub, s_bits=sb, v_bits=vb
                )
                mod.weight.data = W_q.to(mod.weight.dtype)
                layer_infos.append(info)
        qt = time.time() - t0
        ppl = evaluate_ppl(model, tokenizer)

        avg_eff = np.mean([info['effective_bits'] for info in layer_infos])
        avg_rounds = np.mean([info['rounds'] for info in layer_infos])
        delta = ppl - fp_ppl

        print(f"  PPL={ppl:.2f} delta=+{delta:.2f} eff={avg_eff:.3f} rounds={avg_rounds:.1f} time={qt:.0f}s")
        results.append({'name': desc, 'ppl': ppl, 'delta': delta, 'eff': avg_eff, 'rounds': avg_rounds, 'time': qt})
        del model
        torch.cuda.empty_cache()

    # Summary
    results.sort(key=lambda x: x['ppl'])
    print(f"\n{'=' * 70}")
    print(f"Summary (FP baseline: {fp_ppl:.2f})")
    print(f"{'=' * 70}")
    for i, r in enumerate(results):
        print(f"  {i+1}. {r['name']:<25} PPL={r['ppl']:.2f} delta=+{r['delta']:.2f} "
              f"eff={r['eff']:.3f} rounds={r['rounds']:.0f} time={r['time']:.0f}s")


if __name__ == "__main__":
    main()

"""u4s4v4 快速 PPL 测试 - 限制轮数"""

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
    print("u4s4v4 快速 PPL 测试")
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

    # 测试配置: 用较少轮数快速测试，然后外推
    configs = [
        # (rank, n_rounds, desc)
        (1, 50, "rank=1 50轮 eff≈0.52"),
        (1, 100, "rank=1 100轮 eff≈1.04"),
        (1, 200, "rank=1 200轮 eff≈2.08"),
        (4, 25, "rank=4 25轮 eff≈0.78"),
        (4, 50, "rank=4 50轮 eff≈1.56"),
        (8, 12, "rank=8 12轮 eff≈0.75"),
        (8, 25, "rank=8 25轮 eff≈1.56"),
    ]

    results = []
    for rank, n_rounds, desc in configs:
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
                    rank=rank, u_bits=4, s_bits=4, v_bits=4,
                    n_rounds=n_rounds
                )
                mod.weight.data = W_q.to(mod.weight.dtype)
                layer_infos.append(info)
        qt = time.time() - t0
        ppl = evaluate_ppl(model, tokenizer)

        avg_eff = np.mean([info['effective_bits'] for info in layer_infos])
        delta = ppl - fp_ppl

        print(f"  PPL={ppl:.2f} delta=+{delta:.2f} eff={avg_eff:.3f} time={qt:.0f}s")
        results.append({'name': desc, 'ppl': ppl, 'delta': delta, 'eff': avg_eff, 'time': qt, 'rank': rank, 'n_rounds': n_rounds})
        del model
        torch.cuda.empty_cache()

    # Summary
    results.sort(key=lambda x: x['ppl'])
    print(f"\n{'=' * 70}")
    print(f"Summary (FP baseline: {fp_ppl:.2f})")
    print(f"{'=' * 70}")
    for i, r in enumerate(results):
        print(f"  {i+1}. {r['name']:<30} PPL={r['ppl']:.2f} delta=+{r['delta']:.2f} eff={r['eff']:.3f} time={r['time']:.0f}s")

    # 外推到 4.0 bit
    print(f"\n{'=' * 70}")
    print("外推到 eff=4.0 的 PPL 估计")
    print(f"{'=' * 70}")
    for rank in [1, 4, 8]:
        rank_results = [r for r in results if r['rank'] == rank]
        if len(rank_results) >= 2:
            # 线性外推: PPL 随 eff 降低
            x = [r['eff'] for r in rank_results]
            y = [r['ppl'] for r in rank_results]
            # 简单线性拟合
            coeffs = np.polyfit(x, y, 1)
            ppl_4_0 = np.polyval(coeffs, 4.0)
            print(f"  rank={rank}: 拟合 PPL(4.0) ≈ {ppl_4_0:.2f} (delta=+{ppl_4_0-fp_ppl:.2f})")


if __name__ == "__main__":
    main()

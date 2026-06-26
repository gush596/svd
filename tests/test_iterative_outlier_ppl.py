"""迭代异常值SVD + 3bit残差 PPL测试"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import time
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from quantization.iterative_outlier_svd import iterative_outlier_svd
from quantization.core import quantize_mse


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
        except Exception as ex:
            print(f"    [warn] {i}: {ex}")
    return float('inf') if nt == 0 else torch.exp(torch.tensor(sum(nlls) / nt)).item()


def main():
    model_id = "facebook/opt-125m"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    print("=" * 80)
    print("  迭代异常值SVD + 3bit残差 PPL测试")
    print("=" * 80)

    # FP baseline
    print("\nFP baseline...")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, device_map="cpu", low_cpu_mem_usage=True, trust_remote_code=True)
    model.eval()
    fp_ppl = evaluate_ppl(model, tokenizer)
    print(f"FP PPL: {fp_ppl:.2f}")
    del model

    # Direct 4-bit baseline
    print("\nDirect 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, device_map="cpu", low_cpu_mem_usage=True, trust_remote_code=True)
    model.eval()
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and 'lm_head' not in name:
            W = mod.weight.data.float().numpy()
            mod.weight.data = torch.from_numpy(quantize_mse(W, 4, 128)).to(mod.weight.dtype)
    d4_ppl = evaluate_ppl(model, tokenizer)
    print(f"Direct 4-bit PPL: {d4_ppl:.2f} (+{d4_ppl - fp_ppl:.2f})")
    del model

    # Direct 3-bit baseline
    print("\nDirect 3-bit...")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, device_map="cpu", low_cpu_mem_usage=True, trust_remote_code=True)
    model.eval()
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and 'lm_head' not in name:
            W = mod.weight.data.float().numpy()
            mod.weight.data = torch.from_numpy(quantize_mse(W, 3, 128)).to(mod.weight.dtype)
    d3_ppl = evaluate_ppl(model, tokenizer)
    print(f"Direct 3-bit PPL: {d3_ppl:.2f} (+{d3_ppl - fp_ppl:.2f})")
    del model

    # 测试配置: (outlier_ratio, max_svd_eff, rank, u_bits, v_bits, desc)
    configs = [
        (0.10, 0.50, 4, 3, 3, "ratio=0.10 svd_eff=0.50 rank=4 u3v3"),
        (0.10, 1.00, 4, 3, 3, "ratio=0.10 svd_eff=1.00 rank=4 u3v3"),
        (0.15, 1.00, 4, 3, 3, "ratio=0.15 svd_eff=1.00 rank=4 u3v3"),
        (0.10, 1.00, 8, 3, 3, "ratio=0.10 svd_eff=1.00 rank=8 u3v3"),
    ]

    results = []
    for ratio, svd_eff, rank, u, v, desc in configs:
        print(f"\n--- {desc} ---")
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, device_map="cpu", low_cpu_mem_usage=True, trust_remote_code=True)
        model.eval()

        t0 = time.time()
        layer_infos = []
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Linear) and 'lm_head' not in name:
                W = mod.weight.data.float().numpy()
                W_q, info = iterative_outlier_svd(
                    W, outlier_ratio=ratio, max_svd_eff=svd_eff,
                    residual_bits=3, rank=rank, u_bits=u, v_bits=v,
                    s_bits=16, group_size=128)
                mod.weight.data = torch.from_numpy(W_q).to(mod.weight.dtype)
                layer_infos.append(info)
        qt = time.time() - t0

        ppl = evaluate_ppl(model, tokenizer)
        delta = ppl - fp_ppl
        avg_eff = np.mean([i['total_eff_raw'] for i in layer_infos])
        avg_rounds = np.mean([i['rounds'] for i in layer_infos])

        beat4 = "✅" if ppl < d4_ppl else "❌"
        beat3 = "✅" if ppl < d3_ppl else "❌"
        print(f"  {beat4}vs4b {beat3}vs3b  PPL={ppl:.2f} +{delta:.2f} eff={avg_eff:.3f} rounds={avg_rounds:.0f} time={qt:.0f}s")

        results.append({'desc': desc, 'ppl': ppl, 'delta': delta, 'eff': avg_eff, 'rounds': avg_rounds, 'time': qt})
        del model

    # Summary
    results.sort(key=lambda x: x['ppl'])
    print(f"\n{'=' * 80}")
    print(f"  Summary  |  FP={fp_ppl:.2f}  4b={d4_ppl:.2f}(+{d4_ppl-fp_ppl:.2f})  3b={d3_ppl:.2f}(+{d3_ppl-fp_ppl:.2f})")
    print(f"{'=' * 80}")
    for i, r in enumerate(results):
        beat4 = "✅" if r['ppl'] < d4_ppl else "  "
        beat3 = "✅" if r['ppl'] < d3_ppl else "  "
        print(f"  {beat4}vs4b {beat3}vs3b  {i+1}. {r['desc']:<45} PPL={r['ppl']:.2f} +{r['delta']:.2f} eff={r['eff']:.3f} time={r['time']:.0f}s")


if __name__ == "__main__":
    main()

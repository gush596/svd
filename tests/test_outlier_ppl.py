"""异常值 SVD + 残差量化 PPL 测试"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import time
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from quantization.outlier_svd import outlier_svd_quantize
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
    print("  异常值 SVD + 残差量化 PPL 测试")
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

    # 测试配置
    configs = [
        # (outlier_ratio, svd_eff_bits, residual_bits, desc)
        (0.15, 1.0, 4, "outlier=15% svd_eff=1.0 res=4b"),
        (0.15, 1.0, 3, "outlier=15% svd_eff=1.0 res=3b"),
        (0.15, 0.5, 4, "outlier=15% svd_eff=0.5 res=4b"),
        (0.10, 1.0, 4, "outlier=10% svd_eff=1.0 res=4b"),
        (0.20, 1.0, 4, "outlier=20% svd_eff=1.0 res=4b"),
        (0.25, 1.0, 4, "outlier=25% svd_eff=1.0 res=4b"),
        (0.15, 2.0, 3, "outlier=15% svd_eff=2.0 res=3b"),
    ]

    results = []
    for outlier_r, svd_eff, res_bits, desc in configs:
        print(f"\n--- {desc} ---")
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, device_map="cpu", low_cpu_mem_usage=True, trust_remote_code=True)
        model.eval()

        t0 = time.time()
        layer_infos = []
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Linear) and 'lm_head' not in name:
                W = mod.weight.data.float().numpy()
                W_q, info = outlier_svd_quantize(
                    W, outlier_ratio=outlier_r, svd_eff_bits=svd_eff,
                    residual_bits=res_bits, svd_rank=4, svd_u_bits=3, svd_v_bits=3,
                    svd_s_bits=16, group_size=128,
                )
                mod.weight.data = torch.from_numpy(W_q).to(mod.weight.dtype)
                layer_infos.append(info)
        qt = time.time() - t0

        ppl = evaluate_ppl(model, tokenizer)
        delta = ppl - fp_ppl
        avg_eff_raw = np.mean([i['total_eff_raw'] for i in layer_infos])
        avg_eff_full = np.mean([i['total_eff_full'] for i in layer_infos])

        r = {'desc': desc, 'ppl': ppl, 'delta': delta, 'eff_raw': avg_eff_raw, 'eff_full': avg_eff_full, 'time': qt}
        results.append(r)

        beat = "✅" if ppl < d4_ppl else "❌"
        print(f"  {beat} PPL={ppl:.2f} delta=+{delta:.2f} eff_raw={avg_eff_raw:.3f} eff_full={avg_eff_full:.3f} time={qt:.0f}s")

        del model

    # Summary
    results.sort(key=lambda x: x['ppl'])
    print(f"\n{'=' * 80}")
    print(f"  Summary  |  FP={fp_ppl:.2f}  Direct4b={d4_ppl:.2f} (+{d4_ppl-fp_ppl:.2f})")
    print(f"{'=' * 80}")
    for i, r in enumerate(results):
        beat = "✅" if r['ppl'] < d4_ppl else "  "
        print(f"  {beat} {i+1}. {r['desc']:<40} PPL={r['ppl']:.2f} +{r['delta']:.2f} eff_raw={r['eff_raw']:.3f}")


if __name__ == "__main__":
    main()

"""
迭代残差 SVD PPL 测试 - u4v4 / s不量化

测试在真实 OPT-125M 模型上的 PPL 表现。
自动检测 GPU，有 CUDA 时使用 GPU 加速。

用法:
    python tests/test_ppl_u4v4.py              # 完整测试
    python tests/test_ppl_u4v4.py --quick      # 快速测试
    python tests/test_ppl_u4v4.py --rank 32    # 只测指定 rank
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import argparse
import time
import json
import numpy as np

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from quantization.iterative_svd import iterative_residual_svd, compute_max_rounds, _get_device


def get_device():
    """自动选择设备"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


@torch.no_grad()
def evaluate_ppl(model, tokenizer, n_samples=5, seqlen=512):
    """评估模型 PPL"""
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
            inp = ids[:, s:e].to(dev)
            o = model(inp, labels=inp)
            nlls.append(o.loss.item() * (e - s))
            nt += (e - s)
        except Exception as ex:
            print(f"    [warn] sample {i} failed: {ex}")
    return float('inf') if nt == 0 else torch.exp(torch.tensor(sum(nlls) / nt)).item()


def load_model(model_id, device):
    """加载模型到指定设备"""
    if device.type == 'cuda':
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto",
            low_cpu_mem_usage=True, trust_remote_code=True
        )
    else:
        # CPU: 用 float32（CPU 不支持 float16 的 loss 计算）
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float32, device_map="cpu",
            low_cpu_mem_usage=True, trust_remote_code=True
        )
    model.eval()
    return model


def apply_iterative_svd(model, rank, u_bits, s_bits, v_bits, group_size=128, max_eff=4.0, device=None):
    """对模型所有 Linear 层应用迭代残差 SVD 量化"""
    t0 = time.time()
    layer_infos = []
    n_layers = 0
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and 'lm_head' not in name:
            W = mod.weight.data.clone()
            # 如果在 GPU 上，先转到 CPU 做量化（numpy 操作），再转回
            if W.is_cuda:
                W_np = W.float().cpu().numpy()
            else:
                W_np = W.float().numpy()
            
            W_q, info = iterative_residual_svd(
                W_np, group_size, max_eff_bits=max_eff,
                rank=rank, u_bits=u_bits, s_bits=s_bits, v_bits=v_bits
            )
            
            W_q_torch = torch.from_numpy(W_q).to(mod.weight.device, dtype=mod.weight.dtype)
            mod.weight.data = W_q_torch
            layer_infos.append(info)
            n_layers += 1
    
    elapsed = time.time() - t0
    avg_eff = np.mean([info['effective_bits'] for info in layer_infos])
    avg_eff_full = np.mean([info['effective_bits_full'] for info in layer_infos])
    avg_rounds = np.mean([info['rounds'] for info in layer_infos])
    
    return {
        'n_layers': n_layers,
        'avg_eff_raw': float(avg_eff),
        'avg_eff_full': float(avg_eff_full),
        'avg_rounds': float(avg_rounds),
        'time': elapsed,
        'layer_infos': layer_infos,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="facebook/opt-125m")
    parser.add_argument("--n_samples", type=int, default=5)
    parser.add_argument("--seqlen", type=int, default=512)
    parser.add_argument("--rank", type=int, nargs="+", default=None, help="测试的 rank 列表")
    parser.add_argument("--quick", action="store_true", help="快速测试（少量配置）")
    parser.add_argument("--output", default="ppl_results.json")
    args = parser.parse_args()
    
    device = get_device()
    if device.type == 'cuda':
        dev_name = f"GPU ({torch.cuda.get_device_name(0)})"
    else:
        dev_name = "CPU"
    
    print("=" * 80)
    print("  迭代残差 SVD PPL 测试")
    print("=" * 80)
    print(f"  模型: {args.model}")
    print(f"  设备: {dev_name}")
    print(f"  数据: wikitext-2, {args.n_samples} samples, seqlen={args.seqlen}")
    print()
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    # FP baseline
    print("加载 FP baseline...")
    model = load_model(args.model, device)
    fp_ppl = evaluate_ppl(model, tokenizer, args.n_samples, args.seqlen)
    print(f"FP Baseline PPL: {fp_ppl:.2f}")
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # 直接 4-bit baseline
    print("\n加载 direct 4-bit baseline...")
    from quantization.core import quantize_mse
    model = load_model(args.model, device)
    t0 = time.time()
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and 'lm_head' not in name:
            W = mod.weight.data.clone()
            if W.is_cuda:
                W_np = W.float().cpu().numpy()
            else:
                W_np = W.float().numpy()
            W_q = quantize_mse(W_np, n_bits=4, group_size=128)
            mod.weight.data = torch.from_numpy(W_q).to(mod.weight.device, dtype=mod.weight.dtype)
    direct_time = time.time() - t0
    direct_ppl = evaluate_ppl(model, tokenizer, args.n_samples, args.seqlen)
    print(f"Direct 4-bit PPL: {direct_ppl:.2f} (delta=+{direct_ppl - fp_ppl:.2f}, {direct_time:.0f}s)")
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # 测试配置
    if args.quick:
        configs = [
            # (rank, u_bits, s_bits, v_bits, desc)
            (32, 4, None, 4, "rank=32 u4 s-fp32 v4"),
            (32, 3, None, 3, "rank=32 u3 s-fp32 v3"),
        ]
    else:
        configs = [
            # 不同 rank, u4v4, s 不量化
            (8,  4, None, 4, "rank=8  u4 s-fp32 v4"),
            (16, 4, None, 4, "rank=16 u4 s-fp32 v4"),
            (24, 4, None, 4, "rank=24 u4 s-fp32 v4"),
            (32, 4, None, 4, "rank=32 u4 s-fp32 v4"),
            (48, 4, None, 4, "rank=48 u4 s-fp32 v4"),
            (64, 4, None, 4, "rank=64 u4 s-fp32 v4"),
            # 不同 rank, u3v3, s 不量化
            (16, 3, None, 3, "rank=16 u3 s-fp32 v3"),
            (24, 3, None, 3, "rank=24 u3 s-fp32 v3"),
            (32, 3, None, 3, "rank=32 u3 s-fp32 v3"),
            (48, 3, None, 3, "rank=48 u3 s-fp32 v3"),
            (64, 3, None, 3, "rank=64 u3 s-fp32 v3"),
            # u4v3
            (32, 4, None, 3, "rank=32 u4 s-fp32 v3"),
            (48, 4, None, 3, "rank=48 u4 s-fp32 v3"),
            (64, 4, None, 3, "rank=64 u4 s-fp32 v3"),
            (96, 4, None, 3, "rank=96 u4 s-fp32 v3"),
        ]
    
    # 过滤 rank
    if args.rank:
        configs = [c for c in configs if c[0] in args.rank]
    
    # 运行测试
    results = []
    for rank, ub, sb, vb, desc in configs:
        max_r = compute_max_rounds(768, 768, 4.0, rank, ub, sb, vb, 128, use_full_eff=True)
        if max_r < 1:
            print(f"\n  {desc}: 跳过（预算不足）")
            continue
        
        print(f"\n--- {desc} (max_rounds={max_r}) ---")
        model = load_model(args.model, device)
        
        info = apply_iterative_svd(model, rank, ub, sb, vb, 128, 4.0, device)
        ppl = evaluate_ppl(model, tokenizer, args.n_samples, args.seqlen)
        delta = ppl - fp_ppl
        
        result = {
            'name': desc,
            'rank': rank,
            'u_bits': ub,
            's_bits': sb if sb is not None else 'fp32',
            'v_bits': vb,
            'ppl': ppl,
            'delta': delta,
            'avg_eff_raw': info['avg_eff_raw'],
            'avg_eff_full': info['avg_eff_full'],
            'avg_rounds': info['avg_rounds'],
            'time': info['time'],
            'n_layers': info['n_layers'],
        }
        results.append(result)
        
        status = "✅" if ppl < direct_ppl else "❌"
        print(f"  {status} PPL={ppl:.2f} delta=+{delta:.2f} "
              f"eff_raw={info['avg_eff_raw']:.3f} eff_full={info['avg_eff_full']:.3f} "
              f"rounds={info['avg_rounds']:.0f} time={info['time']:.0f}s")
        
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Summary
    results.sort(key=lambda x: x['ppl'])
    print(f"\n{'=' * 80}")
    print(f"  Summary")
    print(f"  FP baseline: {fp_ppl:.2f}")
    print(f"  Direct 4-bit: {direct_ppl:.2f} (delta=+{direct_ppl - fp_ppl:.2f})")
    print(f"{'=' * 80}")
    
    for i, r in enumerate(results):
        beat = "✅" if r['ppl'] < direct_ppl else "  "
        print(f"  {beat} {i+1:2d}. {r['name']:<30} PPL={r['ppl']:.2f} "
              f"delta=+{r['delta']:.2f} eff_full={r['avg_eff_full']:.3f} "
              f"rounds={r['avg_rounds']:.0f} time={r['time']:.0f}s")
    
    # Save
    output = {
        'model': args.model,
        'device': dev_name,
        'fp_baseline': fp_ppl,
        'direct_4bit': {'ppl': direct_ppl, 'delta': direct_ppl - fp_ppl, 'time': direct_time},
        'results': results,
    }
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到 {args.output}")


if __name__ == "__main__":
    main()

"""
迭代残差 SVD PPL 测试 - eff_raw ≤ 4.0 + checkpoint 缓存

- eff_raw ≤ 4.0 约束（允许更多轮数）
- checkpoint 机制：量化结果保存到磁盘，下次直接加载
- 自动 GPU 检测
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import argparse
import time
import json
import math
import numpy as np

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from quantization.iterative_svd import compute_max_rounds, _get_device, _svd_decompose
from quantization.core import quantize_mse


CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


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
            inp = ids[:, s:e].to(dev)
            o = model(inp, labels=inp)
            nlls.append(o.loss.item() * (e - s))
            nt += (e - s)
        except Exception as ex:
            print(f"    [warn] sample {i} failed: {ex}")
    return float('inf') if nt == 0 else torch.exp(torch.tensor(sum(nlls) / nt)).item()


def load_model(model_id, device):
    if device.type == 'cuda':
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto",
            low_cpu_mem_usage=True, trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float32, device_map="cpu",
            low_cpu_mem_usage=True, trust_remote_code=True
        )
    model.eval()
    return model


# ─── checkpoint 核心 ───────────────────────────────────────────────────

def checkpoint_path(config_name):
    """checkpoint 目录"""
    return os.path.join(CHECKPOINT_DIR, config_name)


def save_checkpoint(config_name, layer_name, round_idx, W_approx_np, residual_np, info):
    """保存单层单轮 checkpoint"""
    d = os.path.join(checkpoint_path(config_name), layer_name)
    os.makedirs(d, exist_ok=True)
    np.savez_compressed(
        os.path.join(d, f"round_{round_idx:04d}.npz"),
        W_approx=W_approx_np, residual=residual_np,
        **{k: v for k, v in info.items() if not isinstance(v, (list, dict))}
    )


def load_checkpoint(config_name, layer_name, round_idx):
    """加载单层单轮 checkpoint，返回 (W_approx, residual, info) 或 None"""
    f = os.path.join(checkpoint_path(config_name), layer_name, f"round_{round_idx:04d}.npz")
    if not os.path.exists(f):
        return None
    data = np.load(f)
    info = {k: data[k].item() if data[k].ndim == 0 else data[k] for k in data.files if k not in ('W_approx', 'residual')}
    return data['W_approx'], data['residual'], info


def get_max_saved_round(config_name, layer_name):
    """查询已保存的最大轮数"""
    d = os.path.join(checkpoint_path(config_name), layer_name)
    if not os.path.exists(d):
        return 0
    files = [f for f in os.listdir(d) if f.startswith("round_") and f.endswith(".npz")]
    if not files:
        return 0
    return max(int(f.split("_")[1].split(".")[0]) for f in files)


# ─── 迭代 SVD + checkpoint ─────────────────────────────────────────────

def iterative_svd_with_checkpoint(
    W_np, config_name, layer_name, rank, u_bits, s_bits, v_bits,
    group_size, max_rounds, svd_device=None,
):
    """
    迭代残差 SVD，支持断点续跑。
    返回 (W_approx, info_dict)
    """
    out_dim, in_dim = W_np.shape
    W_f = W_np.astype(np.float32)

    s_use_fp16 = (s_bits == 16)
    s_quant = s_bits is not None and s_bits < 16

    # 找到已有的最大轮数
    start_round = get_max_saved_round(config_name, layer_name)

    if start_round >= max_rounds:
        # 直接加载最后一轮
        W_approx, residual, _ = load_checkpoint(config_name, layer_name, max_rounds - 1)
        return W_approx, {'rounds': max_rounds, 'from_cache': True}

    if start_round > 0:
        W_approx, residual, _ = load_checkpoint(config_name, layer_name, start_round - 1)
        print(f"      从 round {start_round} 续跑", end="", flush=True)
    else:
        W_approx = np.zeros_like(W_f)
        residual = W_f.copy()

    for i in range(start_round, max_rounds):
        U, S, Vt = _svd_decompose(residual, svd_device)
        actual_rank = min(rank, len(S))

        U_k = U[:, :actual_rank]
        S_k = S[:actual_rank]
        V_k = Vt[:actual_rank, :]

        gs = min(group_size, max(8, actual_rank))
        U_q = quantize_mse(U_k, n_bits=u_bits, group_size=gs)
        if s_quant:
            S_q = quantize_mse(S_k.reshape(1, -1), n_bits=s_bits, group_size=gs).reshape(-1)
        elif s_use_fp16:
            S_q = S_k.astype(np.float16).astype(np.float32)
        else:
            S_q = S_k
        V_q = quantize_mse(V_k, n_bits=v_bits, group_size=gs)

        component = U_q @ np.diag(S_q) @ V_q
        W_approx = W_approx + component
        residual = residual - component

        save_checkpoint(config_name, layer_name, i, W_approx, residual,
                        {'rank': actual_rank, 'u_bits': u_bits, 'v_bits': v_bits})

    return W_approx, {'rounds': max_rounds, 'from_cache': start_round > 0}


# ─── eff 计算 ──────────────────────────────────────────────────────────

def compute_eff_raw(out_dim, in_dim, rank, u_bits, s_bits, v_bits, n_rounds, group_size=128):
    s_bits_eff = s_bits if s_bits is not None else 32
    round_bits = rank * (out_dim * u_bits + in_dim * v_bits) + rank * s_bits_eff
    return n_rounds * round_bits / (out_dim * in_dim)


def compute_eff_full(out_dim, in_dim, rank, u_bits, s_bits, v_bits, n_rounds, group_size=128):
    s_bits_eff = s_bits if s_bits is not None else 32
    round_raw = rank * (out_dim * u_bits + in_dim * v_bits) + rank * s_bits_eff
    gs_u = min(group_size, max(8, rank))
    gs_v = min(group_size, max(8, rank))
    n_groups_u = math.ceil(out_dim * rank / gs_u)
    n_groups_v = math.ceil(rank * in_dim / gs_v)
    n_groups_s = 1 if (s_bits is not None and s_bits < 16) else 0
    scale_bits = (n_groups_u + n_groups_s + n_groups_v) * 32
    return n_rounds * (round_raw + scale_bits) / (out_dim * in_dim)


# ─── 主函数 ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="facebook/opt-125m")
    parser.add_argument("--n_samples", type=int, default=5)
    parser.add_argument("--seqlen", type=int, default=512)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--rank", type=int, nargs="+", default=None)
    parser.add_argument("--clear_cache", action="store_true", help="清除 checkpoint 缓存")
    parser.add_argument("--output", default="ppl_results.json")
    args = parser.parse_args()

    device = get_device()
    dev_name = f"GPU ({torch.cuda.get_device_name(0)})" if device.type == 'cuda' else "CPU"

    print("=" * 80)
    print("  迭代残差 SVD PPL 测试 (eff_raw ≤ 4.0, checkpoint 缓存)")
    print("=" * 80)
    print(f"  模型: {args.model}  |  设备: {dev_name}")
    print(f"  数据: wikitext-2, {args.n_samples} samples, seqlen={args.seqlen}")
    print(f"  checkpoint 目录: {CHECKPOINT_DIR}")
    print()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # ── FP baseline ──
    print("加载 FP baseline...")
    model = load_model(args.model, device)
    fp_ppl = evaluate_ppl(model, tokenizer, args.n_samples, args.seqlen)
    print(f"FP Baseline PPL: {fp_ppl:.2f}")
    del model; torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── Direct 4-bit baseline ──
    print("\nDirect 4-bit baseline...")
    model = load_model(args.model, device)
    t0 = time.time()
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and 'lm_head' not in name:
            W = mod.weight.data.clone()
            W_np = W.float().cpu().numpy() if W.is_cuda else W.float().numpy()
            mod.weight.data = torch.from_numpy(quantize_mse(W_np, 4, 128)).to(mod.weight.device, dtype=mod.weight.dtype)
    direct_time = time.time() - t0
    direct_ppl = evaluate_ppl(model, tokenizer, args.n_samples, args.seqlen)
    print(f"Direct 4-bit PPL: {direct_ppl:.2f} (delta=+{direct_ppl - fp_ppl:.2f}, {direct_time:.0f}s)")
    del model; torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── 测试配置 ──
    # 获取各层信息
    model_tmp = load_model(args.model, device)
    layers = []
    for name, mod in model_tmp.named_modules():
        if isinstance(mod, nn.Linear) and 'lm_head' not in name:
            layers.append((name, mod.weight.shape[0], mod.weight.shape[1]))
    del model_tmp; torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print(f"\n共 {len(layers)} 个 Linear 层")
    for lname, out_d, in_d in layers:
        print(f"  {lname:<30} {out_d}×{in_d}")

    # 配置列表
    if args.quick:
        configs = [
            (16, 4, 16, 4, "rank16-u4-sfp16-v4"),
            (32, 4, 16, 4, "rank32-u4-sfp16-v4"),
        ]
    else:
        configs = [
            (16, 4, 16, 4, "rank16-u4-sfp16-v4"),
            (32, 4, 16, 4, "rank32-u4-sfp16-v4"),
            (16, 3, 16, 3, "rank16-u3-sfp16-v3"),
            (32, 3, 16, 3, "rank32-u3-sfp16-v3"),
            (16, 4, None, 4, "rank16-u4-sfp32-v4"),
        ]

    if args.rank:
        configs = [c for c in configs if c[0] in args.rank]

    if args.clear_cache:
        import shutil
        for _, _, _, _, cname in configs:
            p = checkpoint_path(cname)
            if os.path.exists(p):
                shutil.rmtree(p)
                print(f"  已清除缓存: {cname}")

    # ── 逐配置测试 ──
    results = []
    for rank, ub, sb, vb, cfg_name in configs:
        # 计算 eff_raw ≤ 4.0 对应的最大轮数（取各层最小值）
        min_rounds = float('inf')
        for lname, out_d, in_d in layers:
            s_eff = sb if sb is not None else 32
            round_bits = rank * (out_d * ub + in_d * vb) + rank * s_eff
            max_r = int(4.0 * out_d * in_d / round_bits)
            min_rounds = min(min_rounds, max_r)

        if min_rounds < 1:
            print(f"\n  {cfg_name}: 跳过（预算不足）")
            continue

        # 计算各层 eff
        eff_raws, eff_fulls = [], []
        for lname, out_d, in_d in layers:
            eff_raws.append(compute_eff_raw(out_d, in_d, rank, ub, sb, vb, min_rounds, 128))
            eff_fulls.append(compute_eff_full(out_d, in_d, rank, ub, sb, vb, min_rounds, 128))
        avg_eff_raw = np.mean(eff_raws)
        avg_eff_full = np.mean(eff_fulls)

        print(f"\n{'─' * 80}")
        print(f"  {cfg_name}  |  rounds={min_rounds}  |  eff_raw={avg_eff_raw:.4f}  eff_full={avg_eff_full:.4f}")
        print(f"{'─' * 80}")

        # 检查缓存状态
        cached_layers = 0
        for lname, _, _ in layers:
            if get_max_saved_round(cfg_name, lname) >= min_rounds:
                cached_layers += 1
        if cached_layers > 0:
            print(f"  缓存: {cached_layers}/{len(layers)} 层已就绪")

        # 量化
        print("  量化中...", end="", flush=True)
        t0 = time.time()
        model = load_model(args.model, device)
        svd_device = _get_device()

        for lname, mod in model.named_modules():
            if isinstance(mod, nn.Linear) and 'lm_head' not in lname:
                W = mod.weight.data.clone()
                W_np = W.float().cpu().numpy() if W.is_cuda else W.float().numpy()
                W_q, info = iterative_svd_with_checkpoint(
                    W_np, cfg_name, lname, rank, ub, sb, vb, 128, min_rounds, svd_device
                )
                mod.weight.data = torch.from_numpy(W_q).to(mod.weight.device, dtype=mod.weight.dtype)

        quant_time = time.time() - t0
        print(f" 完成 ({quant_time:.0f}s)")

        # PPL
        print("  评估 PPL...", end="", flush=True)
        ppl = evaluate_ppl(model, tokenizer, args.n_samples, args.seqlen)
        delta = ppl - fp_ppl
        print(f" {ppl:.2f}")

        result = {
            'name': cfg_name, 'rank': rank, 'u_bits': ub,
            's_bits': sb if sb is not None else 'fp32', 'v_bits': vb,
            'ppl': ppl, 'delta': delta,
            'avg_eff_raw': float(avg_eff_raw), 'avg_eff_full': float(avg_eff_full),
            'rounds': min_rounds, 'time': quant_time,
        }
        results.append(result)

        beat = "✅" if ppl < direct_ppl else "❌"
        print(f"  {beat} PPL={ppl:.2f} delta=+{delta:.2f} (direct={direct_ppl:.2f})")

        del model; torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── Summary ──
    results.sort(key=lambda x: x['ppl'])
    print(f"\n{'=' * 80}")
    print(f"  Summary")
    print(f"  FP baseline: {fp_ppl:.2f}  |  Direct 4-bit: {direct_ppl:.2f} (+{direct_ppl - fp_ppl:.2f})")
    print(f"{'=' * 80}")
    for i, r in enumerate(results):
        beat = "✅" if r['ppl'] < direct_ppl else "  "
        print(f"  {beat} {i+1:2d}. {r['name']:<30} PPL={r['ppl']:.2f} delta=+{r['delta']:.2f} "
              f"eff_raw={r['avg_eff_raw']:.3f} eff_full={r['avg_eff_full']:.3f} rounds={r['rounds']}")

    with open(args.output, 'w') as f:
        json.dump({'fp_baseline': fp_ppl, 'direct_4bit': direct_ppl, 'results': results}, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到 {args.output}")


if __name__ == "__main__":
    main()

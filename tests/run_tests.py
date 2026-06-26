#!/usr/bin/env python3
"""
统一测试入口

支持在一个函数里可选地测试不同量化方案。

用法:
    # MSE 测试（合成数据，不需要 GPU/模型下载）
    python tests/run_tests.py --scheme mse --method all
    python tests/run_tests.py --scheme mse --method iterative_svd
    python tests/run_tests.py --scheme mse --method outlier_svd
    python tests/run_tests.py --scheme mse --method iterative_outlier

    # PPL 测试（需要模型下载）
    python tests/run_tests.py --scheme ppl --method direct
    python tests/run_tests.py --scheme ppl --method iterative_svd
    python tests/run_tests.py --scheme ppl --method outlier_svd
    python tests/run_tests.py --scheme ppl --method iterative_outlier
    python tests/run_tests.py --scheme ppl --method all

    # 幂迭代对比测试
    python tests/run_tests.py --scheme power_iter

    # 快速测试（少量配置）
    python tests/run_tests.py --scheme mse --method all --quick
"""

import sys
import os
import io
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import argparse
import time
import json
import numpy as np

from tests.common import (
    generate_weight_matrix, evaluate_ppl, load_model_and_tokenizer,
    apply_quant_to_model, compute_eff_raw, compute_eff_full,
    print_section, print_table,
)
from quantization.core import quantize_mse


# ═══════════════════════════════════════════════════════════════════════
#  MSE 测试方案（合成数据，快速验证）
# ═══════════════════════════════════════════════════════════════════════

def test_mse_direct(W, W_f, direct_mse, asymmetric=False):
    """直接 4-bit 量化基线"""
    from quantization.core import quantize_mse
    tag = "非对称" if asymmetric else "对称"
    print_section(f"直接量化基线 ({tag})")
    W_q = quantize_mse(W, 4, 128, asymmetric=asymmetric)
    mse = float(np.mean((W_f - W_q) ** 2))
    print(f"  Direct 4-bit ({tag}) MSE: {mse:.6f} (ratio=1.000)")
    return [{'method': f'direct_4bit_{tag}', 'mse': mse, 'ratio': 1.0, 'eff': 4.0}]


def test_mse_important(W, W_f, direct_mse, quick=False, asymmetric=False):
    """重要值保护"""
    from quantization.important import important_protection
    print_section("重要值保护")

    if quick:
        configs = [(0.15, 5, "r=0.15 b=5")]
    else:
        configs = [
            (0.05, 5, "r=0.05 b=5"), (0.10, 5, "r=0.10 b=5"),
            (0.15, 5, "r=0.15 b=5"), (0.10, 6, "r=0.10 b=6"),
            (0.20, 6, "r=0.20 b=6"), (0.25, 6, "r=0.25 b=6"),
        ]

    results = []
    for ratio, bits, desc in configs:
        W_q, info = important_protection(W, 4, 128, ratio, bits)
        mse = float(np.mean((W_f - W_q.astype(np.float32)) ** 2))
        eff = info['effective_bits']
        status = "✅" if eff <= 4.05 else "⚠️"
        print(f"  {status} {desc:<20} eff={eff:.3f} mse={mse:.6f} ratio={mse/direct_mse:.3f}")
        results.append({'method': f'important_{desc}', 'mse': mse, 'ratio': mse/direct_mse, 'eff': eff})
    return results


def test_mse_svd_hybrid(W, W_f, direct_mse, quick=False, asymmetric=False):
    """SVD Hybrid"""
    from quantization.svd_hybrid import svd_hybrid
    print_section("SVD Hybrid")

    if quick:
        configs = [(0.80, 3, 4, "t=0.80 sb=3")]
    else:
        configs = [
            (0.70, 3, 4, "t=0.70 sb=3"), (0.80, 3, 4, "t=0.80 sb=3"),
            (0.85, 3, 4, "t=0.85 sb=3"), (0.90, 3, 4, "t=0.90 sb=3"),
        ]

    results = []
    for t, sb, rb, desc in configs:
        W_q, info = svd_hybrid(W, 4, 128, t, sb, rb)
        mse = float(np.mean((W_f - W_q.astype(np.float32)) ** 2))
        eff = info['effective_bits']
        status = "✅" if eff <= 4.05 else "⚠️"
        print(f"  {status} {desc:<20} rank={info['rank']:<4} eff={eff:.3f} "
              f"mse={mse:.6f} ratio={mse/direct_mse:.3f}")
        results.append({'method': f'svd_hybrid_{desc}', 'mse': mse, 'ratio': mse/direct_mse, 'eff': eff})
    return results


def test_mse_iterative_svd(W, W_f, direct_mse, quick=False, asymmetric=False):
    """迭代残差 SVD（残差舍弃模式）"""
    from quantization.iterative_svd import iterative_residual_svd
    print_section("迭代残差 SVD（残差舍弃）")

    if quick:
        configs = [
            (4, 3, None, 4, "rank=4 u3v3 s-fp32"),
            (8, 3, None, 4, "rank=8 u3v3 s-fp32"),
        ]
    else:
        configs = [
            (1, 3, None, 4, "rank=1 u3v3 s-fp32"),
            (4, 3, None, 4, "rank=4 u3v3 s-fp32"),
            (8, 3, None, 4, "rank=8 u3v3 s-fp32"),
            (4, 4, None, 4, "rank=4 u4v4 s-fp32"),
            (8, 4, None, 4, "rank=8 u4v4 s-fp32"),
            (16, 4, None, 4, "rank=16 u4v4 s-fp32"),
            (32, 4, None, 4, "rank=32 u4v4 s-fp32"),
            (64, 4, None, 4, "rank=64 u4v4 s-fp32"),
            (32, 3, None, 3, "rank=32 u3v3 s-fp32"),
            (48, 3, None, 3, "rank=48 u3v3 s-fp32"),
        ]

    results = []
    for rank, ub, sb, vb, desc in configs:
        t0 = time.time()
        W_q, info = iterative_residual_svd(
            W, 128, max_eff_bits=4.0, rank=rank,
            u_bits=ub, s_bits=sb, v_bits=vb, asymmetric=asymmetric,
        )
        elapsed = time.time() - t0
        mse = float(np.mean((W_f - W_q.astype(np.float32)) ** 2))
        eff_raw = info['effective_bits']
        eff_full = info.get('effective_bits_full', eff_raw)
        status = "✅" if eff_full <= 4.05 else "⚠️"
        beat = "✅" if mse < direct_mse else "  "
        print(f"  {status}{beat} {desc:<30} rounds={info['rounds']:<4} "
              f"eff_raw={eff_raw:.3f} eff_full={eff_full:.3f} "
              f"mse={mse:.6f} ratio={mse/direct_mse:.3f} ({elapsed:.1f}s)")
        results.append({
            'method': f'iter_svd_{desc}', 'mse': mse,
            'ratio': mse/direct_mse, 'eff_raw': eff_raw, 'eff_full': eff_full,
            'rounds': info['rounds'], 'time': elapsed,
        })
    return results


def test_mse_outlier_svd(W, W_f, direct_mse, quick=False, asymmetric=False):
    """异常值 SVD + 残差量化"""
    from quantization.outlier_svd import outlier_svd_quantize
    print_section("异常值 SVD + 残差量化")

    if quick:
        configs = [
            (0.15, 1.0, 3, "ratio=0.15 svd_eff=1.0 res=3b"),
            (0.15, 1.0, 4, "ratio=0.15 svd_eff=1.0 res=4b"),
        ]
    else:
        configs = [
            (0.05, 1.0, 3, "ratio=0.05 svd_eff=1.0 res=3b"),
            (0.10, 1.0, 3, "ratio=0.10 svd_eff=1.0 res=3b"),
            (0.15, 1.0, 3, "ratio=0.15 svd_eff=1.0 res=3b"),
            (0.20, 1.0, 3, "ratio=0.20 svd_eff=1.0 res=3b"),
            (0.15, 0.5, 3, "ratio=0.15 svd_eff=0.5 res=3b"),
            (0.15, 1.5, 3, "ratio=0.15 svd_eff=1.5 res=3b"),
            (0.15, 1.0, 4, "ratio=0.15 svd_eff=1.0 res=4b"),
            (0.15, 0.5, 4, "ratio=0.15 svd_eff=0.5 res=4b"),
        ]

    results = []
    for ratio, svd_eff, res_bits, desc in configs:
        t0 = time.time()
        W_q, info = outlier_svd_quantize(
            W, outlier_ratio=ratio, svd_eff_bits=svd_eff,
            residual_bits=res_bits, svd_rank=4, svd_u_bits=3, svd_v_bits=3,
            svd_s_bits=16, group_size=128, asymmetric=asymmetric,
        )
        elapsed = time.time() - t0
        mse = float(np.mean((W_f - W_q.astype(np.float32)) ** 2))
        eff_raw = info['total_eff_raw']
        eff_full = info['total_eff_full']
        status = "✅" if eff_full <= 4.05 else "⚠️"
        print(f"  {status} {desc:<35} eff_raw={eff_raw:.3f} eff_full={eff_full:.3f} "
              f"mse={mse:.6f} ratio={mse/direct_mse:.3f} ({elapsed:.1f}s)")
        results.append({
            'method': f'outlier_svd_{desc}', 'mse': mse,
            'ratio': mse/direct_mse, 'eff_raw': eff_raw, 'eff_full': eff_full,
            'time': elapsed,
        })
    return results


def test_mse_iterative_outlier(W, W_f, direct_mse, quick=False, asymmetric=False):
    """迭代异常值 SVD + 残差量化"""
    from quantization.iterative_outlier_svd import iterative_outlier_svd
    print_section("迭代异常值 SVD + 残差量化")

    if quick:
        configs = [
            (0.10, 1.0, 3, 4, 3, 3, "ratio=0.10 svd_eff=1.0 rank=4 u3v3 res=3b"),
        ]
    else:
        configs = [
            (0.10, 0.50, 3, 4, 3, 3, "ratio=0.10 svd_eff=0.50 rank=4 u3v3 res=3b"),
            (0.10, 1.00, 3, 4, 3, 3, "ratio=0.10 svd_eff=1.00 rank=4 u3v3 res=3b"),
            (0.15, 1.00, 3, 4, 3, 3, "ratio=0.15 svd_eff=1.00 rank=4 u3v3 res=3b"),
            (0.10, 1.00, 3, 8, 3, 3, "ratio=0.10 svd_eff=1.00 rank=8 u3v3 res=3b"),
            (0.10, 1.00, 4, 4, 3, 3, "ratio=0.10 svd_eff=1.00 rank=4 u3v3 res=4b"),
        ]

    results = []
    for ratio, svd_eff, res_bits, rank, ub, vb, desc in configs:
        t0 = time.time()
        W_q, info = iterative_outlier_svd(
            W, outlier_ratio=ratio, max_svd_eff=svd_eff,
            residual_bits=res_bits, rank=rank, u_bits=ub, v_bits=vb,
            s_bits=16, group_size=128, asymmetric=asymmetric,
        )
        elapsed = time.time() - t0
        mse = float(np.mean((W_f - W_q.astype(np.float32)) ** 2))
        eff_raw = info['total_eff_raw']
        print(f"  {desc:<45} eff={eff_raw:.3f} mse={mse:.6f} "
              f"ratio={mse/direct_mse:.3f} ({elapsed:.1f}s)")
        results.append({
            'method': f'iter_outlier_{desc}', 'mse': mse,
            'ratio': mse/direct_mse, 'eff_raw': eff_raw, 'time': elapsed,
        })
    return results


def test_mse_iter_outlier_tune(W, W_f, direct_mse, asymmetric=False):
    """迭代异常值 SVD 参数扫描（同时对比对称/非对称）"""
    from quantization.iterative_outlier_svd import iterative_outlier_svd
    print_section("迭代异常值 SVD 参数扫描")

    # 基线
    sym4_mse = float(np.mean((W_f - quantize_mse(W, 4, 128, asymmetric=False)) ** 2))
    asym4_mse = float(np.mean((W_f - quantize_mse(W, 4, 128, asymmetric=True)) ** 2))
    print(f"  基线: sym 4-bit={sym4_mse:.6f}  asym 4-bit={asym4_mse:.6f}")

    configs = []
    for ratio in [0.05, 0.10, 0.15, 0.20]:
        for rank in [4, 8, 16]:
            for ub, vb in [(3, 3)]:
                for svd_eff in [0.5, 1.0, 1.5, 2.0]:
                    desc = f"r={ratio} eff={svd_eff} rk={rank} u{ub}v{vb}"
                    configs.append((ratio, svd_eff, 3, rank, ub, vb, desc))

    print(f"  共 {len(configs)} 个配置 × 2 (对称/非对称)")
    print(f"\n  {'配置':<30} {'sym_MSE':<12} {'asym_MSE':<12} {'sym_ratio':<10} {'asym_ratio':<10} {'改善':<8}")
    print(f"  {'-' * 82}")

    results = []
    for ratio, svd_eff, res_bits, rank, ub, vb, desc in configs:
        # 对称
        W_q_sym, info_sym = iterative_outlier_svd(
            W, outlier_ratio=ratio, max_svd_eff=svd_eff,
            residual_bits=res_bits, rank=rank, u_bits=ub, v_bits=vb,
            s_bits=16, group_size=128, asymmetric=False)
        sym_mse = float(np.mean((W_f - W_q_sym.astype(np.float32)) ** 2))

        # 非对称
        W_q_asym, info_asym = iterative_outlier_svd(
            W, outlier_ratio=ratio, max_svd_eff=svd_eff,
            residual_bits=res_bits, rank=rank, u_bits=ub, v_bits=vb,
            s_bits=16, group_size=128, asymmetric=True)
        asym_mse = float(np.mean((W_f - W_q_asym.astype(np.float32)) ** 2))

        eff_raw = info_sym['total_eff_raw']
        sym_ratio = sym_mse / sym4_mse
        asym_ratio = asym_mse / sym4_mse
        improve = (1 - asym_mse / sym_mse) * 100

        status = "✅" if eff_raw <= 4.0 else "❌"
        print(f"  {status} {desc:<29} {sym_mse:<12.6f} {asym_mse:<12.6f} {sym_ratio:<10.3f} {asym_ratio:<10.3f} {improve:+.1f}%")

        results.append({'method': f'{desc}_sym', 'mse': sym_mse, 'ratio': sym_ratio, 'eff_raw': eff_raw})
        results.append({'method': f'{desc}_asym', 'mse': asym_mse, 'ratio': asym_ratio, 'eff_raw': eff_raw})

    # Top 10 (对称)
    sym_valid = [r for r in results if r['eff_raw'] <= 4.0 and r['method'].endswith('_sym')]
    sym_valid.sort(key=lambda x: x['mse'])
    print(f"\n  --- Top 10 对称 (eff_raw ≤ 4.0) ---")
    for i, r in enumerate(sym_valid[:10]):
        print(f"  {i+1:2d}. {r['method']:<45} eff={r['eff_raw']:.3f} mse={r['mse']:.6f} ratio={r['ratio']:.3f}")

    # Top 10 (非对称)
    asym_valid = [r for r in results if r['eff_raw'] <= 4.0 and r['method'].endswith('_asym')]
    asym_valid.sort(key=lambda x: x['mse'])
    print(f"\n  --- Top 10 非对称 (eff_raw ≤ 4.0) ---")
    for i, r in enumerate(asym_valid[:10]):
        print(f"  {i+1:2d}. {r['method']:<45} eff={r['eff_raw']:.3f} mse={r['mse']:.6f} ratio={r['ratio']:.3f}")

    return results


def run_mse_tests(args):
    """运行 MSE 测试"""
    print_section("SVD 量化 MSE 综合测试 (合成数据)")
    asym_tag = "非对称" if args.asymmetric else "对称"
    print(f"  矩阵: 768×768  |  方法: {args.method}  |  quick={args.quick}  |  量化: {asym_tag}")

    W = generate_weight_matrix(768, 768)
    W_f = W.astype(np.float32)
    direct_mse = float(np.mean((W_f - quantize_mse(W, 4, 128, asymmetric=args.asymmetric)) ** 2))
    print(f"  Direct 4-bit ({asym_tag}) baseline MSE: {direct_mse:.6f}")

    asym = args.asymmetric
    method_map = {
        'direct': lambda: test_mse_direct(W, W_f, direct_mse, asym),
        'important': lambda: test_mse_important(W, W_f, direct_mse, args.quick, asym),
        'svd_hybrid': lambda: test_mse_svd_hybrid(W, W_f, direct_mse, args.quick, asym),
        'iterative_svd': lambda: test_mse_iterative_svd(W, W_f, direct_mse, args.quick, asym),
        'outlier_svd': lambda: test_mse_outlier_svd(W, W_f, direct_mse, args.quick, asym),
        'iterative_outlier': lambda: test_mse_iterative_outlier(W, W_f, direct_mse, args.quick, asym),
        'iter_outlier_tune': lambda: test_mse_iter_outlier_tune(W, W_f, direct_mse),
    }

    all_methods = ['direct', 'important', 'svd_hybrid', 'iterative_svd',
                   'outlier_svd', 'iterative_outlier', 'iter_outlier_tune']

    if args.method == 'all':
        methods_to_run = all_methods
    else:
        methods_to_run = [args.method]

    all_results = []
    for m in methods_to_run:
        if m in method_map:
            all_results.extend(method_map[m]())

    # 汇总
    print_section("MSE 测试汇总 (按 ratio 排序)")
    all_results.sort(key=lambda x: x.get('ratio', 999))
    print(f"  {'方法':<50} {'MSE':<12} {'Ratio':<10} {'EffFull':<10}")
    print(f"  {'-' * 82}")
    for r in all_results:
        eff = r.get('eff_full', r.get('eff', r.get('eff_raw', '-')))
        beat = "✅" if r.get('ratio', 999) < 1.0 else "  "
        print(f"  {beat}{r['method']:<49} {r['mse']:<12.6f} {r['ratio']:<10.3f} {eff}")

    return all_results


# ═══════════════════════════════════════════════════════════════════════
#  PPL 测试方案（真实模型）
# ═══════════════════════════════════════════════════════════════════════

def test_ppl_direct(model_id, tokenizer, fp_ppl, device, asymmetric=False):
    """直接量化 PPL"""
    import torch
    tag = "非对称" if asymmetric else "对称"
    print_section(f"直接量化 PPL ({tag})")

    results = []
    for n_bits in [3, 4]:
        model, _ = load_model_and_tokenizer(model_id, device)
        t0 = time.time()
        for name, mod in model.named_modules():
            if hasattr(mod, 'weight') and 'lm_head' not in name:
                W = mod.weight.data.float().cpu().numpy()
                mod.weight.data = torch.from_numpy(
                    quantize_mse(W, n_bits, 128, asymmetric=asymmetric)
                ).to(mod.weight.dtype)
        elapsed = time.time() - t0
        ppl = evaluate_ppl(model, tokenizer)
        delta = ppl - fp_ppl
        print(f"  Direct {n_bits}-bit ({tag}): PPL={ppl:.2f} delta=+{delta:.2f} ({elapsed:.0f}s)")
        results.append({'method': f'direct_{n_bits}bit_{tag}', 'ppl': ppl, 'delta': delta, 'time': elapsed})
        del model
    return results


def test_ppl_iterative_svd(model_id, tokenizer, fp_ppl, device, quick=False, asymmetric=False):
    """迭代残差 SVD PPL"""
    import torch
    import torch.nn as nn
    from quantization.iterative_svd import iterative_residual_svd

    print_section("迭代残差 SVD PPL")

    if quick:
        configs = [
            (8, 4, None, 4, "rank=8 u4v4 s-fp32"),
        ]
    else:
        configs = [
            (4, 4, None, 4, "rank=4 u4v4 s-fp32"),
            (8, 4, None, 4, "rank=8 u4v4 s-fp32"),
            (16, 4, 16, 4, "rank=16 u4v4 s-fp16"),
            (32, 4, 16, 4, "rank=32 u4v4 s-fp16"),
            (8, 3, None, 3, "rank=8 u3v3 s-fp32"),
        ]

    results = []
    for rank, ub, sb, vb, desc in configs:
        print(f"\n  --- {desc} ---")
        model, _ = load_model_and_tokenizer(model_id, device)

        t0 = time.time()
        layer_infos = []
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Linear) and 'lm_head' not in name:
                W = mod.weight.data.float().cpu().numpy()
                W_q, info = iterative_residual_svd(
                    W, 128, max_eff_bits=4.0, rank=rank,
                    u_bits=ub, s_bits=sb, v_bits=vb, asymmetric=asymmetric,
                )
                mod.weight.data = torch.from_numpy(W_q).to(mod.weight.dtype)
                layer_infos.append(info)
        elapsed = time.time() - t0

        ppl = evaluate_ppl(model, tokenizer)
        delta = ppl - fp_ppl
        avg_eff = np.mean([i['effective_bits'] for i in layer_infos])
        avg_rounds = np.mean([i['rounds'] for i in layer_infos])

        print(f"  PPL={ppl:.2f} delta=+{delta:.2f} eff={avg_eff:.3f} "
              f"rounds={avg_rounds:.0f} time={elapsed:.0f}s")
        results.append({
            'method': f'iter_svd_{desc}', 'ppl': ppl, 'delta': delta,
            'eff': avg_eff, 'rounds': avg_rounds, 'time': elapsed,
        })
        del model
    return results


def test_ppl_outlier_svd(model_id, tokenizer, fp_ppl, device, quick=False, asymmetric=False):
    """异常值 SVD + 残差量化 PPL"""
    import torch
    import torch.nn as nn
    from quantization.outlier_svd import outlier_svd_quantize

    print_section("异常值 SVD + 残差量化 PPL")

    if quick:
        configs = [
            (0.15, 1.0, 4, "outlier=15% svd_eff=1.0 res=4b"),
        ]
    else:
        configs = [
            (0.10, 1.0, 4, "outlier=10% svd_eff=1.0 res=4b"),
            (0.15, 1.0, 4, "outlier=15% svd_eff=1.0 res=4b"),
            (0.15, 1.0, 3, "outlier=15% svd_eff=1.0 res=3b"),
            (0.15, 0.5, 4, "outlier=15% svd_eff=0.5 res=4b"),
        ]

    results = []
    for ratio, svd_eff, res_bits, desc in configs:
        print(f"\n  --- {desc} ---")
        model, _ = load_model_and_tokenizer(model_id, device)

        t0 = time.time()
        layer_infos = []
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Linear) and 'lm_head' not in name:
                W = mod.weight.data.float().cpu().numpy()
                W_q, info = outlier_svd_quantize(
                    W, outlier_ratio=ratio, svd_eff_bits=svd_eff,
                    residual_bits=res_bits, svd_rank=4, svd_u_bits=3,
                    svd_v_bits=3, svd_s_bits=16, group_size=128, asymmetric=asymmetric,
                )
                mod.weight.data = torch.from_numpy(W_q).to(mod.weight.dtype)
                layer_infos.append(info)
        elapsed = time.time() - t0

        ppl = evaluate_ppl(model, tokenizer)
        delta = ppl - fp_ppl
        avg_eff = np.mean([i['total_eff_raw'] for i in layer_infos])

        print(f"  PPL={ppl:.2f} delta=+{delta:.2f} eff={avg_eff:.3f} time={elapsed:.0f}s")
        results.append({
            'method': f'outlier_svd_{desc}', 'ppl': ppl, 'delta': delta,
            'eff': avg_eff, 'time': elapsed,
        })
        del model
    return results


def test_ppl_iterative_outlier(model_id, tokenizer, fp_ppl, device, quick=False, asymmetric=False):
    """迭代异常值 SVD + 残差量化 PPL"""
    import torch
    import torch.nn as nn
    from quantization.iterative_outlier_svd import iterative_outlier_svd

    print_section("迭代异常值 SVD + 残差量化 PPL")

    if quick:
        configs = [
            (0.10, 1.00, 4, 3, 3, "ratio=0.10 svd_eff=1.00 rank=4 u3v3"),
        ]
    else:
        configs = [
            (0.10, 0.50, 4, 3, 3, "ratio=0.10 svd_eff=0.50 rank=4 u3v3"),
            (0.10, 1.00, 4, 3, 3, "ratio=0.10 svd_eff=1.00 rank=4 u3v3"),
            (0.15, 1.00, 4, 3, 3, "ratio=0.15 svd_eff=1.00 rank=4 u3v3"),
        ]

    results = []
    for ratio, svd_eff, rank, ub, vb, desc in configs:
        print(f"\n  --- {desc} ---")
        model, _ = load_model_and_tokenizer(model_id, device)

        t0 = time.time()
        layer_infos = []
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Linear) and 'lm_head' not in name:
                W = mod.weight.data.float().cpu().numpy()
                W_q, info = iterative_outlier_svd(
                    W, outlier_ratio=ratio, max_svd_eff=svd_eff,
                    residual_bits=3, rank=rank, u_bits=ub, v_bits=vb,
                    s_bits=16, group_size=128, asymmetric=asymmetric,
                )
                mod.weight.data = torch.from_numpy(W_q).to(mod.weight.dtype)
                layer_infos.append(info)
        elapsed = time.time() - t0

        ppl = evaluate_ppl(model, tokenizer)
        delta = ppl - fp_ppl
        avg_eff = np.mean([i['total_eff_raw'] for i in layer_infos])

        print(f"  PPL={ppl:.2f} delta=+{delta:.2f} eff={avg_eff:.3f} time={elapsed:.0f}s")
        results.append({
            'method': f'iter_outlier_{desc}', 'ppl': ppl, 'delta': delta,
            'eff': avg_eff, 'time': elapsed,
        })
        del model
    return results


def run_ppl_tests(args):
    """运行 PPL 测试"""
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dev_name = f"GPU ({torch.cuda.get_device_name(0)})" if device == "cuda" else "CPU"

    asym_tag = "非对称" if args.asymmetric else "对称"
    print_section(f"SVD 量化 PPL 综合测试 ({asym_tag})")
    print(f"  模型: {args.model}  |  设备: {dev_name}")
    print(f"  数据: wikitext-2, {args.n_samples} samples, seqlen={args.seqlen}")
    print(f"  方法: {args.method}  |  量化: {asym_tag}")

    # FP baseline
    print("\n  加载 FP baseline...")
    model, tokenizer = load_model_and_tokenizer(args.model, device)
    fp_ppl = evaluate_ppl(model, tokenizer, args.n_samples, args.seqlen)
    print(f"  FP Baseline PPL: {fp_ppl:.2f}")
    del model

    asym = args.asymmetric
    method_map = {
        'direct': lambda: test_ppl_direct(args.model, tokenizer, fp_ppl, device, asym),
        'iterative_svd': lambda: test_ppl_iterative_svd(args.model, tokenizer, fp_ppl, device, args.quick, asym),
        'outlier_svd': lambda: test_ppl_outlier_svd(args.model, tokenizer, fp_ppl, device, args.quick, asym),
        'iterative_outlier': lambda: test_ppl_iterative_outlier(args.model, tokenizer, fp_ppl, device, args.quick, asym),
    }

    all_methods_ppl = ['direct', 'iterative_svd', 'outlier_svd', 'iterative_outlier']

    if args.method == 'all':
        methods_to_run = all_methods_ppl
    else:
        methods_to_run = [args.method]

    all_results = []
    for m in methods_to_run:
        if m in method_map:
            all_results.extend(method_map[m]())

    # 汇总
    all_results.sort(key=lambda x: x['ppl'])
    direct_ppl = next((r['ppl'] for r in all_results if 'direct_4bit' in r['method']), None)

    print_section("PPL 测试汇总")
    print(f"  FP baseline: {fp_ppl:.2f}")
    if direct_ppl:
        print(f"  Direct 4-bit: {direct_ppl:.2f} (+{direct_ppl - fp_ppl:.2f})")
    print()
    print(f"  {'方法':<50} {'PPL':<10} {'Delta':<10} {'Eff':<8}")
    print(f"  {'-' * 78}")
    for r in all_results:
        beat = "✅" if direct_ppl and r['ppl'] < direct_ppl else "  "
        eff_str = f"{r.get('eff', '-'):.3f}" if isinstance(r.get('eff'), (int, float)) else str(r.get('eff', '-'))
        print(f"  {beat}{r['method']:<49} {r['ppl']:<10.2f} +{r['delta']:<9.2f} {eff_str}")

    return all_results


# ═══════════════════════════════════════════════════════════════════════
#  幂迭代对比测试
# ═══════════════════════════════════════════════════════════════════════

def run_power_iter_tests(args):
    """幂迭代 SVD 对比测试"""
    from quantization.iterative_svd import iterative_residual_svd
    from quantization.adaptive_iterative import adaptive_iterative_svd
    from quantization.power_iter_svd import power_iteration_svd

    print_section("幂迭代 SVD 对比测试")

    W = generate_weight_matrix(768, 768)
    W_f = W.astype(np.float32)
    direct_mse = float(np.mean((W_f - quantize_mse(W, 4, 128)) ** 2))

    # 测试 1: 幂迭代精度
    print("\n  --- 幂迭代 SVD 精度 ---")
    for rank in [1, 4, 8, 16]:
        U_full, S_full, Vt_full = np.linalg.svd(W, full_matrices=False)
        U_pi, S_pi, Vt_pi = power_iteration_svd(W, rank=rank, n_iter=4)
        s_err = np.mean(np.abs(S_full[:rank] - S_pi) / (S_full[:rank] + 1e-10))
        proj = U_full[:, :rank].T @ U_pi
        u_err = np.linalg.norm(proj - np.eye(rank), 'fro')
        print(f"  rank={rank:<4} S误差={s_err:.6f} U误差={u_err:.6f}")

    # 测试 2: 迭代对比
    print("\n  --- 迭代残差 SVD 对比 (768×768, max_eff=4.0) ---")
    configs = [
        ("full_svd", 4, False, False, "标准SVD rank=4"),
        ("power", 4, True, False, "幂迭代 rank=4"),
        ("power+ws", 4, True, True, "幂迭代+热启动 rank=4"),
    ]

    for method, rank, use_pi, use_ws, desc in configs:
        t0 = time.time()
        W_q, info = adaptive_iterative_svd(
            W, 128, max_eff_bits=4.0, rank=rank, u_bits=3, s_bits=4, v_bits=3,
            use_power_iter=use_pi, power_iter_steps=4, use_warmstart=use_ws,
        )
        elapsed = time.time() - t0
        mse = float(np.mean((W_f - W_q.astype(np.float32)) ** 2))
        print(f"  {desc:<25} rounds={info['rounds']:<4} eff={info['effective_bits']:.3f} "
              f"mse={mse:.6f} ratio={mse/direct_mse:.3f} ({elapsed:.2f}s)")


# ═══════════════════════════════════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════════════════════════════════

class TeeOutput:
    """同时输出到终端和内存缓冲区"""
    def __init__(self, stream):
        self.stream = stream
        self.buffer = io.StringIO()

    def write(self, data):
        self.stream.write(data)
        self.buffer.write(data)

    def flush(self):
        self.stream.flush()

    def getvalue(self):
        return self.buffer.getvalue()


def make_result_dir(scheme, method, quick):
    """创建结果目录: results/<scheme>_<method>_<timestamp>/"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"{scheme}_{method}"
    if quick:
        tag += "_quick"
    result_dir = os.path.join(project_root, "results", f"{tag}_{ts}")
    os.makedirs(result_dir, exist_ok=True)
    return result_dir


def main():
    parser = argparse.ArgumentParser(
        description="SVD 量化统一测试入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python tests/run_tests.py --scheme mse --method all
  python tests/run_tests.py --scheme mse --method iterative_svd --quick
  python tests/run_tests.py --scheme ppl --method direct
  python tests/run_tests.py --scheme ppl --method all --quick
  python tests/run_tests.py --scheme power_iter
        """,
    )
    parser.add_argument("--scheme", required=True,
                        choices=["mse", "ppl", "power_iter"],
                        help="测试方案: mse=合成数据, ppl=真实模型, power_iter=幂迭代对比")
    parser.add_argument("--method", default="all",
                        choices=["all", "direct", "important", "svd_hybrid",
                                 "iterative_svd", "outlier_svd", "iterative_outlier",
                                 "iter_outlier_tune"],
                        help="量化方法 (默认 all)")
    parser.add_argument("--model", default="facebook/opt-125m",
                        help="PPL 测试用的模型")
    parser.add_argument("--n_samples", type=int, default=5,
                        help="PPL 测试样本数")
    parser.add_argument("--seqlen", type=int, default=512,
                        help="PPL 测试序列长度")
    parser.add_argument("--quick", action="store_true",
                        help="快速测试模式（少量配置）")
    parser.add_argument("--asymmetric", action="store_true",
                        help="使用非对称量化（带 zero-point）")

    args = parser.parse_args()

    # 创建结果目录并捕获输出
    result_dir = make_result_dir(args.scheme, args.method, args.quick)
    tee = TeeOutput(sys.stdout)
    old_stdout = sys.stdout
    sys.stdout = tee

    t_start = time.time()

    try:
        if args.scheme == "mse":
            results = run_mse_tests(args)
        elif args.scheme == "ppl":
            results = run_ppl_tests(args)
        elif args.scheme == "power_iter":
            results = run_power_iter_tests(args)
        else:
            print(f"未知方案: {args.scheme}")
            return

        elapsed_total = time.time() - t_start
        print(f"\n总耗时: {elapsed_total:.1f}s")

    finally:
        sys.stdout = old_stdout

    # 保存控制台日志
    log_path = os.path.join(result_dir, "output.log")
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(tee.getvalue())

    # 保存 JSON 结果
    if results:
        json_path = os.path.join(result_dir, "results.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    # 保存运行参数
    meta_path = os.path.join(result_dir, "meta.json")
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump({
            'scheme': args.scheme,
            'method': args.method,
            'quick': args.quick,
            'model': args.model,
            'n_samples': args.n_samples,
            'seqlen': args.seqlen,
            'elapsed_seconds': time.time() - t_start,
            'timestamp': datetime.now().isoformat(),
        }, f, indent=2, ensure_ascii=False)

    # 打印摘要到终端（详细输出已在运行时通过 tee 实时输出）
    print(f"\n📁 结果已保存到: {result_dir}/")
    print(f"   output.log   - 控制台完整输出")
    if results:
        print(f"   results.json - 结构化结果")
    print(f"   meta.json    - 运行参数")


if __name__ == "__main__":
    main()

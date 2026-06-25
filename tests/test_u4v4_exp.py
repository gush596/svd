"""
迭代残差 SVD 新实验 - u4v4 / s不量化

实验目标：
1. rank >= 4, u4v4, s 不量化（float32）
2. 对比 s 量化 vs 不量化
3. 综合等效 bit（含 scale/zero-point 存储开销）
4. GPU 自动检测
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
try:
    import torch
except ImportError:
    torch = None
from quantization.core import quantize_mse
from quantization.iterative_svd import (
    iterative_residual_svd, compute_max_rounds, _get_device
)


def generate_weight_matrix(rows, cols, seed=42):
    """生成模拟 Transformer 权重的测试矩阵"""
    rng = np.random.RandomState(seed)
    rank = min(rows, cols) // 8
    U = rng.randn(rows, rank).astype(np.float32)
    S = np.exp(-np.arange(rank) * 0.3).astype(np.float32)
    V = rng.randn(rank, cols).astype(np.float32)
    W = (U * S.reshape(1, -1)) @ V + rng.randn(rows, cols).astype(np.float32) * 0.1
    n_out = max(1, rows * cols // 100)
    out_idx = rng.choice(rows * cols, n_out, replace=False)
    out = np.zeros(rows * cols, dtype=np.float32)
    out[out_idx] = rng.randn(n_out) * 5.0
    return W + out.reshape(rows, cols)


def print_header(title):
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")


def main():
    # 设备检测
    device = _get_device()
    if torch is not None and device and device.type == 'cuda':
        dev_name = f"GPU ({torch.cuda.get_device_name(0)})"
    else:
        dev_name = "CPU"
    
    print("=" * 80)
    print("  迭代残差 SVD 实验 - u4v4 / s不量化")
    print("=" * 80)
    print(f"  设备: {dev_name}")
    print(f"  矩阵: 768×768 (模拟 OPT-125M q_proj)")
    print()
    
    W = generate_weight_matrix(768, 768)
    W_f = W.astype(np.float32)
    direct_mse = float(np.mean((W_f - quantize_mse(W, 4, 128)) ** 2))
    print(f"  Direct 4-bit baseline MSE: {direct_mse:.6f}")
    
    # ================================================================
    # 实验 1: s 量化 vs s 不量化（rank=4, u4v4）
    # ================================================================
    print_header("实验 1: s 量化 vs s 不量化 (rank=4, u4v4)")
    
    configs_exp1 = [
        # (s_bits, 描述)  s_bits=None 表示不量化
        (4,     "u4 s4 v4 (全量化)"),
        (3,     "u4 s3 v4"),
        (2,     "u4 s2 v4"),
        (None,  "u4 s-fp32 v4 (s不量化)"),
    ]
    
    print(f"\n  {'配置':<28} {'轮数':<6} {'eff_raw':<10} {'eff_full':<10} "
          f"{'MSE':<12} {'vs Direct':<10} {'残差MSE':<12}")
    print(f"  {'-' * 90}")
    
    for s_bits, desc in configs_exp1:
        W_q, info = iterative_residual_svd(W, 128, 4.0, rank=4, u_bits=4, s_bits=s_bits, v_bits=4)
        mse = float(np.mean((W_f - W_q.astype(np.float32)) ** 2))
        
        status = "✅" if info['effective_bits_full'] <= 4.05 else "⚠️"
        print(f"  {status} {desc:<28} {info['rounds']:<6} "
              f"{info['effective_bits']:<10.4f} {info['effective_bits_full']:<10.4f} "
              f"{mse:<12.6f} {mse/direct_mse:<10.4f} {info['residual_mse']:<12.6f}")
    
    # ================================================================
    # 实验 2: 不同 rank（u4v4, s不量化）
    # ================================================================
    print_header("实验 2: 不同 rank (u4v4, s不量化)")
    
    ranks = [4, 6, 8, 12, 16, 24, 32]
    
    print(f"\n  {'rank':<6} {'轮数':<6} {'eff_raw':<10} {'eff_full':<10} "
          f"{'MSE':<12} {'vs Direct':<10} {'残差MSE':<12} {'耗时(s)':<8}")
    print(f"  {'-' * 80}")
    
    for rank in ranks:
        max_r = compute_max_rounds(768, 768, 4.0, rank, 4, None, 4, 128, use_full_eff=True)
        if max_r < 1:
            print(f"  {rank:<6} {'N/A':<6} {'预算不足':<10}")
            continue
        
        t0 = time.time()
        W_q, info = iterative_residual_svd(W, 128, 4.0, rank=rank, u_bits=4, s_bits=None, v_bits=4)
        elapsed = time.time() - t0
        mse = float(np.mean((W_f - W_q.astype(np.float32)) ** 2))
        
        print(f"  {rank:<6} {info['rounds']:<6} "
              f"{info['effective_bits']:<10.4f} {info['effective_bits_full']:<10.4f} "
              f"{mse:<12.6f} {mse/direct_mse:<10.4f} {info['residual_mse']:<12.6f} {elapsed:<8.1f}")
    
    # ================================================================
    # 实验 3: u4v4 vs u3v3 vs u4v3 vs u3v4（s都不量化，rank=8）
    # ================================================================
    print_header("实验 3: 不同 u/v bit 组合 (s不量化, rank=8)")
    
    uv_configs = [
        (3, 3, "u3 s-fp32 v3"),
        (3, 4, "u3 s-fp32 v4"),
        (4, 3, "u4 s-fp32 v3"),
        (4, 4, "u4 s-fp32 v4"),
        (4, 2, "u4 s-fp32 v2"),
        (2, 4, "u2 s-fp32 v4"),
        (2, 2, "u2 s-fp32 v2"),
    ]
    
    print(f"\n  {'配置':<22} {'轮数':<6} {'eff_raw':<10} {'eff_full':<10} "
          f"{'MSE':<12} {'vs Direct':<10}")
    print(f"  {'-' * 75}")
    
    for ub, vb, desc in uv_configs:
        max_r = compute_max_rounds(768, 768, 4.0, 8, ub, None, vb, 128, use_full_eff=True)
        if max_r < 1:
            print(f"  {desc:<22} {'N/A':<6} {'预算不足'}")
            continue
        
        W_q, info = iterative_residual_svd(W, 128, 4.0, rank=8, u_bits=ub, s_bits=None, v_bits=vb)
        mse = float(np.mean((W_f - W_q.astype(np.float32)) ** 2))
        
        print(f"  {desc:<22} {info['rounds']:<6} "
              f"{info['effective_bits']:<10.4f} {info['effective_bits_full']:<10.4f} "
              f"{mse:<12.6f} {mse/direct_mse:<10.4f}")
    
    # ================================================================
    # 实验 4: 收敛曲线（rank=8, u4 s-fp32 v4）
    # ================================================================
    print_header("实验 4: 收敛曲线 (rank=8, u4 s-fp32 v4)")
    
    max_r = compute_max_rounds(768, 768, 4.0, 8, 4, None, 4, 128, use_full_eff=True)
    check_rounds = sorted(set([1, 2, 3, 5, 8, 10, 15, 20, 30, 40, max_r]))
    check_rounds = [r for r in check_rounds if r <= max_r]
    
    print(f"\n  最大轮数: {max_r}")
    print(f"\n  {'轮数':<6} {'eff_raw':<10} {'eff_full':<10} {'MSE':<12} {'vs Direct':<10} {'残差MSE':<12}")
    print(f"  {'-' * 65}")
    
    for n_r in check_rounds:
        W_q, info = iterative_residual_svd(W, 128, 4.0, rank=8, u_bits=4, s_bits=None, v_bits=4, n_rounds=n_r)
        mse = float(np.mean((W_f - W_q.astype(np.float32)) ** 2))
        
        print(f"  {n_r:<6} {info['effective_bits']:<10.4f} {info['effective_bits_full']:<10.4f} "
              f"{mse:<12.6f} {mse/direct_mse:<10.4f} {info['residual_mse']:<12.6f}")
    
    # ================================================================
    # 实验 5: 不同矩阵尺寸
    # ================================================================
    print_header("实验 5: 矩阵尺寸 scaling (rank=8, u4 s-fp32 v4)")
    
    sizes = [
        ("256×256",   256,  256),
        ("512×512",   512,  512),
        ("768×768",   768,  768),
        ("1024×1024", 1024, 1024),
        ("768×3072",  768,  3072),
        ("3072×768",  3072, 768),
    ]
    
    print(f"\n  {'尺寸':<15} {'轮数':<6} {'eff_raw':<10} {'eff_full':<10} "
          f"{'MSE':<12} {'vs Direct':<10} {'耗时(s)':<8}")
    print(f"  {'-' * 75}")
    
    for name, rows, cols in sizes:
        W_sz = generate_weight_matrix(rows, cols)
        W_sz_f = W_sz.astype(np.float32)
        direct_mse_sz = float(np.mean((W_sz_f - quantize_mse(W_sz, 4, 128)) ** 2))
        
        t0 = time.time()
        W_q, info = iterative_residual_svd(W_sz, 128, 4.0, rank=8, u_bits=4, s_bits=None, v_bits=4)
        elapsed = time.time() - t0
        mse = float(np.mean((W_sz_f - W_q.astype(np.float32)) ** 2))
        
        print(f"  {name:<15} {info['rounds']:<6} "
              f"{info['effective_bits']:<10.4f} {info['effective_bits_full']:<10.4f} "
              f"{mse:<12.6f} {mse/direct_mse_sz:<10.4f} {elapsed:<8.1f}")
    
    # ================================================================
    # 实验 6: group_size 影响（rank=8, u4 s-fp32 v4）
    # ================================================================
    print_header("实验 6: group_size 影响 (rank=8, u4 s-fp32 v4)")
    
    group_sizes = [32, 64, 128, 256, 512]
    
    print(f"\n  {'group_size':<12} {'轮数':<6} {'eff_raw':<10} {'eff_full':<10} "
          f"{'MSE':<12} {'vs Direct':<10}")
    print(f"  {'-' * 60}")
    
    for gs in group_sizes:
        W_q, info = iterative_residual_svd(W, gs, 4.0, rank=8, u_bits=4, s_bits=None, v_bits=4)
        mse = float(np.mean((W_f - W_q.astype(np.float32)) ** 2))
        
        print(f"  {gs:<12} {info['rounds']:<6} "
              f"{info['effective_bits']:<10.4f} {info['effective_bits_full']:<10.4f} "
              f"{mse:<12.6f} {mse/direct_mse:<10.4f}")
    
    print(f"\n{'=' * 80}")
    print("  ✅ 全部实验完成")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()

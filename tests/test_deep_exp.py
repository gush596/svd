"""
迭代残差 SVD 深入实验 - 极限探索

方向：
1. 极高 rank (48, 64, 96, 128) + u2v2/u3v3
2. 不同 max_eff_bits 预算 (3.0, 3.5, 4.0)
3. 残差加权策略（早期轮次用高精度）
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
from quantization.core import quantize_mse
from quantization.iterative_svd import iterative_residual_svd, compute_max_rounds


def generate_weight_matrix(rows, cols, seed=42):
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


def main():
    print("=" * 80)
    print("  迭代残差 SVD 深入实验 - 极限探索")
    print("=" * 80)
    
    W = generate_weight_matrix(768, 768)
    W_f = W.astype(np.float32)
    direct_mse = float(np.mean((W_f - quantize_mse(W, 4, 128)) ** 2))
    print(f"  Direct 4-bit baseline MSE: {direct_mse:.6f}")
    
    # ================================================================
    # 实验 A: 极高 rank 探索 (s不量化, u4v4)
    # ================================================================
    print(f"\n{'=' * 80}")
    print("  实验 A: 极高 rank 探索 (s不量化)")
    print(f"{'=' * 80}")
    
    for ub, vb, label in [(4, 4, "u4v4"), (3, 3, "u3v3"), (4, 3, "u4v3")]:
        print(f"\n  --- {label} ---")
        print(f"  {'rank':<6} {'轮数':<6} {'eff_raw':<10} {'eff_full':<10} "
              f"{'MSE':<12} {'vs Direct':<10} {'耗时(s)':<8}")
        print(f"  {'-' * 65}")
        
        for rank in [4, 8, 12, 16, 24, 32, 48, 64, 96, 128]:
            max_r = compute_max_rounds(768, 768, 4.0, rank, ub, None, vb, 128, use_full_eff=True)
            if max_r < 1:
                continue
            
            t0 = time.time()
            W_q, info = iterative_residual_svd(W, 128, 4.0, rank=rank, u_bits=ub, s_bits=None, v_bits=vb)
            elapsed = time.time() - t0
            mse = float(np.mean((W_f - W_q.astype(np.float32)) ** 2))
            
            print(f"  {rank:<6} {info['rounds']:<6} "
                  f"{info['effective_bits']:<10.4f} {info['effective_bits_full']:<10.4f} "
                  f"{mse:<12.6f} {mse/direct_mse:<10.4f} {elapsed:<8.1f}")
    
    # ================================================================
    # 实验 B: 不同 eff 预算 (rank=32, u4v4, s不量化)
    # ================================================================
    print(f"\n{'=' * 80}")
    print("  实验 B: 不同 eff 预算 (rank=32, u4 s-fp32 v4)")
    print(f"{'=' * 80}")
    
    print(f"\n  {'max_eff':<10} {'轮数':<6} {'eff_raw':<10} {'eff_full':<10} "
          f"{'MSE':<12} {'vs Direct':<10}")
    print(f"  {'-' * 58}")
    
    for max_eff in [2.0, 2.5, 3.0, 3.5, 3.8, 4.0, 4.5, 5.0]:
        max_r = compute_max_rounds(768, 768, max_eff, 32, 4, None, 4, 128, use_full_eff=True)
        if max_r < 1:
            continue
        
        W_q, info = iterative_residual_svd(W, 128, max_eff, rank=32, u_bits=4, s_bits=None, v_bits=4)
        mse = float(np.mean((W_f - W_q.astype(np.float32)) ** 2))
        
        print(f"  {max_eff:<10.1f} {info['rounds']:<6} "
              f"{info['effective_bits']:<10.4f} {info['effective_bits_full']:<10.4f} "
              f"{mse:<12.6f} {mse/direct_mse:<10.4f}")
    
    # ================================================================
    # 实验 C: 混合精度策略 — 早期高精度，后期低精度
    # ================================================================
    print(f"\n{'=' * 80}")
    print("  实验 C: 混合轮次精度 (rank=32, s不量化)")
    print(f"{'=' * 80}")
    
    # 策略: 前 N 轮用 u4v4, 后面用 u3v3
    print(f"\n  思路: 前几轮捕获主要能量(高精度)，后面轮次补充细节(低精度)")
    
    # 先跑一个纯 u4v4 作为 baseline
    W_q_base, info_base = iterative_residual_svd(W, 128, 4.0, rank=32, u_bits=4, s_bits=None, v_bits=4)
    mse_base = float(np.mean((W_f - W_q_base.astype(np.float32)) ** 2))
    total_rounds = info_base['rounds']
    print(f"\n  Baseline u4v4: {total_rounds}轮, eff_full={info_base['effective_bits_full']:.4f}, MSE={mse_base:.6f}")
    
    # 混合策略: 手动实现
    from quantization.core import quantize_mse as qm
    import math
    
    def mixed_precision_svd(W, rank, early_rounds, late_rounds, early_ub, early_vb, late_ub, late_vb, group_size=128):
        """前 N 轮用高精度，后面用低精度"""
        out_dim, in_dim = W.shape
        total_params = W.size
        W_f = W.astype(np.float32)
        residual = W_f.copy()
        W_approx = np.zeros_like(W_f)
        
        all_rounds = early_rounds + late_rounds
        bits_log = []
        
        for i in range(all_rounds):
            ub = early_ub if i < early_rounds else late_ub
            vb = early_vb if i < early_rounds else late_vb
            
            U, S, Vt = np.linalg.svd(residual, full_matrices=False)
            actual_rank = min(rank, len(S))
            U_k, S_k, V_k = U[:, :actual_rank], S[:actual_rank], Vt[:actual_rank, :]
            
            gs = min(group_size, max(8, actual_rank))
            U_q = qm(U_k, n_bits=ub, group_size=gs)
            S_q = S_k  # 不量化
            V_q = qm(V_k, n_bits=vb, group_size=gs)
            
            component = U_q @ np.diag(S_q) @ V_q
            W_approx += component
            residual -= component
            
            round_bits = actual_rank * (out_dim * ub + in_dim * vb) + actual_rank * 32
            bits_log.append(round_bits)
        
        svd_bits = sum(bits_log)
        # scale bits
        gs_u = min(group_size, max(8, rank))
        gs_v = min(group_size, max(8, rank))
        n_groups_u = math.ceil(out_dim * rank / gs_u)
        n_groups_v = math.ceil(rank * in_dim / gs_v)
        scale_per_round = (n_groups_u + n_groups_v) * 32
        scale_total = scale_per_round * all_rounds
        
        eff_raw = svd_bits / total_params
        eff_full = (svd_bits + scale_total) / total_params
        mse = float(np.mean((W_f - W_approx) ** 2))
        
        return W_approx, {'rounds': all_rounds, 'effective_bits': eff_raw, 'effective_bits_full': eff_full, 'mse': mse}
    
    strategies = [
        (8, 15, 4, 4, 4, 4, "8轮u4v4 + 15轮u4v4 (全部u4v4)"),
        (5, 18, 4, 4, 3, 3, "5轮u4v4 + 18轮u3v3"),
        (3, 20, 4, 4, 3, 3, "3轮u4v4 + 20轮u3v3"),
        (3, 20, 4, 4, 2, 2, "3轮u4v4 + 20轮u2v2"),
        (1, 22, 4, 4, 3, 3, "1轮u4v4 + 22轮u3v3"),
        (0, 23, 4, 4, 4, 4, "0轮u4v4 + 23轮u4v4 (baseline)"),
    ]
    
    print(f"\n  {'策略':<32} {'eff_raw':<10} {'eff_full':<10} {'MSE':<12} {'vs Direct':<10}")
    print(f"  {'-' * 75}")
    
    for early_n, late_n, e_ub, e_vb, l_ub, l_vb, desc in strategies:
        _, info_m = mixed_precision_svd(W, 32, early_n, late_n, e_ub, e_vb, l_ub, l_vb)
        mse_m = info_m['mse']
        
        print(f"  {desc:<32} {info_m['effective_bits']:<10.4f} {info_m['effective_bits_full']:<10.4f} "
              f"{mse_m:<12.6f} {mse_m/direct_mse:<10.4f}")
    
    print(f"\n{'=' * 80}")
    print("  ✅ 深入实验完成")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()

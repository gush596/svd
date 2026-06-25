"""
迭代残差 SVD 新实验

实验方向：
1. 非对称量化 vs 对称量化
2. 混合 bit 配置（U/S/V 不同精度）
3. 自适应轮数策略
4. 不同 group_size 的影响
5. 更多 rank 和 bit 组合
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
from quantization.core import quantize_mse, quantize_mse_asymmetric
from quantization.iterative_svd import iterative_residual_svd, compute_max_rounds


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


def iterative_residual_svd_asymmetric(
    W, group_size=128, max_eff_bits=4.0, rank=1,
    u_bits=3, s_bits=4, v_bits=3, n_rounds=None
):
    """使用非对称量化的迭代残差 SVD"""
    out_dim, in_dim = W.shape
    total_params = W.size
    W_f = W.astype(np.float32)
    
    round_bits = rank * (out_dim * u_bits + s_bits + in_dim * v_bits)
    round_eff = round_bits / total_params
    
    if n_rounds is not None:
        n_r = n_rounds
    else:
        n_r = int(max_eff_bits / round_eff)
    
    if n_r <= 0:
        W_q = quantize_mse(W_f, n_bits=4, group_size=group_size)
        return W_q, {'rounds': 0, 'effective_bits': 4.0, 'fallback': True}
    
    residual = W_f.copy()
    W_approx = np.zeros_like(W_f)
    round_infos = []
    
    for i in range(n_r):
        U, S, Vt = np.linalg.svd(residual, full_matrices=False)
        actual_rank = min(rank, len(S))
        
        U_k = U[:, :actual_rank]
        S_k = S[:actual_rank]
        V_k = Vt[:actual_rank, :]
        
        # 非对称量化
        gs_u = min(group_size, max(8, actual_rank))
        U_q = quantize_mse_asymmetric(U_k, n_bits=u_bits, group_size=gs_u)
        
        gs_s = min(group_size, max(8, actual_rank))
        S_q = quantize_mse_asymmetric(S_k.reshape(1, -1), n_bits=s_bits, group_size=gs_s).reshape(-1)
        
        gs_v = min(group_size, max(8, actual_rank))
        V_q = quantize_mse_asymmetric(V_k, n_bits=v_bits, group_size=gs_v)
        
        component = U_q @ np.diag(S_q) @ V_q
        W_approx = W_approx + component
        residual = residual - component
        
        round_infos.append({
            'round': i + 1, 'rank': actual_rank,
            'residual_mse': float(np.mean(residual ** 2)),
        })
    
    svd_bits_total = sum(r['rank'] * (out_dim * u_bits + s_bits + in_dim * v_bits) for r in round_infos)
    eff_bits = svd_bits_total / total_params
    
    return W_approx, {
        'rounds': len(round_infos),
        'effective_bits': float(eff_bits),
        'mse': float(np.mean((W_f - W_approx) ** 2)),
        'residual_mse': float(np.mean(residual ** 2)),
        'asymmetric': True,
    }


def experiment1_asymmetric_vs_symmetric():
    """实验 1: 非对称量化 vs 对称量化"""
    print("=" * 70)
    print("实验 1: 非对称量化 vs 对称量化")
    print("=" * 70)
    
    W = generate_weight_matrix(768, 768)
    W_f = W.astype(np.float32)
    direct_mse = float(np.mean((W_f - quantize_mse(W, 4, 128)) ** 2))
    
    configs = [
        (1, 3, 4, 3, "rank=1 u3s4v3"),
        (1, 2, 3, 2, "rank=1 u2s3v2"),
        (4, 3, 4, 3, "rank=4 u3s4v3"),
        (4, 2, 3, 2, "rank=4 u2s3v2"),
        (8, 3, 4, 3, "rank=8 u3s4v3"),
    ]
    
    print(f"\n{'配置':<22} {'对称MSE':<14} {'非对称MSE':<14} {'改善%':<10} {'轮数':<8} {'EffBits':<10}")
    print("-" * 80)
    
    for rank, ub, sb, vb, desc in configs:
        # 对称
        W_q_sym, info_sym = iterative_residual_svd(W, 128, 4.0, rank, ub, sb, vb)
        mse_sym = float(np.mean((W_f - W_q_sym.astype(np.float32)) ** 2))
        
        # 非对称
        W_q_asym, info_asym = iterative_residual_svd_asymmetric(W, 128, 4.0, rank, ub, sb, vb)
        mse_asym = float(np.mean((W_f - W_q_asym.astype(np.float32)) ** 2))
        
        improvement = (mse_sym - mse_asym) / mse_sym * 100
        
        print(f"{desc:<22} {mse_sym:<14.6f} {mse_asym:<14.6f} {improvement:<10.1f} "
              f"{info_sym['rounds']:<8} {info_sym['effective_bits']:<10.3f}")
    
    print(f"\nDirect 4-bit baseline MSE: {direct_mse:.6f}")


def experiment2_mixed_bits():
    """实验 2: 混合 bit 配置"""
    print("\n" + "=" * 70)
    print("实验 2: 混合 bit 配置 (U/S/V 不同精度)")
    print("=" * 70)
    
    W = generate_weight_matrix(768, 768)
    W_f = W.astype(np.float32)
    direct_mse = float(np.mean((W_f - quantize_mse(W, 4, 128)) ** 2))
    
    # 固定 rank=4, 探索不同 bit 分配
    configs = [
        # (u_bits, s_bits, v_bits, 描述)
        (2, 2, 2, "u2s2v2 (最低)"),
        (2, 3, 2, "u2s3v2"),
        (2, 4, 2, "u2s4v2"),
        (3, 2, 3, "u3s2v3"),
        (3, 3, 3, "u3s3v3"),
        (3, 4, 3, "u3s4v3 (当前最优)"),
        (3, 4, 2, "u3s4v2"),
        (2, 4, 3, "u2s4v3"),
        (4, 4, 4, "u4s4v4"),
        (4, 3, 3, "u4s3v3"),
        (3, 3, 4, "u3s3v4"),
        (4, 2, 2, "u4s2v2"),
    ]
    
    rank = 4
    print(f"\n{'配置':<22} {'轮数':<8} {'EffBits':<10} {'MSE':<14} {'vs Direct':<12} {'vs u3s4v3':<12}")
    print("-" * 80)
    
    baseline_mse = None
    for ub, sb, vb, desc in configs:
        W_q, info = iterative_residual_svd(W, 128, 4.0, rank, ub, sb, vb)
        mse = float(np.mean((W_f - W_q.astype(np.float32)) ** 2))
        
        if desc == "u3s4v3 (当前最优)":
            baseline_mse = mse
        
        ratio_direct = mse / direct_mse
        ratio_baseline = mse / baseline_mse if baseline_mse else 0
        
        status = "✅" if info['effective_bits'] <= 4.05 else "⚠️"
        print(f"{status} {desc:<22} {info['rounds']:<8} {info['effective_bits']:<10.3f} "
              f"{mse:<14.6f} {ratio_direct:<12.3f} {ratio_baseline:<12.3f}")


def experiment3_group_size_impact():
    """实验 3: group_size 对量化精度的影响"""
    print("\n" + "=" * 70)
    print("实验 3: group_size 影响")
    print("=" * 70)
    
    W = generate_weight_matrix(768, 768)
    W_f = W.astype(np.float32)
    direct_mse = float(np.mean((W_f - quantize_mse(W, 4, 128)) ** 2))
    
    group_sizes = [32, 64, 128, 256, 512]
    
    print(f"\n{'group_size':<14} {'轮数':<8} {'EffBits':<10} {'MSE':<14} {'vs Direct':<12}")
    print("-" * 60)
    
    for gs in group_sizes:
        W_q, info = iterative_residual_svd(W, gs, 4.0, 4, 3, 4, 3)
        mse = float(np.mean((W_f - W_q.astype(np.float32)) ** 2))
        
        print(f"{gs:<14} {info['rounds']:<8} {info['effective_bits']:<10.3f} "
              f"{mse:<14.6f} {mse/direct_mse:<12.3f}")


def experiment4_rank_sweep():
    """实验 4: 更细粒度的 rank 扫描"""
    print("\n" + "=" * 70)
    print("实验 4: rank 扫描 (u3s4v3, max_eff=4.0)")
    print("=" * 70)
    
    W = generate_weight_matrix(768, 768)
    W_f = W.astype(np.float32)
    direct_mse = float(np.mean((W_f - quantize_mse(W, 4, 128)) ** 2))
    
    ranks = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
    
    print(f"\n{'rank':<8} {'轮数':<8} {'EffBits':<10} {'MSE':<14} {'vs Direct':<12} {'残差MSE':<14}")
    print("-" * 70)
    
    for rank in ranks:
        max_r = compute_max_rounds(768, 768, 4.0, rank, 3, 4, 3)
        if max_r < 1:
            print(f"{rank:<8} {'N/A':<8} {'N/A':<10} {'N/A':<14} {'N/A':<12} {'预算不足'}")
            continue
        
        W_q, info = iterative_residual_svd(W, 128, 4.0, rank, 3, 4, 3)
        mse = float(np.mean((W_f - W_q.astype(np.float32)) ** 2))
        
        print(f"{rank:<8} {info['rounds']:<8} {info['effective_bits']:<10.3f} "
              f"{mse:<14.6f} {mse/direct_mse:<12.3f} {info['residual_mse']:<14.6f}")


def experiment5_convergence_analysis():
    """实验 5: 收敛性分析 - MSE 随轮数变化"""
    print("\n" + "=" * 70)
    print("实验 5: 收敛性分析 (rank=4, u3s4v3)")
    print("=" * 70)
    
    W = generate_weight_matrix(768, 768)
    W_f = W.astype(np.float32)
    direct_mse = float(np.mean((W_f - quantize_mse(W, 4, 128)) ** 2))
    
    max_r = compute_max_rounds(768, 768, 4.0, 4, 3, 4, 3)
    check_rounds = [1, 2, 5, 10, 20, 30, 50, 80, max_r]
    
    print(f"\n{'轮数':<8} {'EffBits':<10} {'MSE':<14} {'vs Direct':<12} {'残差MSE':<14}")
    print("-" * 65)
    
    for n_r in check_rounds:
        if n_r > max_r:
            break
        W_q, info = iterative_residual_svd(W, 128, 4.0, 4, 3, 4, 3, n_rounds=n_r)
        mse = float(np.mean((W_f - W_q.astype(np.float32)) ** 2))
        
        print(f"{n_r:<8} {info['effective_bits']:<10.3f} {mse:<14.6f} "
              f"{mse/direct_mse:<12.3f} {info['residual_mse']:<14.6f}")


def experiment6_matrix_size_scaling():
    """实验 6: 不同矩阵尺寸的表现"""
    print("\n" + "=" * 70)
    print("实验 6: 矩阵尺寸 scaling (rank=4, u3s4v3, max_eff=4.0)")
    print("=" * 70)
    
    sizes = [
        ("128×128", 128, 128),
        ("256×256", 256, 256),
        ("512×512", 512, 512),
        ("768×768", 768, 768),
        ("1024×1024", 1024, 1024),
        ("768×3072", 768, 3072),
        ("3072×768", 3072, 768),
    ]
    
    print(f"\n{'尺寸':<15} {'轮数':<8} {'EffBits':<10} {'MSE':<14} {'vs Direct':<12} {'耗时(s)':<10}")
    print("-" * 70)
    
    for name, rows, cols in sizes:
        W = generate_weight_matrix(rows, cols)
        W_f = W.astype(np.float32)
        direct_mse = float(np.mean((W_f - quantize_mse(W, 4, 128)) ** 2))
        
        t0 = time.time()
        W_q, info = iterative_residual_svd(W, 128, 4.0, 4, 3, 4, 3)
        elapsed = time.time() - t0
        
        mse = float(np.mean((W_f - W_q.astype(np.float32)) ** 2))
        
        print(f"{name:<15} {info['rounds']:<8} {info['effective_bits']:<10.3f} "
              f"{mse:<14.6f} {mse/direct_mse:<12.3f} {elapsed:<10.1f}")


def main():
    print("=" * 70)
    print("迭代残差 SVD - 新实验")
    print("=" * 70)
    print(f"矩阵: 768×768 (模拟 OPT-125M q_proj 层)")
    print(f"FP baseline MSE: N/A (每组独立计算)")
    print()
    
    t0 = time.time()
    
    experiment1_asymmetric_vs_symmetric()
    experiment2_mixed_bits()
    experiment3_group_size_impact()
    experiment4_rank_sweep()
    experiment5_convergence_analysis()
    experiment6_matrix_size_scaling()
    
    total = time.time() - t0
    print(f"\n✅ 全部实验完成，总耗时: {total:.0f}s")


if __name__ == "__main__":
    main()

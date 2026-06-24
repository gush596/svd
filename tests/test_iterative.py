"""迭代残差 SVD 测试 - NumPy 合成数据"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time


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


def test_numpy():
    """NumPy 合成数据测试"""
    from quantization.core import quantize_mse
    from quantization.iterative_svd import iterative_residual_svd, compute_max_rounds
    
    print("=" * 70)
    print("迭代残差 SVD - NumPy 测试（残差舍弃模式）")
    print("=" * 70)
    
    W = generate_weight_matrix(768, 768)
    W_f = W.astype(np.float32)
    direct_mse = float(np.mean((W_f - quantize_mse(W, 4, 128)) ** 2))
    print(f"Direct 4-bit baseline MSE: {direct_mse:.6f}")
    print(f"权重矩阵: 768×768, 总参数: {W.size:,}")
    print()
    
    # 计算各配置下 4.0 bit 预算内的最大轮数
    print(f"{'rank':<6} {'u_bits':<8} {'s_bits':<8} {'v_bits':<8} {'每轮bits':<12} {'每轮eff':<12} {'4.0bit轮数':<12}")
    print("-" * 70)
    
    configs = [
        (1, 3, 4, 3),
        (1, 2, 3, 2),
        (1, 2, 2, 2),
        (4, 3, 4, 3),
        (4, 2, 3, 2),
        (8, 3, 4, 3),
        (8, 2, 3, 2),
    ]
    
    for rank, ub, sb, vb in configs:
        round_bits = rank * (768 * ub + sb + 768 * vb)
        round_eff = round_bits / W.size
        max_r = int(4.0 / round_eff)
        print(f"{rank:<6} {ub:<8} {sb:<8} {vb:<8} {round_bits:<12,} {round_eff:<12.6f} {max_r:<12}")
    
    print()
    
    # 测试不同配置下的 MSE 表现
    print("=" * 70)
    print("MSE 测试 (max_eff_bits=4.0)")
    print("=" * 70)
    print()
    
    test_configs = [
        (1, 3, 4, 3, "rank=1, u3s4v3"),
        (1, 2, 3, 2, "rank=1, u2s3v2"),
        (1, 2, 2, 2, "rank=1, u2s2v2"),
        (4, 3, 4, 3, "rank=4, u3s4v3"),
        (4, 2, 3, 2, "rank=4, u2s3v2"),
        (8, 3, 4, 3, "rank=8, u3s4v3"),
        (8, 2, 3, 2, "rank=8, u2s3v2"),
    ]
    
    print(f"{'配置':<22} {'轮数':<8} {'EffBits':<10} {'MSE':<12} {'Ratio':<10} {'残差MSE':<12}")
    print("-" * 80)
    
    for rank, ub, sb, vb, desc in test_configs:
        W_q, info = iterative_residual_svd(W, 128, max_eff_bits=4.0, rank=rank, u_bits=ub, s_bits=sb, v_bits=vb)
        mse = float(np.mean((W_f - W_q.astype(np.float32)) ** 2))
        eff = info['effective_bits']
        ratio = mse / direct_mse
        res_mse = info.get('residual_mse', 0)
        
        status = "✅" if eff <= 4.05 else "⚠️"
        print(f"{status} {desc:<22} {info['rounds']:<8} {eff:<10.3f} {mse:<12.6f} {ratio:<10.3f} {res_mse:<12.6f}")
    
    # 测试不同 max_eff_bits
    print()
    print("=" * 70)
    print("不同 max_eff_bits 下的表现 (rank=1, u3s4v3)")
    print("=" * 70)
    print()
    
    print(f"{'MaxEffBits':<12} {'轮数':<8} {'ActualEff':<12} {'MSE':<12} {'Ratio':<10}")
    print("-" * 55)
    
    for max_eff in [1.0, 2.0, 3.0, 3.5, 4.0, 4.05, 4.5, 5.0]:
        W_q, info = iterative_residual_svd(W, 128, max_eff_bits=max_eff, rank=1, u_bits=3, s_bits=4, v_bits=3)
        mse = float(np.mean((W_f - W_q.astype(np.float32)) ** 2))
        print(f"{max_eff:<12.2f} {info['rounds']:<8} {info['effective_bits']:<12.6f} {mse:<12.6f} {mse/direct_mse:<10.3f}")
    
    # 测试不同矩阵尺寸
    print()
    print("=" * 70)
    print("多矩阵尺寸测试 (rank=1, u3s4v3, max_eff=4.0)")
    print("=" * 70)
    print()
    
    sizes = [("256×256", 256, 256), ("768×768", 768, 768), 
             ("512×2048", 512, 2048), ("2048×512", 2048, 512)]
    
    for name, rows, cols in sizes:
        W = generate_weight_matrix(rows, cols)
        W_f = W.astype(np.float32)
        direct_mse = float(np.mean((W_f - quantize_mse(W, 4, 128)) ** 2))
        
        W_q, info = iterative_residual_svd(W, 128, max_eff_bits=4.0, rank=1, u_bits=3, s_bits=4, v_bits=3)
        mse = float(np.mean((W_f - W_q.astype(np.float32)) ** 2))
        
        print(f"  {name:<15} rounds={info['rounds']:<6} eff={info['effective_bits']:.3f} "
              f"mse_ratio={mse/direct_mse:.3f}")


if __name__ == "__main__":
    test_numpy()

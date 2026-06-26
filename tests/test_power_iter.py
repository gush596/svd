"""幂迭代 SVD 对比测试

比较:
1. 标准 SVD 迭代 vs 幂迭代 SVD 迭代
2. 固定 rank vs 自适应 rank
3. 有/无热启动
4. 不同幂迭代步数
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
from quantization.core import quantize_mse
from quantization.iterative_svd import iterative_residual_svd
from quantization.power_iter_svd import power_iteration_svd
from quantization.adaptive_iterative import adaptive_iterative_svd


def generate_weight_matrix(rows, cols, seed=42):
    """生成具有典型结构的权重矩阵（低秩 + 稀疏离群值）"""
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


def test_power_iter_accuracy():
    """测试幂迭代 SVD 的精度（与完整 SVD 对比）"""
    print("=" * 70)
    print("测试 1: 幂迭代 SVD 精度")
    print("=" * 70)
    
    W = generate_weight_matrix(768, 768)
    
    print(f"\n{'rank':<6} {'n_iter':<8} {'SVD时间':<12} {'PI时间':<12} {'PI加速':<10} {'U误差':<12} {'S误差':<12}")
    print("-" * 75)
    
    for rank in [1, 2, 4, 8, 16]:
        # 完整 SVD
        t0 = time.time()
        U_full, S_full, Vt_full = np.linalg.svd(W, full_matrices=False)
        t_svd = time.time() - t0
        U_full_k = U_full[:, :rank]
        S_full_k = S_full[:rank]
        Vt_full_k = Vt_full[:rank, :]
        
        for n_iter in [2, 4, 6]:
            t0 = time.time()
            U_pi, S_pi, Vt_pi = power_iteration_svd(W, rank=rank, n_iter=n_iter)
            t_pi = time.time() - t0
            
            # 比较奇异值误差
            s_err = np.abs(S_full_k - S_pi) / (S_full_k + 1e-10)
            s_err = np.mean(s_err)
            
            # 比较子空间误差（通过投影矩阵）
            # ||U_full^T @ U_pi - I|| (理想情况应为对角矩阵)
            proj = U_full_k.T @ U_pi
            u_err = np.linalg.norm(proj - np.eye(rank), 'fro')
            
            speedup = t_svd / max(t_pi, 1e-6)
            
            print(f"{rank:<6} {n_iter:<8} {t_svd:<12.4f}s {t_pi:<12.4f}s {speedup:<10.1f}x {u_err:<12.6f} {s_err:<12.6f}")
    
    print()


def test_iterative_comparison():
    """对比完整 SVD 迭代 vs 幂迭代 SVD 迭代"""
    print("=" * 70)
    print("测试 2: 迭代残差 SVD 对比 (768×768, max_eff=4.0)")
    print("=" * 70)
    
    W = generate_weight_matrix(768, 768)
    W_f = W.astype(np.float32)
    direct_mse = float(np.mean((W_f - quantize_mse(W, 4, 128)) ** 2))
    
    configs = [
        # (method, rank, u_bits, s_bits, v_bits, kwargs, desc)
        ("full_svd", 1, 3, 4, 3, {}, "标准SVD rank=1 u3s4v3"),
        ("full_svd", 4, 3, 4, 3, {}, "标准SVD rank=4 u3s4v3"),
        ("full_svd", 8, 3, 4, 3, {}, "标准SVD rank=8 u3s4v3"),
        ("power", 1, 3, 4, 3, {"power_iter_steps": 4}, "幂迭代 rank=1 u3s4v3"),
        ("power", 4, 3, 4, 3, {"power_iter_steps": 4}, "幂迭代 rank=4 u3s4v3"),
        ("power", 8, 3, 4, 3, {"power_iter_steps": 4}, "幂迭代 rank=8 u3s4v3"),
        ("power+ws", 4, 3, 4, 3, {"power_iter_steps": 4, "use_warmstart": True}, "幂迭代+热启动 rank=4"),
        ("power+ws", 8, 3, 4, 3, {"power_iter_steps": 4, "use_warmstart": True}, "幂迭代+热启动 rank=8"),
    ]
    
    print(f"\n{'方法':<28} {'轮数':<6} {'EffBits':<10} {'MSE':<12} {'Ratio':<8} {'时间':<10}")
    print("-" * 78)
    
    for method, rank, ub, sb, vb, kwargs, desc in configs:
        t0 = time.time()
        
        if method == "full_svd":
            W_q, info = iterative_residual_svd(
                W, 128, max_eff_bits=4.0, rank=rank, u_bits=ub, s_bits=sb, v_bits=vb,
            )
        elif method == "power":
            W_q, info = adaptive_iterative_svd(
                W, 128, max_eff_bits=4.0, rank=rank, u_bits=ub, s_bits=sb, v_bits=vb,
                use_power_iter=True, use_warmstart=False, **kwargs,
            )
        elif method == "power+ws":
            W_q, info = adaptive_iterative_svd(
                W, 128, max_eff_bits=4.0, rank=rank, u_bits=ub, s_bits=sb, v_bits=vb,
                use_power_iter=True, **kwargs,
            )
        
        elapsed = time.time() - t0
        mse = float(np.mean((W_f - W_q.astype(np.float32)) ** 2))
        ratio = mse / direct_mse
        
        status = "✅" if info['effective_bits'] <= 4.05 else "⚠️"
        print(f"{status} {desc:<28} {info['rounds']:<6} {info['effective_bits']:<10.3f} "
              f"{mse:<12.6f} {ratio:<8.3f} {elapsed:<10.2f}s")
    
    print(f"\n  Direct 4-bit baseline MSE: {direct_mse:.6f}")


def test_adaptive_rank():
    """测试自适应 rank 模式"""
    print("\n" + "=" * 70)
    print("测试 3: 自适应 rank 模式")
    print("=" * 70)
    
    W = generate_weight_matrix(768, 768)
    W_f = W.astype(np.float32)
    direct_mse = float(np.mean((W_f - quantize_mse(W, 4, 128)) ** 2))
    
    thresholds = [0.7, 0.8, 0.9, 0.95]
    max_ranks = [4, 8, 16]
    
    print(f"\n{'阈值':<8} {'MaxRank':<10} {'轮数':<6} {'EffBits':<10} {'MSE':<12} {'Ratio':<8} {'时间':<10}")
    print("-" * 68)
    
    for thr in thresholds:
        for mr in max_ranks:
            t0 = time.time()
            W_q, info = adaptive_iterative_svd(
                W, 128, max_eff_bits=4.0,
                u_bits=3, s_bits=4, v_bits=3,
                use_power_iter=True, power_iter_steps=4,
                use_warmstart=True,
                use_adaptive_rank=True,
                adaptive_energy_threshold=thr,
                max_adaptive_rank=mr,
            )
            elapsed = time.time() - t0
            mse = float(np.mean((W_f - W_q.astype(np.float32)) ** 2))
            ratio = mse / direct_mse
            
            status = "✅" if info['effective_bits'] <= 4.05 else "⚠️"
            print(f"{status} {thr:<8} {mr:<10} {info['rounds']:<6} {info['effective_bits']:<10.3f} "
                  f"{mse:<12.6f} {ratio:<8.3f} {elapsed:<10.2f}s")


def test_power_iter_steps():
    """测试不同幂迭代步数的影响"""
    print("\n" + "=" * 70)
    print("测试 4: 幂迭代步数 vs 精度 vs 速度")
    print("=" * 70)
    
    W = generate_weight_matrix(768, 768)
    W_f = W.astype(np.float32)
    direct_mse = float(np.mean((W_f - quantize_mse(W, 4, 128)) ** 2))
    
    print(f"\n{'步数':<6} {'轮数':<6} {'EffBits':<10} {'MSE':<12} {'Ratio':<8} {'时间':<10}")
    print("-" * 55)
    
    for n_steps in [1, 2, 3, 4, 5, 6, 8]:
        t0 = time.time()
        W_q, info = adaptive_iterative_svd(
            W, 128, max_eff_bits=4.0, rank=4, u_bits=3, s_bits=4, v_bits=3,
            use_power_iter=True, power_iter_steps=n_steps,
            use_warmstart=True,
        )
        elapsed = time.time() - t0
        mse = float(np.mean((W_f - W_q.astype(np.float32)) ** 2))
        ratio = mse / direct_mse
        
        print(f"{n_steps:<6} {info['rounds']:<6} {info['effective_bits']:<10.3f} "
              f"{mse:<12.6f} {ratio:<8.3f} {elapsed:<10.2f}s")


def test_scaling():
    """测试不同矩阵尺寸下的性能"""
    print("\n" + "=" * 70)
    print("测试 5: 矩阵尺寸 scaling (rank=4, u3s4v3, max_eff=4.0)")
    print("=" * 70)
    
    sizes = [
        ("128×128", 128, 128),
        ("256×256", 256, 256),
        ("512×512", 512, 512),
        ("768×768", 768, 768),
        ("1024×1024", 1024, 1024),
        ("512×2048", 512, 2048),
        ("2048×512", 2048, 512),
    ]
    
    print(f"\n{'尺寸':<15} {'标准SVD时间':<14} {'幂迭代时间':<14} {'加速比':<10} {'MSE差异':<12}")
    print("-" * 68)
    
    for name, rows, cols in sizes:
        W = generate_weight_matrix(rows, cols)
        W_f = W.astype(np.float32)
        
        # 标准 SVD
        t0 = time.time()
        W_q_full, info_full = iterative_residual_svd(
            W, 128, max_eff_bits=4.0, rank=4, u_bits=3, s_bits=4, v_bits=3,
        )
        t_full = time.time() - t0
        mse_full = float(np.mean((W_f - W_q_full.astype(np.float32)) ** 2))
        
        # 幂迭代
        t0 = time.time()
        W_q_pi, info_pi = adaptive_iterative_svd(
            W, 128, max_eff_bits=4.0, rank=4, u_bits=3, s_bits=4, v_bits=3,
            use_power_iter=True, power_iter_steps=4, use_warmstart=True,
        )
        t_pi = time.time() - t0
        mse_pi = float(np.mean((W_f - W_q_pi.astype(np.float32)) ** 2))
        
        speedup = t_full / max(t_pi, 1e-6)
        mse_diff = abs(mse_pi - mse_full) / mse_full * 100
        
        print(f"{name:<15} {t_full:<14.3f}s {t_pi:<14.3f}s {speedup:<10.1f}x {mse_diff:<12.2f}%")


def main():
    print("=" * 70)
    print("幂迭代 SVD 量化对比测试")
    print("=" * 70)
    
    test_power_iter_accuracy()
    test_iterative_comparison()
    test_adaptive_rank()
    test_power_iter_steps()
    test_scaling()
    
    print("\n" + "=" * 70)
    print("所有测试完成")
    print("=" * 70)


if __name__ == "__main__":
    main()

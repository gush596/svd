"""
NumPy 算法验证

用合成数据测试所有量化算法的 MSE 表现，不依赖 PyTorch 和模型下载。
快速验证算法正确性和参数调优。

用法:
    python tests/test_numpy.py              # 运行全部测试
    python tests/test_numpy.py --quick      # 快速测试（少量配置）
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
import json
import argparse
from quantization.core import quantize_mse
from quantization.svd_hybrid import svd_hybrid, compute_effective_bits
from quantization.important import important_protection


def generate_weight_matrix(rows: int, cols: int, seed: int = 42) -> np.ndarray:
    """生成模拟 Transformer 权重的测试矩阵
    
    特点：低秩结构 + 离群值 + 噪声
    """
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


def test_svd_hybrid():
    """测试 SVD Hybrid 算法"""
    print("\n" + "=" * 70)
    print("SVD Hybrid 测试")
    print("=" * 70)
    
    W = generate_weight_matrix(768, 768)
    W_f = W.astype(np.float32)
    direct_q = quantize_mse(W, 4, 128)
    direct_mse = float(np.mean((W_f - direct_q) ** 2))
    print(f"Direct 4-bit baseline MSE: {direct_mse:.6f}")
    
    configs = [
        # (energy_threshold, svd_bits, residual_bits, 描述)
        (0.80, 2, 4, "低rank+2bit SVD"),
        (0.80, 3, 4, "低rank+3bit SVD"),
        (0.85, 2, 4, "中rank+2bit SVD"),
        (0.85, 3, 4, "中rank+3bit SVD"),
        (0.90, 3, 4, "高rank+3bit SVD"),
    ]
    
    results = []
    for t, sb, rb, desc in configs:
        W_q, info = svd_hybrid(W, 4, 128, t, sb, rb)
        mse = float(np.mean((W_f - W_q.astype(np.float32)) ** 2))
        ratio = mse / direct_mse
        results.append({**info, 'desc': desc, 'mse_ratio': ratio})
        eff = info['effective_bits']
        status = "✅" if eff <= 4.05 else "⚠️"
        print(f"  {status} {desc:<20} rank={info['rank']:<4} eff={eff:.3f} "
              f"mse={mse:.6f} ratio={ratio:.3f}")
    
    return results


def test_important_protection():
    """测试重要值保护"""
    print("\n" + "=" * 70)
    print("重要值保护测试")
    print("=" * 70)
    
    W = generate_weight_matrix(768, 768)
    W_f = W.astype(np.float32)
    direct_mse = float(np.mean((W_f - quantize_mse(W, 4, 128)) ** 2))
    
    configs = [
        (0.05, 5), (0.10, 5), (0.15, 5),
        (0.10, 6), (0.15, 6), (0.20, 6),
    ]
    
    results = []
    for ratio, bits in configs:
        W_q, info = important_protection(W, 4, 128, ratio, bits)
        mse = float(np.mean((W_f - W_q.astype(np.float32)) ** 2))
        eff = info['effective_bits']
        ratio_mse = mse / direct_mse
        results.append({**info, 'mse_ratio': ratio_mse})
        status = "✅" if eff <= 4.05 else "⚠️"
        print(f"  {status} ratio={ratio:.2f} bits={bits} eff={eff:.2f} "
              f"mse={mse:.6f} ratio={ratio_mse:.3f}")
    
    return results


def test_effective_bits():
    """验证等效 bit 计算"""
    print("\n" + "=" * 70)
    print("等效 bit 计算验证")
    print("=" * 70)
    
    layers = [
        ('q/k/v', 768, 768),
        ('out_proj', 768, 768),
        ('fc1', 768, 3072),
        ('fc2', 3072, 768),
    ]
    
    print(f"\n{'Layer':<12} {'Shape':<12} {'r=4,sb=2':<10} {'r=4,sb=3':<10} "
          f"{'r=6,sb=2':<10} {'r=6,sb=3':<10}")
    print("-" * 70)
    
    for name, out, inp in layers:
        vals = []
        for rank, sb in [(4, 2), (4, 3), (6, 2), (6, 3)]:
            eff = compute_effective_bits(out, inp, rank, sb, 4)
            vals.append(f"{eff:.3f}")
        print(f"{name:<12} {out}x{inp:<8} {vals[0]:<10} {vals[1]:<10} {vals[2]:<10} {vals[3]:<10}")


def test_multi_size():
    """多矩阵尺寸测试"""
    print("\n" + "=" * 70)
    print("多矩阵尺寸测试")
    print("=" * 70)
    
    sizes = [("256x256", 256, 256), ("768x768", 768, 768), 
             ("512x2048", 512, 2048), ("2048x512", 2048, 512)]
    
    all_results = []
    for name, rows, cols in sizes:
        W = generate_weight_matrix(rows, cols)
        W_f = W.astype(np.float32)
        direct_mse = float(np.mean((W_f - quantize_mse(W, 4, 128)) ** 2))
        
        W_q, info = svd_hybrid(W, 4, 128, 0.80, 3, 4)
        mse = float(np.mean((W_f - W_q.astype(np.float32)) ** 2))
        
        print(f"  {name:<15} rank={info['rank']:<4} eff={info['effective_bits']:.3f} "
              f"mse_ratio={mse/direct_mse:.3f}")
        all_results.append({'size': name, **info, 'mse_ratio': mse/direct_mse})
    
    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='快速测试')
    args = parser.parse_args()
    
    print("=" * 70)
    print("SVD 量化算法 NumPy 验证")
    print("=" * 70)
    
    test_effective_bits()
    test_svd_hybrid()
    test_important_protection()
    test_multi_size()
    
    print("\n✅ 全部测试完成")


if __name__ == "__main__":
    main()

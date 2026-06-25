"""异常值 SVD + 残差量化 快速验证"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
from quantization.core import quantize_mse
from quantization.outlier_svd import outlier_svd_quantize


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
    print("  异常值 SVD + 残差量化 测试")
    print("=" * 80)

    W = generate_weight_matrix(768, 768)
    W_f = W.astype(np.float32)
    direct_mse = float(np.mean((W_f - quantize_mse(W, 4, 128)) ** 2))
    print(f"  Direct 4-bit MSE: {direct_mse:.6f}")
    print()

    # ── 不同 outlier_ratio ──
    print(f"{'outlier_ratio':<15} {'svd_rounds':<12} {'opt_eff_raw':<14} {'opt_eff_full':<14} "
          f"{'MSE':<12} {'vs Direct':<10} {'n_outliers%':<12}")
    print("-" * 95)

    for ratio in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        W_q, info = outlier_svd_quantize(
            W, outlier_ratio=ratio, svd_eff_bits=1.0,
            residual_bits=3, svd_rank=4, svd_u_bits=3, svd_v_bits=3,
            svd_s_bits=16, group_size=128,
        )
        mse = float(np.mean((W_f - W_q.astype(np.float32)) ** 2))
        print(f"{ratio:<15.2f} {info['svd_rounds']:<12} {info['optimized_eff_raw']:<14.4f} "
              f"{info['optimized_eff_full']:<14.4f} {mse:<12.6f} {mse/direct_mse:<10.4f} "
              f"{info['outlier_pct_actual']*100:<12.1f}")

    # ── 不同 residual_bits ──
    print(f"\n{'res_bits':<10} {'opt_eff_raw':<14} {'opt_eff_full':<14} "
          f"{'MSE':<12} {'vs Direct':<10}")
    print("-" * 55)

    for rb in [2, 3, 4, 5]:
        W_q, info = outlier_svd_quantize(
            W, outlier_ratio=0.15, svd_eff_bits=1.0,
            residual_bits=rb, svd_rank=4, svd_u_bits=3, svd_v_bits=3,
            svd_s_bits=16, group_size=128,
        )
        mse = float(np.mean((W_f - W_q.astype(np.float32)) ** 2))
        print(f"{rb:<10} {info['optimized_eff_raw']:<14.4f} {info['optimized_eff_full']:<14.4f} "
              f"{mse:<12.6f} {mse/direct_mse:<10.4f}")

    # ── 不同 svd_eff_bits ──
    print(f"\n{'svd_eff':<10} {'svd_rounds':<12} {'opt_eff_raw':<14} {'opt_eff_full':<14} "
          f"{'MSE':<12} {'vs Direct':<10}")
    print("-" * 70)

    for se in [0.5, 0.75, 1.0, 1.5, 2.0]:
        W_q, info = outlier_svd_quantize(
            W, outlier_ratio=0.15, svd_eff_bits=se,
            residual_bits=3, svd_rank=4, svd_u_bits=3, svd_v_bits=3,
            svd_s_bits=16, group_size=128,
        )
        mse = float(np.mean((W_f - W_q.astype(np.float32)) ** 2))
        print(f"{se:<10.2f} {info['svd_rounds']:<12} {info['optimized_eff_raw']:<14.4f} "
              f"{info['optimized_eff_full']:<14.4f} {mse:<12.6f} {mse/direct_mse:<10.4f}")

    print(f"\n  ✅ 测试完成")


if __name__ == "__main__":
    main()

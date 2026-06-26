"""
迭代异常值 SVD + 残差量化

算法：
1. 对当前残差 R，筛选 top-k% 异常值（按绝对值）
2. 构造稀疏异常值矩阵，做 SVD，取 rank，量化 U/V（S 不量化或 fp16）
3. 更新残差 R = R - component
4. 重复 1-3，直到 SVD 部分的 eff_raw 达到目标上限
5. 最终残差做 n-bit 直接量化

W_approx = sum(W_outlier_svd_i) + W_residual_quantized
"""

import numpy as np
import math
from typing import Tuple, Dict, Optional
from .core import quantize_mse


def _svd_decompose(matrix):
    """SVD 分解，带不收敛保护"""
    try:
        U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
        return U, S, Vt
    except np.linalg.LinAlgError:
        # SVD 不收敛时，用更小的矩阵重试或跳过
        # 添加微小噪声后重试
        noisy = matrix + np.random.randn(*matrix.shape).astype(matrix.dtype) * 1e-10
        return np.linalg.svd(noisy, full_matrices=False)


def iterative_outlier_svd(
    W: 'np.ndarray',
    outlier_ratio: float = 0.10,
    max_svd_eff: float = 1.0,
    residual_bits: int = 3,
    rank: int = 4,
    u_bits: int = 3,
    v_bits: int = 3,
    s_bits: Optional[int] = 16,
    group_size: int = 128,
) -> Tuple[np.ndarray, Dict]:
    """迭代异常值 SVD + 残差量化

    Args:
        W: 权重矩阵 [out, in]
        outlier_ratio: 每轮提取的异常值比例
        max_svd_eff: SVD 部分的 eff_raw 上限（不含残差）
        residual_bits: 最终残差量化 bit 数
        rank: 每轮 SVD 的 rank
        u_bits: U 矩量化的 bit 数
        v_bits: V 矩量化的 bit 数
        s_bits: S 值存储方式 (16=fp16, None=fp32, 2/3/4=量化)
        group_size: 量化 group_size

    Returns:
        (W_approx, info)
    """
    out_dim, in_dim = W.shape
    total_params = W.size
    W_f = W.astype(np.float32)

    # s_bits 处理
    s_quant = s_bits is not None and s_bits < 16
    s_use_fp16 = (s_bits == 16)
    s_bits_eff = s_bits if s_bits is not None else 32

    # 每轮 eff_raw 增量
    gs_u = min(group_size, max(8, rank))
    gs_v = min(group_size, max(8, rank))
    round_eff_raw = rank * (out_dim * u_bits + in_dim * v_bits + rank * s_bits_eff) / total_params

    # 最大轮数
    max_rounds = int(max_svd_eff / round_eff_raw) if round_eff_raw > 0 else 0

    if max_rounds <= 0:
        # SVD 预算不足，直接量化
        W_q = quantize_mse(W_f, n_bits=residual_bits, group_size=group_size)
        return W_q, {
            'rounds': 0,
            'svd_eff_raw': 0.0,
            'residual_bits': residual_bits,
            'total_eff_raw': float(residual_bits),
            'mse': float(np.mean((W_f - W_q) ** 2)),
            'fallback': 'direct_quant',
        }

    # 迭代异常值 SVD
    residual = W_f.copy()
    W_svd_approx = np.zeros_like(W_f)
    round_infos = []

    for i in range(max_rounds):
        # 1. 从当前残差中筛选 top-k% 异常值
        R_flat = residual.reshape(-1)
        n_outliers = max(1, int(total_params * outlier_ratio))
        threshold_idx = np.argsort(np.abs(R_flat))[-n_outliers]
        threshold_val = np.abs(R_flat[threshold_idx])
        outlier_mask = np.abs(residual) >= threshold_val

        # 2. 构造稀疏异常值矩阵
        R_outlier = np.where(outlier_mask, residual, 0.0)

        # 3. SVD 分解
        U, S, Vt = _svd_decompose(R_outlier)
        actual_rank = min(rank, len(S))

        U_k = U[:, :actual_rank]
        S_k = S[:actual_rank]
        V_k = Vt[:actual_rank, :]

        # 4. 量化 U, S, V
        U_q = quantize_mse(U_k, n_bits=u_bits, group_size=gs_u)

        if s_quant:
            gs_s = min(group_size, max(8, actual_rank))
            S_q = quantize_mse(S_k.reshape(1, -1), n_bits=s_bits, group_size=gs_s).reshape(-1)
        elif s_use_fp16:
            S_q = S_k.astype(np.float16).astype(np.float32)
        else:
            S_q = S_k

        V_q = quantize_mse(V_k, n_bits=v_bits, group_size=gs_v)

        # 5. 重建本轮分量
        component = U_q @ np.diag(S_q) @ V_q
        W_svd_approx = W_svd_approx + component
        residual = residual - component

        n_actual_outliers = int(outlier_mask.sum())
        round_infos.append({
            'round': i + 1,
            'rank': actual_rank,
            'n_outliers': n_actual_outliers,
            'outlier_pct': n_actual_outliers / total_params,
            'threshold': float(threshold_val),
            'residual_norm': float(np.linalg.norm(residual)),
            'residual_mse': float(np.mean(residual ** 2)),
        })

    # 最终残差做 n-bit 量化
    W_residual_q = quantize_mse(residual, n_bits=residual_bits, group_size=group_size)

    # 合并
    W_approx = W_svd_approx + W_residual_q

    # 计算等效 bit
    svd_eff_raw = len(round_infos) * round_eff_raw
    # 残差 eff: residual_bits (不含 scale，与直接量化对比)
    total_eff_raw = svd_eff_raw + residual_bits

    mse = float(np.mean((W_f - W_approx) ** 2))
    mse_svd_only = float(np.mean((W_f - W_svd_approx) ** 2))
    mse_residual = float(np.mean((residual - W_residual_q) ** 2))

    info = {
        'rounds': len(round_infos),
        'round_details': round_infos,
        'outlier_ratio': outlier_ratio,
        'svd_eff_raw': float(svd_eff_raw),
        'residual_bits': residual_bits,
        'total_eff_raw': float(total_eff_raw),
        'mse': mse,
        'mse_svd_only': mse_svd_only,
        'mse_residual_quant': mse_residual,
        'direct_4bit_mse': float(np.mean((W_f - quantize_mse(W_f, 4, group_size)) ** 2)),
        'direct_3bit_mse': float(np.mean((W_f - quantize_mse(W_f, 3, group_size)) ** 2)),
    }

    return W_approx, info

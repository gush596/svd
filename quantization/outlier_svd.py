"""
异常值 SVD + 残差直接量化

算法：
1. 从 W 中筛选 top-k% 异常值（按绝对值）
2. 构造异常值矩阵 W_outlier（异常值保留原位，其余为 0）
3. 对 W_outlier 做迭代残差 SVD（u=3, v=3, s-fp16, eff_raw ≤ target_eff）
4. 残差 = W - W_outlier_approx
5. 对残差做 n-bit 直接量化（默认 3-bit）

最终: W_approx = W_outlier_svd + W_residual_quantized
"""

import numpy as np
import math
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False
from typing import Tuple, Dict, Optional
from .core import quantize_mse
from .iterative_svd import iterative_residual_svd


def outlier_svd_quantize(
    W: 'torch.Tensor | np.ndarray',
    outlier_ratio: float = 0.15,
    svd_eff_bits: float = 1.0,
    residual_bits: int = 3,
    svd_rank: int = 4,
    svd_u_bits: int = 3,
    svd_v_bits: int = 3,
    svd_s_bits: Optional[int] = 16,
    group_size: int = 128,
) -> Tuple['torch.Tensor | np.ndarray', Dict]:
    """异常值 SVD + 残差直接量化

    Args:
        W: 权重矩阵 [out, in]
        outlier_ratio: 异常值比例 (0.0-1.0)
        svd_eff_bits: 异常值 SVD 的 eff_raw 上限
        residual_bits: 残差量化 bit 数
        svd_rank: SVD 每轮 rank
        svd_u_bits: SVD U 矩 bit
        svd_v_bits: SVD V 矩 bit
        svd_s_bits: SVD S 值 bit (16=fp16, None=fp32)
        group_size: 量化 group_size

    Returns:
        (W_approx, info)
    """
    is_torch = HAS_TORCH and isinstance(W, torch.Tensor)
    if is_torch:
        device, dtype = W.device, W.dtype
        W_np = W.float().cpu().numpy()
    else:
        W_np = W.astype(np.float32)

    result, info = _outlier_svd_numpy(
        W_np, outlier_ratio, svd_eff_bits, residual_bits,
        svd_rank, svd_u_bits, svd_v_bits, svd_s_bits, group_size,
    )

    if is_torch:
        return torch.from_numpy(result).to(device=device, dtype=dtype), info
    return result, info


def _outlier_svd_numpy(
    W: np.ndarray,
    outlier_ratio: float,
    svd_eff_bits: float,
    residual_bits: int,
    svd_rank: int,
    svd_u_bits: int,
    svd_v_bits: int,
    svd_s_bits: Optional[int],
    group_size: int,
) -> Tuple[np.ndarray, Dict]:
    """核心实现"""
    out_dim, in_dim = W.shape
    total_params = W.size
    W_f = W.astype(np.float32)

    # ── Step 1: 筛选异常值 ──
    W_flat = W_f.reshape(-1)
    n_outliers = max(1, int(total_params * outlier_ratio))
    threshold_idx = np.argsort(np.abs(W_flat))[-n_outliers]
    threshold_val = np.abs(W_flat[threshold_idx])

    outlier_mask = np.abs(W_f) >= threshold_val  # [out, in], bool
    n_actual = int(outlier_mask.sum())

    # ── Step 2: 构造异常值矩阵 ──
    W_outlier = np.where(outlier_mask, W_f, 0.0)

    # ── Step 3: 对异常值矩阵做迭代残差 SVD ──
    W_outlier_approx, svd_info = iterative_residual_svd(
        W_outlier, group_size=group_size, max_eff_bits=svd_eff_bits,
        rank=svd_rank, u_bits=svd_u_bits, s_bits=svd_s_bits, v_bits=svd_v_bits,
    )
    if isinstance(W_outlier_approx, np.ndarray) is False:
        W_outlier_approx = np.array(W_outlier_approx, dtype=np.float32)

    # ── Step 4: 计算残差 ──
    residual = W_f - W_outlier_approx

    # ── Step 5: 对残差做 n-bit 直接量化 ──
    W_residual_q = quantize_mse(residual, n_bits=residual_bits, group_size=group_size)

    # ── Step 6: 合并 ──
    W_approx = W_outlier_approx + W_residual_q

    # ── 计算综合等效 bit ──
    #
    # 存储内容:
    #   1. SVD: U[out,rank], S[rank], V[rank,in] × n_rounds (覆盖全矩阵)
    #   2. 残差量化: residual_bits per param × ALL 参数 (残差在所有位置非零)
    #
    # ⚠️ 不能省略 outlier 位置的残差！
    #    residual = W - W_outlier_approx, outlier 位置的残差 = 原始异常值 - SVD近似 ≠ 0
    #
    svd_eff_raw = svd_info['effective_bits']
    svd_eff_full = svd_info.get('effective_bits_full', svd_eff_raw)

    # 残差部分的 eff: 所有参数用 residual_bits，加上 scale 开销
    # quantize_mse 对全矩阵做 group 量化，每 group 一个 float32 scale
    n_groups_residual = math.ceil(total_params / group_size)
    residual_bits_total = total_params * residual_bits + n_groups_residual * 32
    residual_eff_full = residual_bits_total / total_params

    # 正确的总等效 bit: SVD + 残差 (全矩阵)
    total_eff_raw = svd_eff_raw + residual_bits
    total_eff_full = svd_eff_full + residual_eff_full

    mse = float(np.mean((W_f - W_approx) ** 2))
    mse_svd_only = float(np.mean((W_f - W_outlier_approx) ** 2))
    mse_residual = float(np.mean((residual - W_residual_q) ** 2))

    info = {
        'outlier_ratio': outlier_ratio,
        'outlier_threshold': float(threshold_val),
        'n_outliers': n_actual,
        'outlier_pct_actual': n_actual / total_params,
        'svd_rounds': svd_info['rounds'],
        'svd_eff_raw': float(svd_eff_raw),
        'svd_eff_full': float(svd_eff_full),
        'residual_bits': residual_bits,
        'residual_eff_raw': float(residual_bits),
        'residual_eff_full': float(residual_eff_full),
        'total_eff_raw': float(total_eff_raw),
        'total_eff_full': float(total_eff_full),
        'mse': mse,
        'mse_svd_only': mse_svd_only,
        'mse_residual_quant': mse_residual,
        'direct_4bit_mse': float(np.mean((W_f - quantize_mse(W_f, 4, group_size)) ** 2)),
    }

    return W_approx, info

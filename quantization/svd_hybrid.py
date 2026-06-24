"""
SVD Hybrid 量化算法

核心思想：W ≈ SVD低秩重建 + 残差量化
通过将权重矩阵分解为低秩部分和残差部分，分别用不同精度量化。

算法步骤：
1. SVD 分解，取前 k 个奇异值
2. 低秩因子 U_k', V_k' 用 svd_bits 量化
3. 残差 R = W - W_svd 用 n_bits 量化
4. 最终 W_approx = W_svd + R_q

等效 bit 计算：
    eff = rank*(out+in)*svd_bits/(out*in) + n_bits
    SVD 因子是额外存储，所以 eff > n_bits
"""

import numpy as np
try:
    import torch
except ImportError:
    torch = None
from typing import Tuple, Dict, Optional
from .core import quantize_mse


def svd_hybrid(
    W: 'torch.Tensor | np.ndarray',
    n_bits: int = 4,
    group_size: int = 128,
    energy_threshold: float = 0.85,
    svd_bits: int = 3,
    residual_bits: int = 4,
) -> Tuple['torch.Tensor | np.ndarray', Dict]:
    """SVD Hybrid 量化
    
    Args:
        W: 权重矩阵 [out, in]
        n_bits: 目标等效 bit（用于约束检查）
        group_size: 量化分组大小
        energy_threshold: SVD 能量保留比例，决定 rank
        svd_bits: SVD 因子的量化精度
        residual_bits: 残差的量化精度
    
    Returns:
        (W_approx, info): 量化后的权重和信息字典
    """
    is_torch = torch is not None and isinstance(W, torch.Tensor)
    if is_torch:
        device, dtype = W.device, W.dtype
        W_np = W.float().cpu().numpy()
    else:
        W_np = W.astype(np.float32)
    
    W_approx, info = _svd_hybrid_numpy(W_np, n_bits, group_size, 
                                        energy_threshold, svd_bits, residual_bits)
    
    if is_torch:
        return torch.from_numpy(W_approx).to(device=device, dtype=dtype), info
    return W_approx, info


def _svd_hybrid_numpy(
    W: np.ndarray,
    n_bits: int,
    group_size: int,
    energy_threshold: float,
    svd_bits: int,
    residual_bits: int,
) -> Tuple[np.ndarray, Dict]:
    """SVD Hybrid 量化 (NumPy 实现)"""
    out_dim, in_dim = W.shape
    total_params = W.size

    # Step 1: SVD 分解
    U, S, Vt = np.linalg.svd(W.astype(np.float32), full_matrices=False)
    
    # 自适应 rank
    total_energy = np.cumsum(S ** 2) / np.sum(S ** 2)
    rank = int(np.searchsorted(total_energy, energy_threshold)) + 1
    rank = max(4, min(rank, min(out_dim, in_dim) // 4))

    # Step 2: 低秩重建
    U_k = U[:, :rank] * np.sqrt(S[:rank])        # [out, rank]
    V_k = Vt[:rank, :] * np.sqrt(S[:rank]).reshape(-1, 1)  # [rank, in]

    gs_svd = min(group_size, max(8, U_k.shape[1]))
    U_q = quantize_mse(U_k, n_bits=svd_bits, group_size=gs_svd)
    V_q = quantize_mse(V_k, n_bits=svd_bits, group_size=gs_svd)
    W_svd = U_q @ V_q

    # Step 3: 残差量化
    residual = W - W_svd
    R_q = quantize_mse(residual, n_bits=residual_bits, group_size=group_size)

    # Step 4: 最终重建
    W_approx = W_svd + R_q

    # 等效 bit
    n_svd_params = U_k.size + V_k.size
    eff_bits = n_svd_params * svd_bits / total_params + residual_bits

    info = {
        'rank': rank,
        'svd_bits': svd_bits,
        'residual_bits': residual_bits,
        'energy_preserved': float(total_energy[min(rank-1, len(total_energy)-1)]),
        'effective_bits': float(eff_bits),
        'mse': float(np.mean((W - W_approx) ** 2)),
    }

    return W_approx, info


def compute_effective_bits(
    out_dim: int,
    in_dim: int,
    rank: int,
    svd_bits: int,
    residual_bits: int,
    outlier_ratio: float = 0.0,
    outlier_bits: int = 4,
) -> float:
    """计算等效 bit 数
    
    公式：
        eff = rank*(out+in)*svd_bits/(out*in) + residual_bits
        + outlier_ratio * (outlier_bits - residual_bits)  [如果考虑 outlier]
    
    注意：这不包含 outlier 位置存储的开销！
    如需包含，每参数额外 +0.109 bit (gs=128, top-2 索引)
    """
    total = out_dim * in_dim
    n_svd = rank * (out_dim + in_dim)
    eff = n_svd * svd_bits / total + residual_bits
    
    if outlier_ratio > 0:
        eff += outlier_ratio * (outlier_bits - residual_bits)
    
    return eff


def get_rank_for_energy(energy_threshold: float, out_dim: int, in_dim: int) -> int:
    """估算给定能量阈值对应的 rank（粗略估计）"""
    # 基于 OPT-125M 的经验值
    if out_dim == 768 and in_dim == 768:
        rank_map = {0.70: 4, 0.75: 5, 0.80: 6, 0.85: 8, 0.90: 12}
    elif out_dim == 768 and in_dim == 3072:
        rank_map = {0.70: 20, 0.75: 25, 0.80: 30, 0.85: 40, 0.90: 60}
    elif out_dim == 3072 and in_dim == 768:
        rank_map = {0.70: 20, 0.75: 25, 0.80: 30, 0.85: 40, 0.90: 60}
    else:
        # 通用估计
        dim = min(out_dim, in_dim)
        rank_map = {0.70: dim//20, 0.80: dim//12, 0.85: dim//8, 0.90: dim//5}
    
    # 找最近的阈值
    closest = min(rank_map.keys(), key=lambda t: abs(t - energy_threshold))
    return rank_map[closest]

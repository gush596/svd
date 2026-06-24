"""
迭代残差 SVD 量化

算法：
1. 对 W 做 SVD，保留 top-k 个奇异值
2. 用指定 bit 分别量化 U, S, V
3. 计算残差 R = W - U_q @ diag(S_q) @ V_q
4. 对 R 重复步骤 1-3，直到达到等效 bit 预算
5. 最终残差**舍弃**，不量化

每轮迭代的 rank、u_bits、s_bits、v_bits 保持不变。

等效 bit 公式：
    eff = n_rounds × rank × (out × u_bits + s_bits + in × v_bits) / (out × in)
"""

import numpy as np
try:
    import torch
except ImportError:
    torch = None
from typing import Tuple, Dict, List, Optional
from .core import quantize_mse


def iterative_residual_svd(
    W: 'torch.Tensor | np.ndarray',
    group_size: int = 128,
    max_eff_bits: float = 4.0,
    rank: int = 1,
    u_bits: int = 3,
    s_bits: int = 4,
    v_bits: int = 3,
    n_rounds: Optional[int] = None,
) -> Tuple['torch.Tensor | np.ndarray', Dict]:
    """迭代残差 SVD 量化
    
    Args:
        W: 权重矩阵 [out, in]
        group_size: 量化分组大小
        max_eff_bits: 最大等效 bit（当 n_rounds 为 None 时用于自动计算轮数）
        rank: 每轮保留的奇异值数量（固定不变）
        u_bits: U 矩量化的 bit 数（固定不变）
        s_bits: S 奇异值量化的 bit 数（固定不变）
        v_bits: V 矩量化的 bit 数（固定不变）
        n_rounds: 显式指定轮数（如果指定，忽略 max_eff_bits）
    
    Returns:
        (W_approx, info)
    """
    is_torch = torch is not None and isinstance(W, torch.Tensor)
    if is_torch:
        device, dtype = W.device, W.dtype
        W_np = W.float().cpu().numpy()
    else:
        W_np = W.astype(np.float32)
    
    result, info = _iterative_residual_svd_numpy(
        W_np, group_size, max_eff_bits, rank, u_bits, s_bits, v_bits, n_rounds
    )
    
    if is_torch:
        return torch.from_numpy(result).to(device=device, dtype=dtype), info
    return result, info


def _iterative_residual_svd_numpy(
    W: np.ndarray,
    group_size: int,
    max_eff_bits: float,
    rank: int,
    u_bits: int,
    s_bits: int,
    v_bits: int,
    n_rounds_override: Optional[int],
) -> Tuple[np.ndarray, Dict]:
    """迭代残差 SVD 量化 (NumPy 实现)
    
    算法：
    - 每轮对当前残差做 SVD，取 top-rank 个奇异值
    - 分别量化 U, S, V，重建本轮分量
    - 残差 = 上一轮残差 - 本轮分量
    - 最终残差舍弃（不量化）
    """
    out_dim, in_dim = W.shape
    total_params = W.size
    W_f = W.astype(np.float32)
    
    # 计算每轮的等效 bit 增量
    round_bits = rank * (out_dim * u_bits + s_bits + in_dim * v_bits)
    round_eff = round_bits / total_params
    
    # 确定轮数
    if n_rounds_override is not None:
        n_rounds = n_rounds_override
    else:
        # 根据 max_eff_bits 自动计算最大轮数
        n_rounds = int(max_eff_bits / round_eff)
    
    if n_rounds <= 0:
        # 连一轮都跑不了
        W_q = quantize_mse(W_f, n_bits=4, group_size=group_size)
        return W_q, {
            'rounds': 0,
            'effective_bits': 4.0,
            'mse': float(np.mean((W_f - W_q) ** 2)),
            'fallback': 'direct_quant',
        }
    
    # 执行迭代
    residual = W_f.copy()
    W_approx = np.zeros_like(W_f)
    round_infos = []
    
    for i in range(n_rounds):
        # SVD 分解当前残差
        U, S, Vt = np.linalg.svd(residual, full_matrices=False)
        
        # 限制 rank
        actual_rank = min(rank, len(S))
        
        # 提取 top-k
        U_k = U[:, :actual_rank]         # [out, rank]
        S_k = S[:actual_rank]            # [rank]
        V_k = Vt[:actual_rank, :]        # [rank, in]
        
        # 分别量化 U, S, V
        gs_u = min(group_size, max(8, actual_rank))
        U_q = quantize_mse(U_k, n_bits=u_bits, group_size=gs_u)
        
        gs_s = min(group_size, max(8, actual_rank))
        S_q = quantize_mse(S_k.reshape(1, -1), n_bits=s_bits, group_size=gs_s).reshape(-1)
        
        gs_v = min(group_size, max(8, actual_rank))
        V_q = quantize_mse(V_k, n_bits=v_bits, group_size=gs_v)
        
        # 重建本轮分量
        component = U_q @ np.diag(S_q) @ V_q
        W_approx = W_approx + component
        
        # 更新残差
        residual = residual - component
        
        # 记录信息
        round_info = {
            'round': i + 1,
            'rank': actual_rank,
            'u_bits': u_bits,
            's_bits': s_bits,
            'v_bits': v_bits,
            'round_bits': actual_rank * (out_dim * u_bits + s_bits + in_dim * v_bits),
            'residual_norm': float(np.linalg.norm(residual)),
            'residual_mse': float(np.mean(residual ** 2)),
        }
        round_infos.append(round_info)
    
    # 最终残差舍弃，不量化
    # W_approx 就是最终结果
    
    # 计算总等效 bit（不含残差）
    svd_bits_total = sum(r['round_bits'] for r in round_infos)
    eff_bits = svd_bits_total / total_params
    
    info = {
        'rounds': len(round_infos),
        'round_details': round_infos,
        'effective_bits': float(eff_bits),
        'mse': float(np.mean((W_f - W_approx) ** 2)),
        'residual_discarded': True,
        'residual_mse': float(np.mean(residual ** 2)),
    }
    
    return W_approx, info


def compute_max_rounds(
    out_dim: int,
    in_dim: int,
    max_eff_bits: float,
    rank: int = 1,
    u_bits: int = 3,
    s_bits: int = 4,
    v_bits: int = 3,
) -> int:
    """计算给定预算下最大可执行轮数
    
    公式:
        n_rounds = floor(max_eff_bits × out × in / (rank × (out × u_bits + s_bits + in × v_bits)))
    """
    total_params = out_dim * in_dim
    round_bits = rank * (out_dim * u_bits + s_bits + in_dim * v_bits)
    return int(max_eff_bits * total_params / round_bits)

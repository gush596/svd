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

综合等效 bit 公式（含 scale/zero-point 存储开销）：
    每轮每个量化矩阵的 group 数 = ceil(元素数 / group_size)
    每个 group 需要 1 个 float32 scale（32 bit）
    eff_full = n_rounds × [rank × (out × u_bits + in × v_bits) + s_bits]
               + n_rounds × n_groups_per_round × 32] / (out × in)
"""

import numpy as np
import math
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False
from typing import Tuple, Dict, List, Optional
from .core import quantize_mse


def _get_device():
    """自动检测最优设备：GPU 优先，否则 CPU"""
    if HAS_TORCH and torch.cuda.is_available():
        return torch.device('cuda')
    return None  # 使用 numpy (CPU)


def _svd_decompose(matrix, device=None):
    """SVD 分解，自动选择 GPU/CPU"""
    if device is not None and device.type == 'cuda':
        t = torch.from_numpy(matrix).to(device)
        U, S, Vt = torch.linalg.svd(t, full_matrices=False)
        return U.cpu().numpy(), S.cpu().numpy(), Vt.cpu().numpy()
    else:
        return np.linalg.svd(matrix, full_matrices=False)


def iterative_residual_svd(
    W: 'torch.Tensor | np.ndarray',
    group_size: int = 128,
    max_eff_bits: float = 4.0,
    rank: int = 4,
    u_bits: int = 4,
    s_bits: Optional[int] = None,
    v_bits: int = 4,
    n_rounds: Optional[int] = None,
    asymmetric: bool = False,
) -> Tuple['torch.Tensor | np.ndarray', Dict]:
    """迭代残差 SVD 量化
    
    Args:
        W: 权重矩阵 [out, in]
        group_size: 量化分组大小
        max_eff_bits: 最大等效 bit（当 n_rounds 为 None 时用于自动计算轮数）
        rank: 每轮保留的奇异值数量（固定不变）
        u_bits: U 矩量化的 bit 数
        s_bits: S 奇异值量化的 bit 数，None 表示不量化（float32）
        v_bits: V 矩量化的 bit 数
        n_rounds: 显式指定轮数（如果指定，忽略 max_eff_bits）
    
    Returns:
        (W_approx, info)
    """
    is_torch = HAS_TORCH and isinstance(W, torch.Tensor)
    if is_torch:
        device, dtype = W.device, W.dtype
        W_np = W.float().cpu().numpy()
    else:
        W_np = W.astype(np.float32)
    
    # 自动选择 SVD 设备
    svd_device = _get_device()
    
    result, info = _iterative_residual_svd_numpy(
        W_np, group_size, max_eff_bits, rank, u_bits, s_bits, v_bits,
        n_rounds, svd_device, asymmetric
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
    s_bits: Optional[int],
    v_bits: int,
    n_rounds_override: Optional[int],
    svd_device=None,
    asymmetric: bool = False,
) -> Tuple[np.ndarray, Dict]:
    """迭代残差 SVD 量化 (NumPy 实现)"""
    out_dim, in_dim = W.shape
    total_params = W.size
    W_f = W.astype(np.float32)
    
    # s_bits 处理:
    #   None → 不量化, fp32 存储 (32 bit)
    #   16   → 不量化, fp16 存储 (16 bit)
    #   2/3/4 → 量化到指定位数
    s_quant = s_bits is not None and s_bits < 16
    s_use_fp16 = (s_bits == 16)
    s_bits_eff = s_bits if s_bits is not None else 32  # None=fp32, 16=fp16, 2/3/4=quantized
    
    # 计算每轮的等效 bit 增量（不含 scale 开销）
    round_bits_raw = rank * (out_dim * u_bits + in_dim * v_bits) + rank * s_bits_eff
    round_eff_raw = round_bits_raw / total_params
    
    # 计算含 scale 的综合等效 bit
    # U: [out, rank] → ceil(out * rank / group_size) 个 scale
    # S: [rank] → 1 个 scale（如果量化）
    # V: [rank, in] → ceil(rank * in / group_size) 个 scale
    gs_u = min(group_size, max(8, rank))
    gs_v = min(group_size, max(8, rank))
    n_groups_u = math.ceil(out_dim * rank / gs_u)
    n_groups_v = math.ceil(rank * in_dim / gs_v)
    n_groups_s = 1 if s_quant else 0  # fp16/fp32 都不需要 scale
    scale_bits_per_round = (n_groups_u + n_groups_s + n_groups_v) * 32
    round_bits_full = round_bits_raw + scale_bits_per_round
    round_eff_full = round_bits_full / total_params
    
    # 确定轮数
    if n_rounds_override is not None:
        n_rounds = n_rounds_override
    else:
        # 用综合 eff 计算最大轮数（更保守）
        n_rounds = int(max_eff_bits / round_eff_full)
    
    if n_rounds <= 0:
        W_q = quantize_mse(W_f, n_bits=4, group_size=group_size, asymmetric=asymmetric)
        return W_q, {
            'rounds': 0, 'effective_bits': 4.0,
            'effective_bits_full': 4.0,
            'mse': float(np.mean((W_f - W_q) ** 2)),
            'fallback': 'direct_quant',
        }
    
    # 执行迭代
    residual = W_f.copy()
    W_approx = np.zeros_like(W_f)
    round_infos = []
    
    device_tag = 'cuda' if (svd_device and svd_device.type == 'cuda') else 'cpu'
    
    for i in range(n_rounds):
        # SVD 分解当前残差
        U, S, Vt = _svd_decompose(residual, svd_device)
        
        actual_rank = min(rank, len(S))
        
        U_k = U[:, :actual_rank]         # [out, rank]
        S_k = S[:actual_rank]            # [rank]
        V_k = Vt[:actual_rank, :]        # [rank, in]
        
        # 量化 U
        gs_u = min(group_size, max(8, actual_rank))
        U_q = quantize_mse(U_k, n_bits=u_bits, group_size=gs_u, asymmetric=asymmetric)
        
        # 量化 S
        if s_quant:
            gs_s = min(group_size, max(8, actual_rank))
            S_q = quantize_mse(S_k.reshape(1, -1), n_bits=s_bits, group_size=gs_s, asymmetric=asymmetric).reshape(-1)
        elif s_use_fp16:
            S_q = S_k.astype(np.float16).astype(np.float32)  # fp16 截断，计算用 fp32
        else:
            S_q = S_k  # fp32
        
        # 量化 V
        gs_v = min(group_size, max(8, actual_rank))
        V_q = quantize_mse(V_k, n_bits=v_bits, group_size=gs_v, asymmetric=asymmetric)
        
        # 重建本轮分量
        component = U_q @ np.diag(S_q) @ V_q
        W_approx = W_approx + component
        residual = residual - component
        
        round_infos.append({
            'round': i + 1,
            'rank': actual_rank,
            'u_bits': u_bits,
            's_bits': s_bits_eff,
            'v_bits': v_bits,
            'round_bits_raw': actual_rank * (out_dim * u_bits + in_dim * v_bits) + actual_rank * s_bits_eff,
            'scale_bits': scale_bits_per_round,
            'residual_norm': float(np.linalg.norm(residual)),
            'residual_mse': float(np.mean(residual ** 2)),
        })
    
    # 最终残差舍弃
    svd_bits_total = sum(r['round_bits_raw'] for r in round_infos)
    scale_bits_total = sum(r['scale_bits'] for r in round_infos)
    
    eff_bits_raw = svd_bits_total / total_params
    eff_bits_full = (svd_bits_total + scale_bits_total) / total_params
    
    info = {
        'rounds': len(round_infos),
        'round_details': round_infos,
        'effective_bits': float(eff_bits_raw),           # 不含 scale
        'effective_bits_full': float(eff_bits_full),     # 含 scale/zero-point
        'scale_bits_per_round': scale_bits_per_round,
        'mse': float(np.mean((W_f - W_approx) ** 2)),
        'residual_discarded': True,
        'residual_mse': float(np.mean(residual ** 2)),
        'svd_device': device_tag,
    }
    
    return W_approx, info


def compute_max_rounds(
    out_dim: int,
    in_dim: int,
    max_eff_bits: float,
    rank: int = 4,
    u_bits: int = 4,
    s_bits: Optional[int] = None,
    v_bits: int = 4,
    group_size: int = 128,
    use_full_eff: bool = True,
) -> int:
    """计算给定预算下最大可执行轮数
    
    Args:
        use_full_eff: True 用综合 eff（含 scale），更保守；False 用原始 eff
    """
    total_params = out_dim * in_dim
    s_bits_eff = s_bits if s_bits is not None else 32
    
    round_bits_raw = rank * (out_dim * u_bits + in_dim * v_bits) + rank * s_bits_eff
    
    if use_full_eff:
        gs_u = min(group_size, max(8, rank))
        gs_v = min(group_size, max(8, rank))
        n_groups_u = math.ceil(out_dim * rank / gs_u)
        n_groups_v = math.ceil(rank * in_dim / gs_v)
        n_groups_s = 1 if s_bits is not None else 0
        scale_bits = (n_groups_u + n_groups_s + n_groups_v) * 32
        round_bits = round_bits_raw + scale_bits
    else:
        round_bits = round_bits_raw
    
    return int(max_eff_bits * total_params / round_bits)

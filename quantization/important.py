"""
重要值保护量化

核心思想：识别绝对值最大的 top-k% 权重，用更高精度量化，
其余用标准精度。比 SVD 更直接地保护离群值。

等效 bit 计算：
    eff = (1 - ratio) * n_bits + ratio * protection_bits
"""

import numpy as np
import math
try:
    import torch
except ImportError:
    torch = None
from typing import Tuple, Dict
from .core import quantize_mse


def important_protection(
    W: 'torch.Tensor | np.ndarray',
    n_bits: int = 4,
    group_size: int = 128,
    protection_ratio: float = 0.15,
    protection_bits: int = 5,
) -> Tuple['torch.Tensor | np.ndarray', Dict]:
    """重要值保护量化
    
    1. 找到绝对值最大的 protection_ratio 比例的权重
    2. 这些权重用 protection_bits 量化
    3. 其余用 n_bits 量化
    
    Args:
        W: 权重矩阵
        n_bits: 基础量化精度
        group_size: 分组大小
        protection_ratio: 保护比例 (0.0-1.0)
        protection_bits: 保护权重的量化精度
    
    Returns:
        (W_q, info)
    """
    is_torch = torch is not None and isinstance(W, torch.Tensor)
    if is_torch:
        device, dtype = W.device, W.dtype
        W_np = W.float().cpu().numpy()
    else:
        W_np = W.astype(np.float32)
    
    result, info = _important_protection_numpy(W_np, n_bits, group_size, 
                                                protection_ratio, protection_bits)
    
    if is_torch:
        return torch.from_numpy(result).to(device=device, dtype=dtype), info
    return result, info


def _important_protection_numpy(
    W: np.ndarray,
    n_bits: int,
    group_size: int,
    protection_ratio: float,
    protection_bits: int,
) -> Tuple[np.ndarray, Dict]:
    """重要值保护量化 (NumPy 实现)"""
    original_shape = W.shape
    W_flat = W.reshape(-1)
    n = W_flat.size

    # 找到重要权重
    n_protect = max(1, int(n * protection_ratio))
    indices = np.argsort(np.abs(W_flat))[-n_protect:]
    
    mask = np.zeros(n, dtype=bool)
    mask[indices] = True

    # 分别量化
    W_important = W_flat[mask]
    W_remaining = W_flat[~mask]

    gs_imp = min(group_size, max(8, W_important.size))
    W_imp_q = quantize_mse(W_important.reshape(1, -1), n_bits=protection_bits, group_size=gs_imp).reshape(-1)
    W_rem_q = quantize_mse(W_remaining.reshape(1, -1), n_bits=n_bits, group_size=group_size).reshape(-1)

    # 合并
    W_q_flat = np.zeros_like(W_flat)
    W_q_flat[mask] = W_imp_q
    W_q_flat[~mask] = W_rem_q

    # ── 等效 bit 计算（修正）──
    # 存储内容：
    #   1. 量化值：(1-ratio)*n_bits + ratio*protection_bits
    #   2. 位置 bitmap：每参数 1 bit 标记是否为重要值
    #   3. scale/zero-point：两组量化各需 group 级 scale
    #
    # bitmap 方案 vs 索引方案：
    #   bitmap:  1 bit/param（固定）
    #   索引:    ratio * log2(n) bits/param（ratio<50% 时比 bitmap 贵）
    #   → 采用 bitmap（更通用）
    #
    # scale 开销：
    #   重要值组: ceil(n*ratio/group_size) 个 float32 scale
    #   非重要值组: ceil(n*(1-ratio)/group_size) 个 float32 scale
    n_groups_imp = math.ceil(n * protection_ratio / group_size)
    n_groups_rem = math.ceil(n * (1 - protection_ratio) / group_size)
    scale_bits = (n_groups_imp + n_groups_rem) * 32  # 每 group 一个 float32 scale
    
    eff_bits_values = (1 - protection_ratio) * n_bits + protection_ratio * protection_bits
    eff_bits_bitmap = 1.0  # 每参数 1 bit 位置标记
    eff_bits_scale = scale_bits / n
    eff_bits = eff_bits_values + eff_bits_bitmap + eff_bits_scale

    info = {
        'protection_ratio': protection_ratio,
        'protection_bits': protection_bits,
        'effective_bits_values': float(eff_bits_values),
        'effective_bits_bitmap': 1.0,
        'effective_bits_scale': float(eff_bits_scale),
        'effective_bits': float(eff_bits),
        'n_protect': n_protect,
        'n_total': n,
        'mse': float(np.mean((W_flat - W_q_flat) ** 2)),
    }

    return W_q_flat.reshape(original_shape), info

"""
自适应量化策略

根据每层权重的特征（离群值程度、低秩程度）自动选择最优量化策略。
"""

import numpy as np
try:
    import torch
except ImportError:
    torch = None
from typing import Tuple, Dict
from .core import quantize_mse
from .svd_hybrid import svd_hybrid
from .important import important_protection


def adaptive_quant(
    W: 'torch.Tensor | np.ndarray',
    n_bits: int = 4,
    group_size: int = 128,
) -> Tuple['torch.Tensor | np.ndarray', Dict]:
    """自适应量化 - 根据权重特征自动选择策略
    
    策略选择逻辑：
    1. 有低秩结构 → SVD hybrid
    2. 有离群值但无低秩 → 重要值保护
    3. 均匀分布 → 直接量化
    
    Args:
        W: 权重矩阵
        n_bits: 目标 bit 数
        group_size: 分组大小
    
    Returns:
        (W_q, info)
    """
    is_torch = torch is not None and isinstance(W, torch.Tensor)
    if is_torch:
        device, dtype = W.device, W.dtype
        W_np = W.float().cpu().numpy()
    else:
        W_np = W.astype(np.float32)
    
    result, info = _adaptive_numpy(W_np, n_bits, group_size)
    
    if is_torch:
        return torch.from_numpy(result).to(device=device, dtype=dtype), info
    return result, info


def _adaptive_numpy(W: np.ndarray, n_bits: int, group_size: int) -> Tuple[np.ndarray, Dict]:
    """自适应量化 (NumPy 实现)"""
    W_f = W.astype(np.float32)
    n = W_f.size

    # 分析权重特征
    w_abs = np.abs(W_f).reshape(-1)
    k_top = max(1, n // 100)
    top_vals = np.sort(w_abs)[-k_top:]
    outlier_energy = float(np.sum(top_vals ** 2) / np.sum(w_abs ** 2))

    try:
        _, S, _ = np.linalg.svd(W_f, full_matrices=False)
        k_svd = max(1, len(S) // 5)
        lowrank_energy = float(np.sum(S[:k_svd] ** 2) / np.sum(S ** 2))
    except Exception:
        lowrank_energy = 0.5

    has_outliers = outlier_energy > 0.2
    has_lowrank = lowrank_energy > 0.7

    if has_lowrank:
        # SVD hybrid - 低 rank 配置
        W_q, info = _svd_hybrid_low_rank(W_f, n_bits, group_size)
        info['strategy'] = 'adaptive→svd_hybrid'
    elif has_outliers:
        # 重要值保护
        W_q, info = _important_protection_numpy(W_f, n_bits, group_size, 0.15, 5)
        info['strategy'] = 'adaptive→important'
    else:
        W_q = quantize_mse(W_f, n_bits=n_bits, group_size=group_size)
        info = {
            'strategy': 'adaptive→direct',
            'effective_bits': float(n_bits),
            'mse': float(np.mean((W_f - W_q) ** 2)),
        }

    info['outlier_energy_ratio'] = outlier_energy
    info['lowrank_energy'] = lowrank_energy

    return W_q, info


def _svd_hybrid_low_rank(W: np.ndarray, n_bits: int, group_size: int):
    """低 rank SVD hybrid，用于自适应策略"""
    from .svd_hybrid import _svd_hybrid_numpy
    return _svd_hybrid_numpy(W, n_bits, group_size, 
                              energy_threshold=0.80, svd_bits=3, residual_bits=n_bits)


# 从 important 模块导入（避免循环引用的备用实现）
def _important_protection_numpy(W, n_bits, group_size, ratio, bits):
    from .important import _important_protection_numpy
    return _important_protection_numpy(W, n_bits, group_size, ratio, bits)

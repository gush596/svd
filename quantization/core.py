"""
核心量化函数

提供基础的对称/非对称 MSE 最优量化，是所有上层算法的基础。
"""

import numpy as np
try:
    import torch
except ImportError:
    torch = None
from typing import Tuple, Dict


def quantize_mse(
    W: 'torch.Tensor | np.ndarray',
    n_bits: int = 4,
    group_size: int = 128,
    n_pct_search: int = 20,
) -> 'torch.Tensor | np.ndarray':
    """MSE 最优对称量化
    
    对每个 group 独立搜索最优 scale，使量化后 MSE 最小。
    
    Args:
        W: 权重矩阵，支持 torch.Tensor 和 np.ndarray
        n_bits: 量化比特数
        group_size: 分组大小
        n_pct_search: 百分位数搜索数量
    
    Returns:
        量化-反量化后的权重（与输入类型相同）
    """
    is_torch = torch is not None and isinstance(W, torch.Tensor)
    
    if is_torch:
        device, dtype = W.device, W.dtype
        W_np = W.float().cpu().numpy()
    else:
        W_np = W.astype(np.float32)
    
    result = _quantize_mse_numpy(W_np, n_bits, group_size, n_pct_search)
    
    if is_torch:
        return torch.from_numpy(result).to(device=device, dtype=dtype)
    return result


def _quantize_mse_numpy(
    W: np.ndarray,
    n_bits: int,
    group_size: int,
    n_pct_search: int,
) -> np.ndarray:
    """MSE 最优对称量化 (NumPy 实现)"""
    original_shape = W.shape
    W_flat = W.reshape(-1)
    n = W_flat.size

    if n % group_size != 0:
        pad = group_size - (n % group_size)
        W_flat = np.concatenate([W_flat, np.zeros(pad, dtype=np.float32)])

    n_groups = W_flat.size // group_size
    W_groups = W_flat.reshape(n_groups, group_size)
    n_levels = 2 ** (n_bits - 1)

    pcts = np.concatenate([np.linspace(0.5, 1.0, n_pct_search), [0.995, 0.999]])
    best_scale = np.ones((n_groups, 1), dtype=np.float32)
    best_mse = np.full(n_groups, np.inf, dtype=np.float32)

    for pct in pcts:
        abs_max = np.percentile(np.abs(W_groups), pct * 100, axis=1, keepdims=True)
        abs_max = np.maximum(abs_max, 1e-8)
        scale = abs_max / n_levels
        W_q = np.clip(np.round(W_groups / scale), -n_levels, n_levels - 1)
        mse = ((W_groups - W_q * scale) ** 2).mean(axis=1)
        mask = mse < best_mse
        best_scale = np.where(mask.reshape(-1, 1), scale, best_scale)
        best_mse = np.where(mask, mse, best_mse)

    W_q = np.clip(np.round(W_groups / best_scale), -n_levels, n_levels - 1)
    return (W_q * best_scale).reshape(-1)[:n].reshape(original_shape)


def quantize_mse_asymmetric(
    W: 'torch.Tensor | np.ndarray',
    n_bits: int = 4,
    group_size: int = 128,
    n_pct_search: int = 20,
) -> 'torch.Tensor | np.ndarray':
    """非对称 MSE 最优量化（带 zero-point）"""
    is_torch = torch is not None and isinstance(W, torch.Tensor)
    if is_torch:
        device, dtype = W.device, W.dtype
        W_np = W.float().cpu().numpy()
    else:
        W_np = W.astype(np.float32)
    
    result = _quantize_asym_numpy(W_np, n_bits, group_size, n_pct_search)
    
    if is_torch:
        return torch.from_numpy(result).to(device=device, dtype=dtype)
    return result


def _quantize_asym_numpy(W, n_bits, group_size, n_pct_search):
    original_shape = W.shape
    W_flat = W.reshape(-1)
    n = W_flat.size
    if n % group_size != 0:
        W_flat = np.concatenate([W_flat, np.zeros(group_size - (n % group_size), dtype=np.float32)])
    n_groups = W_flat.size // group_size
    W_groups = W_flat.reshape(n_groups, group_size)
    n_levels = 2 ** n_bits
    pcts = np.linspace(0.5, 1.0, n_pct_search)
    best_scale = np.ones((n_groups, 1), dtype=np.float32)
    best_zero = np.zeros((n_groups, 1), dtype=np.float32)
    best_mse = np.full(n_groups, np.inf, dtype=np.float32)
    for pct in pcts:
        p_min = np.percentile(W_groups, (1-pct)*100, axis=1, keepdims=True)
        p_max = np.percentile(W_groups, pct*100, axis=1, keepdims=True)
        scale = (p_max - p_min) / (n_levels - 1)
        scale = np.maximum(scale, 1e-8)
        zero = p_min
        W_q = np.clip(np.round((W_groups - zero) / scale), 0, n_levels - 1)
        mse = ((W_groups - (W_q * scale + zero)) ** 2).mean(axis=1)
        mask = mse < best_mse
        best_scale = np.where(mask.reshape(-1,1), scale, best_scale)
        best_zero = np.where(mask.reshape(-1,1), zero, best_zero)
        best_mse = np.where(mask, mse, best_mse)
    W_q = np.clip(np.round((W_groups - best_zero) / best_scale), 0, n_levels - 1)
    return (W_q * best_scale + best_zero).reshape(-1)[:n].reshape(original_shape)

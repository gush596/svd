"""
自适应幂迭代残差 SVD 量化

在迭代残差 SVD 基础上引入:
1. 幂迭代 SVD 替代完整 SVD（加速）
2. 自适应 rank：每轮根据残差能量自动选择 rank
3. 热启动：利用上一轮子空间加速收敛
4. 精细化：对收敛结果做 fixed-point refinement

等效 bit 公式（残差舍弃模式）:
    eff = sum_i(rank_i × (out × u_bits + s_bits + in × v_bits)) / (out × in)
    
由于 rank_i 每轮可能不同，需要逐轮累加。
"""

import numpy as np
try:
    import torch
except ImportError:
    torch = None
from typing import Tuple, Dict, Optional
from .core import quantize_mse
from .power_iter_svd import (
    power_iteration_svd,
    power_iteration_svd_with_warmstart,
    adaptive_rank_svd,
)


def adaptive_iterative_svd(
    W: 'torch.Tensor | np.ndarray',
    group_size: int = 128,
    max_eff_bits: float = 4.0,
    rank: int = 1,
    u_bits: int = 3,
    s_bits: int = 4,
    v_bits: int = 3,
    n_rounds: Optional[int] = None,
    use_power_iter: bool = True,
    power_iter_steps: int = 4,
    use_warmstart: bool = True,
    use_adaptive_rank: bool = False,
    adaptive_energy_threshold: float = 0.8,
    max_adaptive_rank: int = 8,
) -> Tuple['torch.Tensor | np.ndarray', Dict]:
    """自适应幂迭代残差 SVD 量化
    
    Args:
        W: 权重矩阵 [out, in]
        group_size: 量化分组大小
        max_eff_bits: 最大等效 bit
        rank: 每轮保留的奇异值数量（use_adaptive_rank=False 时使用）
        u_bits, s_bits, v_bits: U, S, V 的量化 bit 数
        n_rounds: 显式指定轮数
        use_power_iter: 是否使用幂迭代（False 则用完整 SVD）
        power_iter_steps: 幂迭代步数
        use_warmstart: 是否使用热启动
        use_adaptive_rank: 是否自适应选择 rank
        adaptive_energy_threshold: 自适应 rank 的能量阈值
        max_adaptive_rank: 自适应 rank 的上限
    
    Returns:
        (W_approx, info)
    """
    is_torch = torch is not None and isinstance(W, torch.Tensor)
    if is_torch:
        device, dtype = W.device, W.dtype
        W_np = W.float().cpu().numpy()
    else:
        W_np = W.astype(np.float32)
    
    result, info = _adaptive_iterative_svd_numpy(
        W_np, group_size, max_eff_bits, rank, u_bits, s_bits, v_bits,
        n_rounds, use_power_iter, power_iter_steps, use_warmstart,
        use_adaptive_rank, adaptive_energy_threshold, max_adaptive_rank,
    )
    
    if is_torch:
        return torch.from_numpy(result).to(device=device, dtype=dtype), info
    return result, info


def _adaptive_iterative_svd_numpy(
    W: np.ndarray,
    group_size: int,
    max_eff_bits: float,
    rank: int,
    u_bits: int,
    s_bits: int,
    v_bits: int,
    n_rounds_override: Optional[int],
    use_power_iter: bool,
    power_iter_steps: int,
    use_warmstart: bool,
    use_adaptive_rank: bool,
    adaptive_energy_threshold: float,
    max_adaptive_rank: int,
) -> Tuple[np.ndarray, Dict]:
    """核心实现"""
    out_dim, in_dim = W.shape
    total_params = W.size
    W_f = W.astype(np.float32)
    
    # 确定 SVD 方法
    if use_power_iter:
        def svd_fn(W_residual, r, prev_U=None, prev_Vt=None):
            if use_warmstart and prev_Vt is not None:
                return power_iteration_svd_with_warmstart(
                    W_residual, rank=r, n_iter=power_iter_steps,
                    prev_U=prev_U, prev_Vt=prev_Vt,
                )
            else:
                return power_iteration_svd(
                    W_residual, rank=r, n_iter=power_iter_steps,
                )
    else:
        def svd_fn(W_residual, r, prev_U=None, prev_Vt=None):
            U, S, Vt = np.linalg.svd(W_residual, full_matrices=False)
            return U[:, :r], S[:r], Vt[:r, :]
    
    # 计算每轮 bit 开销（固定 rank 情况）
    if not use_adaptive_rank:
        round_bits = rank * (out_dim * u_bits + s_bits + in_dim * v_bits)
        round_eff = round_bits / total_params
        if n_rounds_override is not None:
            n_rounds = n_rounds_override
        else:
            n_rounds = int(max_eff_bits / round_eff)
    else:
        # 自适应 rank：用平均 rank 估算
        est_rank = min(max_adaptive_rank, min(out_dim, in_dim)) // 2
        round_bits = est_rank * (out_dim * u_bits + s_bits + in_dim * v_bits)
        round_eff = round_bits / total_params
        if n_rounds_override is not None:
            n_rounds = n_rounds_override
        else:
            n_rounds = int(max_eff_bits / round_eff)
    
    if n_rounds <= 0:
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
    prev_U, prev_Vt = None, None
    total_bits_used = 0
    
    for i in range(n_rounds):
        # 选择本轮 rank
        if use_adaptive_rank:
            _, S_probe, _, actual_rank = adaptive_rank_svd(
                residual,
                energy_threshold=adaptive_energy_threshold,
                max_rank=max_adaptive_rank,
                min_rank=1,
                n_iter=power_iter_steps,
            )
            actual_rank = min(actual_rank, max_adaptive_rank)
        else:
            actual_rank = rank
        
        # 检查 bit 预算
        this_round_bits = actual_rank * (out_dim * u_bits + s_bits + in_dim * v_bits)
        if total_bits_used + this_round_bits > max_eff_bits * total_params:
            # 预算不够再跑一轮完整 rank，尝试降 rank
            remaining_bits = max_eff_bits * total_params - total_bits_used
            max_possible_rank = int(remaining_bits / (out_dim * u_bits + s_bits + in_dim * v_bits))
            if max_possible_rank <= 0:
                break
            actual_rank = min(actual_rank, max_possible_rank)
            this_round_bits = actual_rank * (out_dim * u_bits + s_bits + in_dim * v_bits)
            if total_bits_used + this_round_bits > max_eff_bits * total_params:
                break
        
        # SVD 分解
        U, S, Vt = svd_fn(residual, actual_rank, prev_U, prev_Vt)
        
        # 量化 U, S, V
        gs_u = min(group_size, max(8, actual_rank))
        U_q = quantize_mse(U, n_bits=u_bits, group_size=gs_u)
        
        gs_s = min(group_size, max(8, actual_rank))
        S_q = quantize_mse(S.reshape(1, -1), n_bits=s_bits, group_size=gs_s).reshape(-1)
        
        gs_v = min(group_size, max(8, actual_rank))
        V_q = quantize_mse(Vt, n_bits=v_bits, group_size=gs_v)
        
        # 重建本轮分量
        component = U_q @ np.diag(S_q) @ V_q
        W_approx = W_approx + component
        residual = residual - component
        
        # 更新热启动状态
        prev_U = U
        prev_Vt = Vt
        
        total_bits_used += this_round_bits
        
        round_info = {
            'round': i + 1,
            'rank': actual_rank,
            'u_bits': u_bits,
            's_bits': s_bits,
            'v_bits': v_bits,
            'round_bits': this_round_bits,
            'residual_norm': float(np.linalg.norm(residual)),
            'residual_mse': float(np.mean(residual ** 2)),
            'singular_values': S.tolist(),
        }
        round_infos.append(round_info)
    
    eff_bits = total_bits_used / total_params
    
    info = {
        'rounds': len(round_infos),
        'round_details': round_infos,
        'effective_bits': float(eff_bits),
        'mse': float(np.mean((W_f - W_approx) ** 2)),
        'residual_discarded': True,
        'residual_mse': float(np.mean(residual ** 2)),
        'method': 'power_iter' if use_power_iter else 'full_svd',
        'warmstart': use_warmstart,
        'adaptive_rank': use_adaptive_rank,
    }
    
    return W_approx, info

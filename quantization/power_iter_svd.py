"""
幂迭代奇异值分解 (Power Iteration SVD)

使用正交迭代（orthogonal iteration）找到 top-rank 奇异子空间，
然后在子空间上做精确 SVD 得到准确的奇异值。

算法:
1. 初始化随机矩阵 Q ∈ R^{in × rank}
2. 正交迭代:
   Y = W @ Q;  Q, _ = qr(Y)
   Y = W.T @ Q;  Q, _ = qr(Y)
3. 子空间 SVD:
   B = W @ Q          # B ∈ R^{out × rank}
   Ub, S, Vt_small = svd(B)  # 精确 SVD
   U = Ub[:, :rank]
   Vt = Vt_small[:rank, :] @ Q.T  # 映射回原始空间

优势:
- 正交迭代只做矩阵乘法，O(n_iter × out × in × rank)
- 精确 SVD 在 out × rank 的小矩阵上做
- 当 rank << min(out,in) 时远快于完整 SVD O(out × in × min(out,in))
"""

import numpy as np
from typing import Tuple, Optional


def power_iteration_svd(
    W: np.ndarray,
    rank: int = 8,
    n_iter: int = 4,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """幂迭代 SVD，返回 top-rank 奇异值/向量
    
    Args:
        W: 权重矩阵 [out, in]
        rank: 需要的奇异值数量
        n_iter: 正交迭代次数（2-6 通常足够）
        seed: 随机种子
    
    Returns:
        (U, S, Vt) 与 np.linalg.svd 格式一致
        U: [out, rank], S: [rank], Vt: [rank, in]
    """
    out_dim, in_dim = W.shape
    rank = min(rank, min(out_dim, in_dim))
    
    rng = np.random.RandomState(seed)
    Q = rng.randn(in_dim, rank).astype(np.float32)
    Q, _ = np.linalg.qr(Q)
    
    # 正交迭代：交替投影到行空间和列空间
    for _ in range(n_iter):
        Y = W @ Q           # [out, rank]
        Q_u, _ = np.linalg.qr(Y)
        Y = W.T @ Q_u       # [in, rank]
        Q, _ = np.linalg.qr(Y)
    
    # 在子空间上做精确 SVD
    # B = W @ Q → [out, rank]
    B = W @ Q
    Ub, S, Vt_sub = np.linalg.svd(B, full_matrices=False)
    
    # 映射回原始空间: V = Q @ Vt_sub^T
    Vt = Vt_sub @ Q.T  # [rank, in]
    
    # 取 top-rank
    U = Ub[:, :rank]
    S = S[:rank]
    Vt = Vt[:rank, :]
    
    return U, S, Vt


def power_iteration_svd_with_warmstart(
    W: np.ndarray,
    rank: int = 8,
    n_iter: int = 4,
    prev_U: Optional[np.ndarray] = None,
    prev_Vt: Optional[np.ndarray] = None,
    warmstart_weight: float = 0.7,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """带热启动的幂迭代 SVD
    
    利用上一轮的 V 子空间作为初始猜测，加速收敛。
    
    Args:
        W: 权重矩阵 [out, in]
        rank: 需要的奇异值数量
        n_iter: 正交迭代次数
        prev_U: 上一轮的 U 矩阵 [out, rank]（可选）
        prev_Vt: 上一轮的 Vt 矩阵 [rank, in]（可选）
        warmstart_weight: 热启动混合权重（0=纯随机, 1=纯历史）
    
    Returns:
        (U, S, Vt)
    """
    out_dim, in_dim = W.shape
    rank = min(rank, min(out_dim, in_dim))
    
    # 初始化 Q：混合上一轮的 V 和随机初始化
    if prev_Vt is not None and prev_Vt.shape[0] >= rank:
        Q = prev_Vt[:rank, :].T.copy()  # [in, rank]
        rng = np.random.RandomState(0)
        noise = rng.randn(in_dim, rank).astype(np.float32) * 0.01
        Q = warmstart_weight * Q + (1 - warmstart_weight) * noise
        Q, _ = np.linalg.qr(Q)
    else:
        rng = np.random.RandomState(0)
        Q = rng.randn(in_dim, rank).astype(np.float32)
        Q, _ = np.linalg.qr(Q)
    
    for _ in range(n_iter):
        Y = W @ Q
        Q_u, _ = np.linalg.qr(Y)
        Y = W.T @ Q_u
        Q, _ = np.linalg.qr(Y)
    
    B = W @ Q
    Ub, S, Vt_sub = np.linalg.svd(B, full_matrices=False)
    Vt = Vt_sub @ Q.T
    
    U = Ub[:, :rank]
    S = S[:rank]
    Vt = Vt[:rank, :]
    
    return U, S, Vt


def adaptive_rank_svd(
    W: np.ndarray,
    energy_threshold: float = 0.9,
    max_rank: int = 32,
    min_rank: int = 1,
    n_iter: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """自适应 rank 的幂迭代 SVD
    
    通过逐步增加 rank，找到满足能量阈值的最小 rank。
    
    Args:
        W: 权重矩阵
        energy_threshold: 保留的奇异值能量比例 (0-1)
        max_rank: 最大 rank
        min_rank: 最小 rank
        n_iter: 幂迭代次数
    
    Returns:
        (U, S, Vt, actual_rank)
    """
    probe_rank = min(max_rank, min(W.shape))
    U, S, Vt = power_iteration_svd(W, rank=probe_rank, n_iter=n_iter)
    
    total_energy = np.sum(S ** 2)
    cumulative = np.cumsum(S ** 2)
    
    actual_rank = min_rank
    for i in range(len(S)):
        if cumulative[i] / total_energy >= energy_threshold:
            actual_rank = i + 1
            break
    
    actual_rank = max(min_rank, min(actual_rank, max_rank))
    
    return U[:, :actual_rank], S[:actual_rank], Vt[:actual_rank, :], actual_rank

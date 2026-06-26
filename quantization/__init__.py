"""
SVD 量化工具包

模块结构：
    core.py          - 基础量化函数 (quantize_mse, quantize_mse_asymmetric)
    svd_hybrid.py    - SVD Hybrid 量化算法
    important.py     - 重要值保护量化
    adaptive.py      - 自适应量化策略
"""

from .core import quantize_mse, quantize_mse_asymmetric
from .svd_hybrid import svd_hybrid, compute_effective_bits
from .important import important_protection
from .adaptive import adaptive_quant
from .iterative_svd import iterative_residual_svd, compute_max_rounds
from .outlier_svd import outlier_svd_quantize
from .iterative_outlier_svd import iterative_outlier_svd

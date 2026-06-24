# SVD 量化项目

基于 SVD 分解的后训练量化方法，目标是在等效 ≤4-bit、group_size ≥128 的约束下最大化量化精度。

## 项目结构

```
svd/
├── quantization/              # 核心算法
│   ├── __init__.py
│   ├── core.py                # 基础量化函数 (quantize_mse)
│   ├── svd_hybrid.py          # SVD Hybrid 算法
│   ├── important.py           # 重要值保护量化
│   ├── adaptive.py            # 自适应策略
│   └── iterative_svd.py       # 迭代残差 SVD
├── tests/                     # 测试
│   ├── test_numpy.py          # NumPy 合成数据验证（快速，无需 GPU）
│   ├── test_model.py          # 真实模型 PPL 测试（需要下载 OPT-125M）
│   └── test_iterative.py      # 迭代残差 SVD 测试
├── EXPERIMENT_RESULTS.md      # 实验报告
└── README.md                  # 本文件
```

## 快速开始

```bash
# NumPy 算法验证（快速，不需要模型下载）
python tests/test_numpy.py

# 迭代残差 SVD 测试
python tests/test_iterative.py

# 真实模型 PPL 测试
python tests/test_model.py --quick

# 完整测试
python tests/test_model.py

# 只测 SVD hybrid
python tests/test_model.py --method svd_hybrid
```

## 算法概览

### 1. 直接量化 (baseline)
```
W_q = quantize_mse(W, n_bits=4, group_size=128)
等效 bit = 4.0, PPL delta ≈ +6.0
```

### 2. 重要值保护
```
保护 top-k% 离群值用高精度量化
eff = (1-ratio)*n_bits + ratio*prot_bits
最优: ratio=0.15, bits=5 → eff=4.15, PPL delta ≈ +0.7
```

### 3. SVD Hybrid
```
W ≈ SVD低秩重建(2-3bit) + 残差(4bit)
eff = rank*(out+in)*svd_bits/(out*in) + 4
注意: eff > 4.0，无法满足 ≤4-bit 约束
```

### 4. 迭代残差 SVD (残差舍弃模式)
```
每轮: SVD分解 → 量化U/S/V → 更新残差
最终残差舍弃，不量化
eff = n_rounds × rank × (out × u_bits + s_bits + in × v_bits) / (out × in)
```

## 等效 bit 计算

```
重要值保护:  eff = (1-ratio)*n_bits + ratio*prot_bits
SVD Hybrid:  eff = rank*(out+in)*svd_bits/(out*in) + residual_bits
迭代残差SVD: eff = n_rounds × rank × (out × u_bits + s_bits + in × v_bits) / (out × in)
```

## 实验结果

详见 [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md)

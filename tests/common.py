"""
共享工具模块

所有测试文件共用的工具函数，避免重复代码。
"""

import os
import sys
import time
import math
import numpy as np

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ─── 数据生成 ─────────────────────────────────────────────────────────

def generate_weight_matrix(rows: int, cols: int, seed: int = 42,
                           low_rank_ratio: float = 0.125,
                           outlier_pct: float = 0.01,
                           noise_scale: float = 0.1) -> np.ndarray:
    """生成模拟 Transformer 权重的测试矩阵
    
    特点：低秩结构 + 稀疏离群值 + 高斯噪声
    
    Args:
        rows, cols: 矩阵尺寸
        seed: 随机种子
        low_rank_ratio: 低秩部分的 rank = min(rows,cols) * low_rank_ratio
        outlier_pct: 离群值占比
        noise_scale: 噪声标准差
    """
    rng = np.random.RandomState(seed)
    rank = max(1, int(min(rows, cols) * low_rank_ratio))
    U = rng.randn(rows, rank).astype(np.float32)
    S = np.exp(-np.arange(rank) * 0.3).astype(np.float32)
    V = rng.randn(rank, cols).astype(np.float32)
    W = (U * S.reshape(1, -1)) @ V + rng.randn(rows, cols).astype(np.float32) * noise_scale
    # 添加离群值
    n_out = max(1, int(rows * cols * outlier_pct))
    out_idx = rng.choice(rows * cols, n_out, replace=False)
    out = np.zeros(rows * cols, dtype=np.float32)
    out[out_idx] = rng.randn(n_out) * 5.0
    return W + out.reshape(rows, cols)


# ─── PPL 评估 ─────────────────────────────────────────────────────────

def load_model_and_tokenizer(model_id: str = "facebook/opt-125m",
                              device: str = "cpu",
                              dtype=None):
    """加载模型和 tokenizer"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if dtype is None:
        dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype,
        device_map="auto" if device == "cuda" else "cpu",
        low_cpu_mem_usage=True, trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


@staticmethod
def _evaluate_ppl_impl(model, tokenizer, n_samples=5, seqlen=512):
    """评估模型 PPL（内部实现）"""
    import torch
    from datasets import load_dataset

    data = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n".join(data["text"])
    ids = tokenizer(text, return_tensors="pt").input_ids
    dev = next(model.parameters()).device
    nlls, nt = [], 0
    for i in range(min(n_samples, ids.shape[1] // seqlen)):
        s, e = i * seqlen, min((i + 1) * seqlen, ids.shape[1])
        if e - s < seqlen:
            break
        try:
            inp = ids[:, s:e].to(dev)
            o = model(inp, labels=inp)
            nlls.append(o.loss.item() * (e - s))
            nt += (e - s)
        except Exception as ex:
            print(f"    [warn] sample {i}: {ex}")
    return float('inf') if nt == 0 else torch.exp(torch.tensor(sum(nlls) / nt)).item()


def evaluate_ppl(model, tokenizer, n_samples=5, seqlen=512):
    """评估模型 PPL"""
    return _evaluate_ppl_impl(model, tokenizer, n_samples, seqlen)


# ─── 量化辅助 ─────────────────────────────────────────────────────────

def apply_quant_to_model(model, quant_fn, **kwargs):
    """对模型所有 Linear 层应用量化函数
    
    Args:
        model: HuggingFace 模型
        quant_fn: 量化函数 W_q, info = quant_fn(W, **kwargs)
        **kwargs: 传递给 quant_fn 的参数
    
    Returns:
        (elapsed_time, layer_infos)
    """
    import torch
    import torch.nn as nn

    t0 = time.time()
    layer_infos = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and 'lm_head' not in name:
            W = mod.weight.data.float().cpu().numpy()
            W_q, info = quant_fn(W, **kwargs)
            mod.weight.data = torch.from_numpy(W_q).to(mod.weight.dtype)
            layer_infos.append(info)
    return time.time() - t0, layer_infos


# ─── 等效 bit 计算 ────────────────────────────────────────────────────

def compute_eff_raw(out_dim, in_dim, rank, u_bits, s_bits, v_bits, n_rounds):
    """计算不含 scale 的等效 bit"""
    s_bits_eff = s_bits if s_bits is not None else 32
    round_bits = rank * (out_dim * u_bits + in_dim * v_bits) + rank * s_bits_eff
    return n_rounds * round_bits / (out_dim * in_dim)


def compute_eff_full(out_dim, in_dim, rank, u_bits, s_bits, v_bits,
                     n_rounds, group_size=128):
    """计算含 scale 的综合等效 bit"""
    s_bits_eff = s_bits if s_bits is not None else 32
    round_raw = rank * (out_dim * u_bits + in_dim * v_bits) + rank * s_bits_eff
    gs_u = min(group_size, max(8, rank))
    gs_v = min(group_size, max(8, rank))
    n_groups_u = math.ceil(out_dim * rank / gs_u)
    n_groups_v = math.ceil(rank * in_dim / gs_v)
    n_groups_s = 1 if (s_bits is not None and s_bits < 16) else 0
    scale_bits = (n_groups_u + n_groups_s + n_groups_v) * 32
    return n_rounds * (round_raw + scale_bits) / (out_dim * in_dim)


# ─── 打印辅助 ─────────────────────────────────────────────────────────

def print_table(headers, rows, col_widths=None):
    """打印格式化表格"""
    if col_widths is None:
        col_widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0)) + 2
                      for i, h in enumerate(headers)]
    header_line = "".join(str(h).ljust(w) for h, w in zip(headers, col_widths))
    print(header_line)
    print("-" * sum(col_widths))
    for row in rows:
        print("".join(str(v).ljust(w) for v, w in zip(row, col_widths)))


def print_section(title, width=70):
    """打印分隔标题"""
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")

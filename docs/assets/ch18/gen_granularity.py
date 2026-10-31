# -*- coding: utf-8 -*-
"""图②：量化粒度对比——同一真实权重上 per-tensor / per-channel / per-group 的 RTN 误差（CPU 即可复现）。

数据：openai-community/gpt2 的 h.11.attn.c_proj.weight（重尾权重，量化粒度影响最明显），
与 gen_rtn_bitwidth.py 共用同目录 gpt2_weights_sample.npz（抽取命令见该脚本头部注释）。
实验：int8 / int4 / int3 三档位宽 x 三种粒度，对称 RTN，真实计算相对 MSE。
运行：python gen_granularity.py
"""
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))


def load_weight():
    """返回 [out, in]（GPT-2 Conv1D 原始存储为 [in, out]，此处转置，与线性层习惯一致）。"""
    npz = os.path.join(HERE, "gpt2_weights_sample.npz")
    if os.path.exists(npz):
        return np.load(npz)["attn_c_proj_L11"].T, "真实权重：openai-community/gpt2 h.11.attn.c_proj"
    env = os.environ.get("GPT2_SAFETENSORS")
    if env and os.path.exists(env):
        from safetensors.numpy import load_file
        return load_file(env)["h.11.attn.c_proj.weight"].T, "真实权重：openai-community/gpt2"
    rng = np.random.default_rng(0)
    w = rng.normal(0, 0.182, size=(768, 768))
    idx = rng.choice(w.size, size=int(w.size * 0.005), replace=False)
    w.flat[idx] *= rng.uniform(20, 50, size=idx.size)
    return w, "合成分布：标准正态 + 0.5% 大离群（非真实模型权重）"


def rtn_quantize(w, bits, granularity, group=128):
    """对称 RTN，granularity ∈ {tensor, channel, group}。w 形状 [out, in]，按输入通道维分组。"""
    qmax = 2 ** (bits - 1) - 1
    if granularity == "tensor":
        g = w.reshape(1, -1)
    elif granularity == "channel":
        g = w  # 每行（输出通道对输入维）一个 scale
    else:
        g = w.reshape(w.shape[0], -1, group)
    s = np.abs(g).max(axis=-1, keepdims=True) / qmax
    q = np.clip(np.round(g / s), -qmax - 1, qmax)
    return (q * s).reshape(w.shape)


W, source = load_weight()
print("数据源：", source, "| 形状:", W.shape, "std=%.4f absmax=%.3f" % (W.std(), np.abs(W).max()))

grans = [("per-tensor", "tensor"), ("per-channel", "channel"), ("per-group g=128", "group")]
bitss = [8, 4, 3]
results = np.zeros((len(bitss), len(grans)))
for bi, b in enumerate(bitss):
    for gi, (name, key) in enumerate(grans):
        Wq = rtn_quantize(W, b, key)
        rel = np.mean((Wq - W) ** 2) / np.mean(W ** 2)
        results[bi, gi] = rel
        print(f"int{b} {name:<14s}: 相对MSE = {rel:.3e}")

x = np.arange(len(grans))
w = 0.26
colors = ["#43a047", "#1976d2", "#c62828"]
fig, ax = plt.subplots(figsize=(9.5, 5.6), dpi=150)
for bi, b in enumerate(bitss):
    bars = ax.bar(x + (bi - 1) * w, results[bi], w, label=f"int{b}", color=colors[bi], zorder=3)
    for r, xx in zip(results[bi], x + (bi - 1) * w):
        ax.text(xx, r * 1.25, f"{r:.1e}", ha="center", fontsize=8, rotation=0)

ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels([g[0] for g in grans], fontsize=11)
ax.set_ylabel("权重相对 MSE（对数轴）", fontsize=11)
ax.set_ylim(results.min() / 8, results.max() * 6)
ax.set_title("量化粒度越细，误差越小：同一 GPT-2 权重上的真实计算（对称 RTN）", fontsize=12)
ax.grid(True, axis="y", which="both", alpha=0.3, zorder=0)
ax.legend(fontsize=11, title="位宽", title_fontsize=10)

fig.tight_layout()
out = os.path.join(HERE, "granularity_mse.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved:", out)

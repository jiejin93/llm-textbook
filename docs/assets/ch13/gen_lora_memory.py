# -*- coding: utf-8 -*-
"""生成 13.6.4 节图：Qwen2.5-7B 微调显存 vs LoRA 秩（按显存公式逐项真实计算）。

显存模型（与本章正文公式一致，GB 按 10^9 字节计）：
  基座权重      bf16: 2N 字节        | NF4(含双重量化): 0.5N 字节
  LoRA 状态     可训练参数 P = 28 × r × (2×(d+d) + 2×(d+k_v))，每参数
                bf16 权重 2 + bf16 梯度 2 + fp32 主权重/m/v 12 = 18 字节
  全参微调      梯度 bf16 2N + AdamW fp32 状态（m+v+主权重）12N
  激活+峰值     梯度检查点下逐层输入 28×2×B×T×d 字节 + logits 峰值 T×V×6 字节
                （bf16 logits + fp32 softmax）+ CUDA 上下文 1.0 GB
  常数取 Qwen2.5-7B：N=7.6e9, d=3584, L=28, k_v=512, T=2048, B=1, V=151936

运行：python gen_lora_memory.py   （依赖 numpy + matplotlib）
输出：gen_lora_memory.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

GB = 1e9
# ---- Qwen2.5-7B 常数 ----
N, d, L, k_v, T, B, V = 7.6e9, 3584, 28, 512, 2048, 1, 151936

def lora_params(r: int) -> int:
    """注意力四个投影（q/k/v/o）挂 LoRA 的可训练参数量。"""
    per_layer = r * (d + d) + r * (d + k_v) * 2 + r * (d + d)  # q, k, v, o
    return L * per_layer

def act_gb(logits_factor: float = 6.0) -> float:
    """激活（梯度检查点）+ logits 峰值 + CUDA 上下文，单位 GB。"""
    ckpt = L * 2 * B * T * d                    # 逐层保存的输入（字节）
    logits = T * V * logits_factor              # bf16 logits + fp32 softmax
    return (ckpt + logits) / GB + 1.0           # +1 GB CUDA 上下文/工作区

def fft_gb() -> tuple:
    weights = 2 * N / GB
    grads = 2 * N / GB
    opt = 12 * N / GB
    total = weights + grads + opt + act_gb()
    return weights, grads, opt, total

def lora_gb(r: int) -> tuple:
    weights = 2 * N / GB
    P = lora_params(r)
    lora_state = P * 18 / GB                    # 权重2+梯度2+优化器12 字节/参数
    other = act_gb()
    return weights, lora_state, other, weights + lora_state + other

def qlora_gb(r: int) -> tuple:
    weights = 0.5 * N / GB                      # NF4 + 双重量化 ≈ 0.5 B/参数
    P = lora_params(r)
    lora_state = P * 18 / GB
    dequant = 0.5                               # 反量化工作区/分页缓冲（估）
    other = act_gb() + dequant
    return weights, lora_state, other, weights + lora_state + other

def infer_gb() -> tuple:
    weights = 2 * N / GB
    kv = 2 * L * 2 * k_v * T * B / GB           # KV cache（bf16，粗估）
    other = T * V * 2 / GB + 1.0
    return weights, 0.0, kv + other, weights + kv + other

bars = [
    ("r=0\n(仅推理 bf16)", infer_gb()),
    ("LoRA\nr=8", lora_gb(8)),
    ("LoRA\nr=16", lora_gb(16)),
    ("LoRA\nr=64", lora_gb(64)),
    ("QLoRA\nr=16", qlora_gb(16)),
    ("全参微调\nFFT", fft_gb()),
]
print("显存明细（GB）：")
for name, parts in bars:
    print(f"  {name.replace(chr(10), ' '):<14} 基座权重={parts[0]:7.2f}  LoRA状态={parts[1]:7.3f}  "
          f"激活/优化器/上下文={parts[2]:7.2f}  合计={parts[3]:8.2f}")

labels = [b[0] for b in bars]
# 三段堆叠：基座权重 / LoRA 状态与梯度 / 激活·优化器·上下文
seg1 = np.array([b[1][0] for b in bars])
seg2 = np.array([b[1][1] for b in bars])
seg3 = np.array([b[1][2] for b in bars])
fft_idx = labels.index("全参微调\nFFT")
seg3[fft_idx] += bars[fft_idx][1][3] - seg1[fft_idx] - seg2[fft_idx] - seg3[fft_idx]  # FFT 的优化器状态并入第三段
totals = np.array([b[1][3] for b in bars])

colors = ["#4C78A8", "#F58518", "#B0B0B0"]
fig, ax = plt.subplots(figsize=(9.6, 5.4))
x = np.arange(len(labels))
ax.bar(x, seg1, 0.62, label="基座权重（bf16 / NF4）", color=colors[0])
ax.bar(x, seg2, 0.62, bottom=seg1, label="可训练参数相关（LoRA 旁路 / bf16 梯度）", color=colors[1])
ax.bar(x, seg3, 0.62, bottom=seg1 + seg2, label="激活/logits/优化器/上下文", color=colors[2])

for i, t in enumerate(totals):
    ax.text(x[i], t + 2.5, f"{t:.1f}", ha="center", fontsize=10, fontweight="bold")

ax.axhline(12, color="crimson", ls="--", lw=1.4)
ax.text(-0.42, 13.2, "12 GB 显存上限", color="crimson", fontsize=10, ha="left")
ax.set_yscale("log")
ax.set_ylim(1, 400)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9.5)
ax.set_ylabel("训练显存（GB，对数轴）")
ax.set_title("Qwen2.5-7B 各微调方式的显存构成（按本章公式逐项计算，seq=2048、batch=1、梯度检查点开启）",
             fontsize=11)
ax.legend(fontsize=9, loc="upper left")
ax.grid(axis="y", alpha=0.3, which="both")

fig.tight_layout()
fig.savefig("gen_lora_memory.png", dpi=150, bbox_inches="tight")
print("已保存 gen_lora_memory.png")

# -*- coding: utf-8 -*-
"""KV 缓存显存 vs 注意力方案 × 上下文长度 的分组柱状图。
按 8.1 公式 M_KV = 2 × L × h_kv × d_k × T × bytes 真实计算：
统一取 7B 级骨架（L=32 层、d_k=128、fp16 bytes=2、batch=1）：
  MHA    h_kv=32          -> 512 KiB/token
  GQA-8  h_kv=8           -> 128 KiB/token
  GQA-4  h_kv=4           ->  64 KiB/token
  MLA    每层缓存 d_c+d_R^k = 512+64 = 576 个元素（DeepSeek-V2 的压缩宽度）-> 36 KiB/token
运行：python gen_kv_cache.py，输出 kv_cache.png。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

L, DK, BYTES = 32, 128, 2                      # 7B 级：32 层，头维 128，fp16
GIB = 2 ** 30

def kv_gib(h_kv_or_width, T, mla=False):
    if mla:                                    # MLA：每层每 token 缓存 576 个元素
        per_token = h_kv_or_width * BYTES * L
    else:                                      # MHA/GQA：2(K,V) × h_kv × d_k × bytes × L
        per_token = 2 * h_kv_or_width * DK * BYTES * L
    return per_token * T / GIB

schemes = [
    ("MHA（h_kv=32）",  32, False, "#4C72B0"),
    ("GQA-8（h_kv=8）", 8,  False, "#DD8452"),
    ("GQA-4（h_kv=4）", 4,  False, "#55A868"),
    ("MLA（576 维/层）", 576, True, "#C44E52"),
]
contexts = [(4096, "4K"), (32768, "32K"), (131072, "128K")]

fig, ax = plt.subplots(figsize=(9, 5), dpi=140)
x = np.arange(len(contexts))
width = 0.19
for i, (name, w, mla, color) in enumerate(schemes):
    vals = [kv_gib(w, T, mla) for T, _ in contexts]
    bars = ax.bar(x + (i - 1.5) * width, vals, width, label=name, color=color,
                  edgecolor="black", linewidth=0.4)
    for b, v in zip(bars, vals):
        txt = f"{v:.2f}" if v < 1 else f"{v:.1f}"
        ax.text(b.get_x() + b.get_width() / 2, v * 1.15, txt,
                ha="center", fontsize=8, rotation=0)

ax.set_yscale("log")
ax.set_ylim(0.05, 300)
ax.set_xticks(x)
ax.set_xticklabels([lab for _, lab in contexts], fontsize=11)
ax.set_xlabel("上下文长度 T（batch=1）")
ax.set_ylabel("KV 缓存显存（GiB，对数轴）")
ax.set_title("同一 7B 级骨架下，KV 缓存显存由注意力方案决定（公式真实计算，fp16）",
             fontsize=11.5)
ax.legend(fontsize=9, loc="upper left", framealpha=0.95)
ax.grid(True, which="major", axis="y", ls="--", lw=0.4, alpha=0.5)

fig.tight_layout()
fig.savefig("kv_cache.png", dpi=140)
for name, w, mla, _ in schemes:
    row = [f"T={lab:>4s}: {kv_gib(w, T, mla):8.3f} GiB" for T, lab in contexts]
    print(f"{name:16s} " + "   ".join(row))
print("已保存 kv_cache.png")

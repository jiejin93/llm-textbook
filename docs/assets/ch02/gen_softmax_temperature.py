# -*- coding: utf-8 -*-
"""softmax 温度缩放实验：同一组 logits 在 T=0.1 / 1.0 / 10.0 下的分布变化。
由代码生成，脚本同目录可复现：python gen_softmax_temperature.py
依赖：matplotlib, numpy
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 一组典型的词表打分（logits）：下一词候选从热门到冷门
labels = ["候选 1", "候选 2", "候选 3", "候选 4", "候选 5", "候选 6"]
z = np.array([3.2, 2.1, 1.4, 0.7, 0.0, -1.3])

def softmax_with_T(z, T):
    zz = (z - z.max()) / T          # 减最大值：数值稳定（正文 2.3 节）
    e = np.exp(zz)
    return e / e.sum()

temps = [(0.1, "#c0504d", "T=0.1（低温：接近 argmax）"),
         (1.0, "#e8a33d", "T=1.0（原分布）"),
         (10.0, "#2e6da4", "T=10（高温：趋近均匀）")]

fig, ax = plt.subplots(figsize=(8.6, 5.2), dpi=150)
x = np.arange(len(labels))
width = 0.26
for i, (T, color, name) in enumerate(temps):
    p = softmax_with_T(z, T)
    ax.bar(x + (i - 1) * width, p, width, color=color, alpha=0.88, label=name)
    for xi, pi in zip(x + (i - 1) * width, p):
        ax.text(xi, pi + 0.008, f"{pi:.2f}", ha="center", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10.5)
ax.set_ylabel("概率", fontsize=11)
ax.set_title("softmax 温度缩放：同一组 logits，温度决定分布的“尖”与“平”\n"
             r"$p_i = e^{z_i / T} \,/\, \sum_j e^{z_j / T}$：低温赢者通吃，高温人人有份",
             fontsize=11.5)
ax.legend(fontsize=10)
ax.grid(axis="y", ls="--", alpha=0.4)
fig.tight_layout()
fig.savefig(__file__.replace("gen_softmax_temperature.py", "softmax_temperature.png"),
            bbox_inches="tight")
print("saved: softmax_temperature.png")
for T, _, _ in temps:
    p = softmax_with_T(z, T)
    print(f"T={T:>4}: ", np.round(p, 3))

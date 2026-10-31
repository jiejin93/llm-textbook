# -*- coding: utf-8 -*-
"""激活函数族谱：Sigmoid / Tanh / ReLU / GELU / SiLU 的函数曲线与导数曲线双面板。
由代码生成，脚本同目录可复现：python gen_activations.py
依赖：matplotlib, numpy, scipy（erf）
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

x = np.linspace(-5, 5, 500)

sigmoid = 1 / (1 + np.exp(-x))
tanh = np.tanh(x)
relu = np.maximum(x, 0)
phi = 0.5 * (1 + erf(x / np.sqrt(2)))          # 标准正态 CDF
gelu = x * phi
silu = x * sigmoid

d_sigmoid = sigmoid * (1 - sigmoid)
d_tanh = 1 - tanh ** 2
d_relu = (x > 0).astype(float)
d_gelu = phi + x * np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi)   # Φ(x) + x·φ(x)
d_silu = sigmoid * (1 + x * (1 - sigmoid))

funcs = [
    ("Sigmoid", sigmoid, d_sigmoid, "#c0504d"),
    ("Tanh", tanh, d_tanh, "#e8a33d"),
    ("ReLU", relu, d_relu, "#7f7f7f"),
    ("GELU", gelu, d_gelu, "#2e6da4"),
    ("SiLU(Swish)", silu, d_silu, "#3a9d5d"),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 5.0), dpi=150)
for name, f, df, c in funcs:
    ax1.plot(x, f, lw=2.0, color=c, label=name)
    ax2.plot(x, df, lw=2.0, color=c, label=name + " 的导数")

ax1.axhline(0, color="#888888", lw=0.8)
ax1.axvline(0, color="#888888", lw=0.8)
ax1.set_xlabel("输入 $x$", fontsize=11)
ax1.set_ylabel("输出 $f(x)$", fontsize=11)
ax1.set_title("(a) 函数曲线：饱和 vs 不饱和", fontsize=11.5)
ax1.legend(fontsize=9.5)
ax1.grid(ls="--", alpha=0.4)
ax1.set_ylim(-1.3, 5.5)

ax2.axhline(1, color="#bbbbbb", lw=1.0, ls="--")
ax2.text(-4.8, 1.06, "导数 = 1：梯度原样通过", fontsize=9, color="#888888")
ax2.axhline(0, color="#888888", lw=0.8)
ax2.set_xlabel("输入 $x$", fontsize=11)
ax2.set_ylabel(r"导数 $f'(x)$", fontsize=11)
ax2.set_title("(b) 导数曲线：Sigmoid/Tanh 两端趋零 = 梯度消失", fontsize=11.5)
ax2.legend(fontsize=9.5)
ax2.grid(ls="--", alpha=0.4)
ax2.set_ylim(-0.15, 1.25)

fig.suptitle("激活函数族谱：从饱和（Sigmoid/Tanh）到 ReLU 一族再到平滑的 GELU/SiLU",
             fontsize=13, fontweight="bold", y=1.03)
fig.tight_layout()
fig.savefig(__file__.replace("gen_activations.py", "activations.png"),
            bbox_inches="tight")
print("saved: activations.png")

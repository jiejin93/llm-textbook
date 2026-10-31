# -*- coding: utf-8 -*-
"""LayerNorm / RMSNorm 对同一真实向量的归一化效果对比（三面板条形图）。
向量构造：d=64，主体为标准正态 + 整体均值偏移 +2.0 + 一个尖峰 8.0
（模拟 Transformer 中间激活常见的"偏置漂移 + 离群维度"）。
由代码生成，脚本同目录可复现：python gen_norm_demo.py
依赖：matplotlib, numpy
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(0)
d = 64
x = rng.normal(0, 1, d) + 2.0          # 均值偏移 +2
x[37] += 8.0                           # 一个离群维度（尖峰）

eps = 1e-5
mu = x.mean()
var = ((x - mu) ** 2).mean()
ln = (x - mu) / np.sqrt(var + eps)     # LayerNorm：减均值、除标准差

rms = np.sqrt((x ** 2).mean() + eps)
rmsn = x / rms                         # RMSNorm：只除均方根

panels = [
    (x, "原始向量", "#c0504d"),
    (ln, "LayerNorm 之后", "#2e6da4"),
    (rmsn, "RMSNorm 之后", "#3a9d5d"),
]

fig, axes = plt.subplots(3, 1, figsize=(9.6, 7.6), dpi=150, sharex=True,
                         gridspec_kw={"hspace": 0.42})
for ax, (v, title, color) in zip(axes, panels):
    ax.bar(np.arange(d), v, color=color, alpha=0.85, width=1.0)
    m, r = v.mean(), np.sqrt((v ** 2).mean())
    ax.axhline(m, color="#222222", ls="--", lw=1.1)
    ax.text(0.99, 0.86, f"$\\mu$={m:+.3f}    RMS={r:.3f}    $\\sigma$={v.std():.3f}",
            transform=ax.transAxes, ha="right", fontsize=10.5,
            bbox=dict(fc="white", ec="#999999", alpha=0.85))
    ax.set_title(title, fontsize=11.5, loc="left")
    ax.grid(axis="y", ls="--", alpha=0.35)

axes[-1].set_xlabel("特征维度（d=64，第 37 维注入 8.0 尖峰）", fontsize=11)
axes[-1].set_xlim(-1, d)
fig.suptitle("同一个向量过 LayerNorm / RMSNorm：两者都把尺度钉住，"
             "RMSNorm 保留均值（少一次 reduce、少一组 $\\beta$）",
             fontsize=12.5, fontweight="bold", y=0.965)
fig.savefig(__file__.replace("gen_norm_demo.py", "norm_demo.png"), bbox_inches="tight")
print("saved: norm_demo.png")
for v, name, _ in panels:
    print(f"{name}: mu={v.mean():+.4f}, rms={np.sqrt((v**2).mean()):.4f}, std={v.std():.4f}")

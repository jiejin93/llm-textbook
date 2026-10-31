# -*- coding: utf-8 -*-
"""生成 14.3.5 节图：PPO clipped 目标函数在 (ratio, 优势) 平面上的形状（ε=0.2 真实绘制）。

L_clip(rho, A) = min( rho * A, clip(rho, 1-eps, 1+eps) * A )
  左图 A = +1（好动作，想上推概率）：rho 超过 1+eps 后目标被封顶，梯度归零；
  右图 A = -1（坏动作，想压低概率）：rho 低于 1-eps 后对称封底。
对照虚线为无 clip 的朴素目标 rho * A——它会无限奖励"一步涨到天上"。

运行：python gen_ppo_clip.py   （依赖 numpy + matplotlib）
输出：gen_ppo_clip.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

eps = 0.2
rho = np.linspace(0.4, 1.9, 400)


def l_clip(r, A):
    return np.minimum(r * A, np.clip(r, 1 - eps, 1 + eps) * A)


fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)

for ax, A, title in [
    (axes[0], +1.0, "(a) 正优势 $A=+1$（好动作：上推概率）"),
    (axes[1], -1.0, "(b) 负优势 $A=-1$（坏动作：压低概率）"),
]:
    ax.plot(rho, rho * A, ls="--", lw=1.6, color="gray", label="无 clip：$\\rho A$")
    ax.plot(rho, l_clip(rho, A), lw=2.6, color="tab:blue", label="clip 目标：$\\min(\\rho A,\\ \\mathrm{clip}(\\rho,\\ 1{-}\\epsilon,\\ 1{+}\\epsilon)A)$")
    ax.axvspan(1 - eps, 1 + eps, color="tab:blue", alpha=0.08)
    ax.axvline(1 - eps, color="k", lw=0.8, ls=":")
    ax.axvline(1 + eps, color="k", lw=0.8, ls=":")
    ax.set_xlabel("概率比 $\\rho_t = \\pi_\\theta / \\pi_{\\theta_{old}}$")
    ax.set_title(title, fontsize=11)

axes[0].annotate("置信区间\n$[1{-}\\epsilon,\\ 1{+}\\epsilon]$", xy=(1.0, -1.55), fontsize=9, ha="center", color="tab:blue")
axes[0].annotate("封顶：$\\rho>1{+}\\epsilon$ 后\n目标不再增长，梯度=0", xy=(1.45, 1.2), xytext=(1.42, 0.72),
                 fontsize=9, arrowprops=dict(arrowstyle="->", lw=1))
axes[1].annotate("对称封底：$\\rho<1{-}\\epsilon$ 后\n继续压低也没有额外收益", xy=(0.62, -0.8), xytext=(0.52, -0.35),
                 fontsize=9, arrowprops=dict(arrowstyle="->", lw=1))
for ax in axes:
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8.5, loc="upper left")
axes[0].set_ylabel("截断代理目标 $L^{CLIP}(\\rho, A)$")

fig.suptitle("PPO 的 clip 机制：把单步更新限制在置信区间附近（$\\epsilon=0.2$，真实绘制）", fontsize=11.5, y=1.02)
fig.tight_layout()
fig.savefig("gen_ppo_clip.png", dpi=150, bbox_inches="tight")
print("已保存 gen_ppo_clip.png")

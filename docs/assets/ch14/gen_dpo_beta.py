# -*- coding: utf-8 -*-
"""生成 14.5.1 节图：DPO 中不同 β 的损失函数族与"认对概率"（真实绘制）。

记偏好对的似然比差（log-prob margin）为
    m = log[pi_theta(y_w)/pi_ref(y_w)] - log[pi_theta(y_l)/pi_ref(y_l)]
隐式奖励差 u = beta * m，DPO 损失 = -log sigmoid(u)。
  左图：beta 越小，同一 margin 下损失越大、梯度越强 —— "推得更狠"，KL 约束越松；
  右图：sigma(u) 是模型把 chosen 排在 rejected 之上的概率（隐式 BT 概率）。

运行：python gen_dpo_beta.py   （依赖 numpy + matplotlib）
输出：gen_dpo_beta.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

m = np.linspace(-2, 12, 400)          # 似然比差 m
betas = [0.05, 0.1, 0.3, 1.0]
colors = ["#B279A2", "#4C78A8", "#F58518", "#54A24B"]

def dpo_loss(m, beta):
    return -np.log(1.0 / (1.0 + np.exp(-beta * m)))   # -log sigmoid(beta*m)

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))

ax = axes[0]
for beta, c in zip(betas, colors):
    ax.plot(m, dpo_loss(m, beta), lw=2.2, color=c, label=f"$\\beta$ = {beta}")
ax.axhline(np.log(2), color="gray", ls=":", lw=1)
ax.text(-1.6, np.log(2) + 0.06, "$m=0$ 时 $-\\log\\sigma(0)=\\log 2$（分不出好坏）", fontsize=8.5, color="gray")
ax.set_xlabel("似然比差 $m$（好答案相对坏答案的 log 概率差距）")
ax.set_ylabel("DPO 损失 $-\\log\\sigma(\\beta m)$")
ax.set_title("(a) 损失族：$\\beta$ 越小，同一差距下推力越强", fontsize=11)
ax.grid(alpha=0.3)
ax.legend(fontsize=9)

ax = axes[1]
for beta, c in zip(betas, colors):
    ax.plot(m, 1.0 / (1.0 + np.exp(-beta * m)), lw=2.2, color=c, label=f"$\\beta$ = {beta}")
    y_at_5 = 1.0 / (1.0 + np.exp(-beta * 5))
    ax.scatter([5], [y_at_5], color=c, s=28, zorder=5)
ax.axvline(5, color="k", ls=":", lw=1)
ax.annotate("$m=5$ 处：$\\beta=0.05$ 仍只有 0.62，\n损失继续推大 margin；\n$\\beta=1$ 已 0.99，几乎不再更新",
            xy=(5, 0.8), xytext=(5.6, 0.42), fontsize=9,
            arrowprops=dict(arrowstyle="->", lw=1))
ax.set_xlabel("似然比差 $m$")
ax.set_ylabel("认对概率 $\\sigma(\\beta m)$")
ax.set_title("(b) 隐式 BT 概率：$\\beta$ 越大越早饱和（越保守）", fontsize=11)
ax.grid(alpha=0.3)
ax.legend(fontsize=9, loc="lower right")

fig.suptitle("DPO 的 $\\beta$ 旋钮：KL 锚定强度决定\"推多狠\"与\"多稳\"（真实绘制）", fontsize=11.5, y=1.02)
fig.tight_layout()
fig.savefig("gen_dpo_beta.png", dpi=150, bbox_inches="tight")

for beta in betas:
    print(f"beta={beta:<5} m=5 时 损失={dpo_loss(5.0, beta):.3f}  认对概率={1/(1+np.exp(-beta*5)):.3f}")
print("已保存 gen_dpo_beta.png")

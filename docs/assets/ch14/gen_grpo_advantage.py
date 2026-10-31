# -*- coding: utf-8 -*-
"""生成 14.6.1 节图：GRPO 组采样相对优势（对一组真实采样奖励做标准化）。

模拟同一 prompt 采样 G=8 个回答（奖励由固定种子的随机数真实产生）：
  情形 1（混合组）：部分答对部分答错 -> 标准化优势 (r - mean) / (std + eps)
                    答对的拿到正优势、答错的负优势，critic 被"组内统计量"替代；
  情形 2（全对组）：组内奖励全为 1 -> std=0、优势全 0，这组样本梯度无效
                    —— 正是 DAPO"动态采样"要过滤的对象。

运行：python gen_grpo_advantage.py   （依赖 numpy + matplotlib）
输出：gen_grpo_advantage.png + 各样本优势数值（stdout）
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(42)
G = 8
eps = 1e-4

def group_adv(rewards):
    r = np.asarray(rewards, dtype=float)
    return (r - r.mean()) / (r.std() + eps), r.mean(), r.std()

# 情形 1：混合组（模拟答对率约 0.375）
r1 = rng.binomial(1, 0.375, size=G)
a1, mu1, sd1 = group_adv(r1)
if r1.std() == 0:            # 极小概率全同，强制换种子重采
    rng = np.random.default_rng(7)
    r1 = rng.binomial(1, 0.375, size=G)
    a1, mu1, sd1 = group_adv(r1)

# 情形 2：全对组（太简单的题，全答对）
r2 = np.ones(G)
a2, mu2, sd2 = group_adv(r2)

print(f"混合组奖励 = {r1.tolist()}，mean = {mu1:.3f}，std = {sd1:.3f}")
print(f"混合组标准化优势 = {np.round(a1, 2).tolist()}")
print(f"全对组奖励 = {r2.tolist()}，mean = {mu2:.3f}，std = {sd2:.3f}，优势 = {np.round(a2, 2).tolist()}")

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), sharey=True)
x = np.arange(G)

ax = axes[0]
ax.bar(x - 0.2, r1, 0.38, color="#B8CCE4", label="奖励 $r_i$（0/1）")
ax.bar(x + 0.2, a1, 0.38, color="tab:blue", label="标准化优势 $\\hat A_i$")
ax.axhline(mu1, color="gray", ls="--", lw=1.2)
ax.text(G - 0.4, mu1 + 0.06, f"组均值 {mu1:.2f}（baseline）", fontsize=9, ha="right", color="gray")
ax.set_ylim(-1.7, 1.7)
ax.set_xticks(x)
ax.set_xlabel("组内回答编号 $i$（同一 prompt 采样 $G$=8 次）")
ax.set_title("(a) 混合组：答对比组均值好就加强、差就抑制", fontsize=11)
ax.legend(fontsize=9, loc="lower left")

ax = axes[1]
ax.bar(x - 0.2, r2, 0.38, color="#E4D3B8", label="奖励 $r_i$（0/1）")
ax.bar(x + 0.2, a2, 0.38, color="tab:orange", label="标准化优势 $\\hat A_i$")
ax.text(3.5, 0.3, "std=0 → 优势全为 0 → 这组样本梯度无效\n（DAPO 动态采样要过滤的对象）",
        fontsize=9.5, ha="center",
        bbox=dict(boxstyle="round,pad=0.35", fc="#FFF3E0", ec="tab:orange"))
ax.set_ylim(-1.7, 1.7)
ax.set_xticks(x)
ax.set_xlabel("组内回答编号 $i$")
ax.set_title("(b) 全对组：组内无差异，替代 critic 的基线失效", fontsize=11)
ax.legend(fontsize=9, loc="lower left")

fig.suptitle("GRPO 的组内标准化优势：用一组真实采样奖励的统计量替代 critic",
             fontsize=11.5, y=1.02)
fig.tight_layout()
fig.savefig("gen_grpo_advantage.png", dpi=150, bbox_inches="tight")
print("已保存 gen_grpo_advantage.png")

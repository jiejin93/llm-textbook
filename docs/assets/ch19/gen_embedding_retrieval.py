# -*- coding: utf-8 -*-
"""图①：2D 嵌入空间中的向量检索可视化（toy 数据，真实计算余弦相似度）。

生成 gen_embedding_retrieval.png：文档向量散点 + 查询向量 + top-3 圈选
+ 查询与两个文档夹角的余弦示意。脚本同目录可复现：
    python gen_embedding_retrieval.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(19)

# ---- 1) 构造三个主题的 toy 文档向量（真实计算余弦） --------------------------
centers = {
    "运维文档": np.array([2.4, 2.1]),
    "美食评论": np.array([-2.3, 1.9]),
    "球赛战报": np.array([0.2, -2.4]),
}
docs, labels = [], []
for name, c in centers.items():
    for _ in range(6):
        docs.append(c + rng.normal(scale=0.55, size=2))
        labels.append(name)
docs = np.array(docs)

query = np.array([1.6, 3.1])          # 查询：“服务器 CPU 打满了怎么办”

# ---- 2) 余弦相似度与 top-3 检索 -------------------------------------------
norm = lambda v: v / np.linalg.norm(v)
dq = norm(query)
cos = docs @ dq / np.linalg.norm(docs, axis=1)     # 全部余弦相似度
order = np.argsort(-cos)
top3 = order[:3]

# ---- 3) 画图 ---------------------------------------------------------------
colors = {"运维文档": "#1f77b4", "美食评论": "#d62728", "球赛战报": "#2ca02c"}
fig, ax = plt.subplots(figsize=(8.6, 7.2), dpi=150)

for name, c in centers.items():
    ax.scatter(*docs[np.array(labels) == name].T, s=64, c=colors[name],
               label=f"{name}（6 篇）", alpha=0.85, edgecolors="white", zorder=3)

# 查询向量与最近/最远文档的向量箭头
nearest, farthest = order[0], order[-1]
for idx, c in [(nearest, "#ff7f0e"), (farthest, "#7f7f7f")]:
    v = docs[idx]
    ax.annotate("", xy=v, xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=c, lw=1.4, alpha=0.75))
ax.annotate("", xy=query, xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color="black", lw=2.2))

# 余弦角示意：theta_1 = 查询与最近文档的夹角，theta_2 = 查询与最远文档的夹角
ang = lambda v: float(np.degrees(np.arctan2(norm(v)[1], norm(v)[0]))) % 360
a_q, a_n, a_f = ang(query), ang(docs[nearest]), ang(docs[farthest])
t1a, t1b = sorted([a_q, a_n])
ax.add_patch(Arc((0, 0), 3.4, 3.4, angle=0, theta1=t1a, theta2=t1b,
                 color="#ff7f0e", lw=2))
am = (a_q + a_n) / 2
ax.text(np.cos(np.radians(am)) * 2.05, np.sin(np.radians(am)) * 2.05,
        f"$\\theta_1$={t1b - t1a:.0f}°，cos$\\theta_1$={cos[nearest]:.2f}",
        color="#ff7f0e", fontsize=11)
t2a, t2b = sorted([a_q, a_f])
ax.add_patch(Arc((0, 0), 4.4, 4.4, angle=0, theta1=t2a, theta2=t2b,
                 color="#7f7f7f", lw=2))
am2 = (t2a + t2b) / 2
ax.text(np.cos(np.radians(am2)) * 2.9, np.sin(np.radians(am2)) * 2.9 + 0.55,
        f"$\\theta_2$={t2b - t2a:.0f}°，cos$\\theta_2$={cos[farthest]:.2f}",
        color="#7f7f7f", fontsize=11, ha="center")

# top-3 圈选
mu = docs[top3].mean(axis=0)
r = np.max(np.linalg.norm(docs[top3] - mu, axis=1)) + 0.5
ax.add_patch(Circle(mu, r, fill=False, color="black", ls="--", lw=1.8))
ax.text(mu[0] + 0.1, mu[1] - r - 0.5, "top-3 召回", fontsize=12, weight="bold")

ax.scatter(*query, marker="*", s=430, c="black", zorder=5)
ax.annotate("查询向量 q\n“服务器 CPU 打满了怎么办”", xy=query, xytext=(-3.6, 3.8),
            fontsize=11, arrowprops=dict(arrowstyle="->", lw=1.2))

ax.axhline(0, color="0.8", lw=0.6)
ax.axvline(0, color="0.8", lw=0.6)
ax.set_xlim(-3.8, 4.6)
ax.set_ylim(-3.6, 4.4)
ax.set_xlabel("嵌入维度 1（toy 2D，真实模型为 512~1024 维）")
ax.set_ylabel("嵌入维度 2")
ax.set_title("嵌入空间检索：语义相近的文档向量彼此靠近，\n"
             "查询按余弦相似度圈选 top-k（图中数值由本脚本真实计算）")
ax.legend(loc="lower right", fontsize=10)
fig.tight_layout()
fig.savefig(__file__.replace("gen_embedding_retrieval.py", "embedding_retrieval.png"),
            dpi=150, bbox_inches="tight")
print("cos(查询, 最近文档) =", round(float(cos[nearest]), 3))
print("cos(查询, 最远文档) =", round(float(cos[farthest]), 3))
print("top-3 索引:", top3.tolist(), "其余弦值:", np.round(cos[top3], 3).tolist())
print("已保存 embedding_retrieval.png")

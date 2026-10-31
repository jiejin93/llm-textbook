# -*- coding: utf-8 -*-
"""图③：混合检索的 RRF 倒数排名融合——BM25 与稠密检索两路名次的合并前后对比。

候选 8 篇文档，两路检索各给一份排名（真实场景中两路各自独立打分，量纲不可加），
RRF 只看名次融合。斜率图左端为 BM25 名次，右端为融合后名次。
脚本同目录可复现：python gen_rrf_merge.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ---- 1) 场景：查询“苹果发布会 新品 头显”，库内 8 篇候选 ----------------------
docs = {
    "d1": "苹果秋季发布会官宣：新一代头显亮点汇总",
    "d2": "发布会前瞻：新品头显的供应链消息",
    "d3": "苹果公司财报电话会纪要（无新品信息）",
    "d4": "国产头显新品评测：对比主流设备",
    "d5": "水果种植技术：苹果树四季修剪指南",
    "d6": "历代发布会经典时刻回顾",
    "d7": "头显设备的光学原理科普",
    "d8": "苹果新品发布会门票与直播观看方式",
}
# 两路检索的排名（BM25 按词面精确匹配，稠密检索按语义；两者名次真实地不一致）
rank_bm25  = ["d8", "d1", "d6", "d5", "d2", "d3", "d4", "d7"]
rank_dense = ["d2", "d7", "d1", "d4", "d8", "d3", "d6", "d5"]

# ---- 2) RRF 倒数排名融合（k=60） --------------------------------------------
def rrf(rankings, k=60):
    score = {d: 0.0 for d in docs}
    for r in rankings:
        for pos, d in enumerate(r):
            score[d] += 1.0 / (k + pos + 1)     # rank 从 1 计
    return score

score = rrf([rank_bm25, rank_dense])
rank_fused = sorted(docs, key=lambda d: (-score[d], d))
print("RRF 得分（k=60）:")
for d in rank_fused:
    i, j = rank_bm25.index(d) + 1, rank_dense.index(d) + 1
    print(f"  {d}: BM25 第{i} + 稠密 第{j} -> RRF={1/(60+i)+1/(60+j):.5f}")

# ---- 3) 斜率图：左 BM25 名次 -> 右融合后名次 --------------------------------
fig, ax = plt.subplots(figsize=(10.4, 5.8), dpi=150)
for d in docs:
    y0, y1 = rank_bm25.index(d), rank_fused.index(d)
    top3 = d in rank_fused[:3]
    rose = y1 < y0
    color = ("#2ca02c" if top3 and rose else "#d62728" if top3 else "#bbbbbb")
    lw = 2.4 if top3 else 1.1
    ax.plot([0, 1], [y0, y1], "-", color=color, lw=lw, alpha=0.9, zorder=2)
    ax.scatter([0, 1], [y0, y1], s=90 if top3 else 45, color=color, zorder=3)
    ax.text(-0.04, y0, f"{d}  {docs[d]}", ha="right", va="center", fontsize=9.5,
            color="black" if top3 else "0.45", weight="bold" if top3 else "normal")
    ax.text(1.04, y1, f"{d}｜RRF={score[d]:.5f}", ha="left", va="center",
            fontsize=9.5, color="black" if top3 else "0.45",
            weight="bold" if top3 else "normal")

ax.set_xticks([0, 1])
ax.set_xticklabels(["BM25 单路排名", "RRF 融合后排名（并参考稠密路）"], fontsize=12)
ax.set_yticks(range(8))
ax.set_yticklabels([f"第 {i+1} 名" for i in range(8)])
ax.invert_yaxis()
ax.set_xlim(-2.9, 1.35)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
ax.set_title("混合检索：BM25 与稠密检索分数量纲不同不可相加，RRF 只看名次融合\n"
             "绿=融合后进 top-3 且名次上升，红=进 top-3 但名次下降（本脚本真实计算，k=60）",
             fontsize=11.5)
ax.grid(alpha=0.25, axis="y")
fig.tight_layout()
fig.savefig(__file__.replace("gen_rrf_merge.py", "rrf_merge.png"),
            dpi=150, bbox_inches="tight")
print("已保存 rrf_merge.png")

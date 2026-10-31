# -*- coding: utf-8 -*-
"""真实开源 MoE 的"总参数-激活参数"散点图（对数轴），对角线为稠密模型基线。
数据取自本章 9.5 演进对比表与第 7 章架构总表（均为官方报告数字）。
运行：python gen_moe_scale.py，输出 moe_scale.png。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# (名称, 总参数/B, 激活参数/B, 年份, 标签偏移px)
models = [
    ("Mixtral-8x7B",    46.7, 12.9, 2023, (6, 4)),
    ("DeepSeekMoE-16B", 16.4, 2.8,  2024, (6, -12)),
    ("DeepSeek-V2",     236,  21,   2024, (10, 10)),
    ("Qwen3-30B-A3B",   30.5, 3.3,  2025, (6, 6)),
    ("DeepSeek-V3",     671,  37,   2024, (6, 6)),
    ("Qwen3-235B-A22B", 235,  22,   2025, (-108, -6)),
    ("GLM-4.5",         355,  32,   2025, (6, 6)),
    ("Kimi K2",         1040, 32,   2025, (6, -13)),
]

fig, ax = plt.subplots(figsize=(8.4, 5.4), dpi=140)

import numpy as np
diag = np.logspace(0.5, 3.2, 50)
ax.plot(diag, diag, ls="--", color="gray", lw=1.2, label="稠密模型（激活 = 总参）")

for name, total, act, year, off in models:
    ax.scatter(total, act, s=70, color="#4C72B0", edgecolor="black",
               linewidth=0.6, zorder=3)
    ax.annotate(f"{name}\n稀疏比 {act / total:.0%}", (total, act),
                textcoords="offset points", xytext=off, fontsize=8.2, zorder=4)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(10, 3000)
ax.set_ylim(1.5, 2000)
ax.set_xlabel("总参数（B，对数轴）——决定显存账单")
ax.set_ylabel("激活参数（B，对数轴）——决定每 token 算力账单")
ax.set_title("开源 MoE 的总参数 vs 激活参数：离对角线越远，稀疏比越低", fontsize=11.5)
ax.grid(True, which="both", ls="--", lw=0.35, alpha=0.45)
ax.legend(fontsize=9, loc="upper left")

fig.tight_layout()
fig.savefig("moe_scale.png", dpi=140)
for name, total, act, _, _ in models:
    print(f"{name:18s} 总参 {total:7.1f} B  激活 {act:5.1f} B  稀疏比 {act / total:5.1%}")
print("已保存 moe_scale.png")

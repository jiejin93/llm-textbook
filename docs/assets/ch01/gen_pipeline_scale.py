# -*- coding: utf-8 -*-
"""训练一个 LLM 的完整流水线：阶段-数据规模-算力 对数坐标图。
由代码生成，脚本同目录可复现：python gen_pipeline_scale.py
依赖：matplotlib >= 3.5, numpy
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

fig = plt.figure(figsize=(11, 6.8), dpi=150)
gs = fig.add_gridspec(2, 2, height_ratios=[0.62, 1], hspace=0.42, wspace=0.26)

# ---------- (a) 三阶段流程框图 ----------
ax0 = fig.add_subplot(gs[0, :])
ax0.set_xlim(0, 10)
ax0.set_ylim(0, 1.6)
ax0.axis("off")
stages = [
    ("① 预训练", "下一词交叉熵\n→ 基座模型(base)"),
    ("② SFT 监督微调", "模仿示范回答\n→ 指令模型(instruct)"),
    ("③ 偏好优化", "RLHF / DPO\n→ 对齐模型(chat)"),
]
inputs = ["网页/代码/书籍\n$\\geq 10^{12}$ token",
          "人工示范问答\n$10^4 \\sim 10^6$ 条",
          "偏好对/可验证奖励\n$10^4 \\sim 10^6$ 对"]
for i, (name, out) in enumerate(stages):
    box = FancyBboxPatch((0.6 + i * 3.2, 0.28), 2.0, 0.95,
                         boxstyle="round,pad=0.08", fc="#dbe9f6", ec="#3b6ea5", lw=1.4)
    ax0.add_patch(box)
    ax0.text(0.6 + i * 3.2 + 1.0, 0.97, name, ha="center", va="center",
             fontsize=11.5, fontweight="bold")
    ax0.text(0.6 + i * 3.2 + 1.0, 0.55, out, ha="center", va="center", fontsize=8.8)
    xin = ax0.text(0.6 + i * 3.2 + 1.0, 1.44, inputs[i], ha="center", va="center",
                   fontsize=8.8, color="#555555")
    if i < 2:
        arrow = FancyArrowPatch((2.75 + i * 3.2, 0.75), (3.55 + i * 3.2, 0.75),
                                arrowstyle="-|>", mutation_scale=16, color="#c05a3e", lw=1.8)
        ax0.add_patch(arrow)
ax0.text(5.0, 0.06, "三阶段共用同一个模型骨架，区别只在数据与目标函数（一次预训练、多次后训练）",
         ha="center", fontsize=9.2, color="#c05a3e")

# ---------- (b) 数据规模（对数坐标） ----------
stages_short = ["① 预训练", "② SFT", "③ 偏好优化"]
data_lo = np.array([1e12, 1e4, 1e4])     # 各阶段典型数据量下界
data_hi = np.array([1e15, 1e6, 1e6])     # 上界
ax1 = fig.add_subplot(gs[1, 0])
x = np.arange(3)
ax1.bar(x, np.log10(data_hi), width=0.52, color="#4c84b8", alpha=0.85,
        label="典型规模上界")
ax1.bar(x, np.log10(data_lo), width=0.52, color="#9dc3e0", alpha=0.9,
        label="典型规模下界")
for i in range(3):
    ax1.text(i, np.log10(data_hi[i]) + 0.12, f"$10^{{{int(np.log10(data_lo[i]))}}}$~"
             f"$10^{{{int(np.log10(data_hi[i]))}}}$",
             ha="center", fontsize=9.5, fontweight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels(stages_short, fontsize=10.5)
ax1.set_ylabel("数据规模（log 尺度，单位：token/条/对）", fontsize=9.5)
ax1.set_yticks([4, 6, 8, 10, 12, 15])
ax1.set_yticklabels(["$10^4$", "$10^6$", "$10^8$", "$10^{10}$", "$10^{12}$", "$10^{15}$"])
ax1.set_ylim(0, 16.4)
ax1.set_title("(b) 各阶段数据规模：预训练比后训练多 6~10 个数量级", fontsize=10.5)
ax1.legend(fontsize=8.5, loc="upper right")
ax1.grid(axis="y", ls="--", alpha=0.4)

# ---------- (c) 训练算力（对数坐标） ----------
# 量级依据：GPT-3 预训练约 3.14e23 FLOPs（arXiv:2005.14165 附录）；
# InstructGPT 的 SFT/RLHF 计算量不足预训练的 3%（arXiv:2203.02155）。
flops = np.array([3.1e23, 3e21, 6e21])
ax2 = fig.add_subplot(gs[1, 1])
bars = ax2.bar(x, np.log10(flops), width=0.52,
               color=["#c0504d", "#e8a33d", "#7f9e5a"], alpha=0.88)
for i, f in enumerate(flops):
    ax2.text(i, np.log10(f) + 0.12, f"$\\approx 10^{{{round(np.log10(f))}}}$ FLOPs",
             ha="center", fontsize=9.5, fontweight="bold")
pct = [">95%", "<1%", "约 1~3%"]
for i, p in enumerate(pct):
    ax2.text(i, 2.1, f"算力占比\n{p}", ha="center", fontsize=9, color="white",
             fontweight="bold")
ax2.set_xticks(x)
ax2.set_xticklabels(stages_short, fontsize=10.5)
ax2.set_ylabel("训练算力（log 尺度，FLOPs）", fontsize=9.5)
ax2.set_yticks([6, 9, 12, 15, 18, 21, 24])
ax2.set_yticklabels(["$10^6$", "$10^9$", "$10^{12}$", "$10^{15}$", "$10^{18}$",
                     "$10^{21}$", "$10^{24}$"])
ax2.set_ylim(0, 26.5)
ax2.set_title("(c) 各阶段算力：能力几乎全部产生于①", fontsize=10.5)
ax2.grid(axis="y", ls="--", alpha=0.4)

fig.suptitle("训练一个 LLM 的完整流水线：三阶段、数据规模与算力（对数坐标）",
             fontsize=13, fontweight="bold", y=0.985)
fig.savefig(__file__.replace("gen_pipeline_scale.py", "pipeline_scale.png"),
            bbox_inches="tight")
print("saved: pipeline_scale.png")

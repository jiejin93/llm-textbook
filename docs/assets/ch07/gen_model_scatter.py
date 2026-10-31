# -*- coding: utf-8 -*-
"""主流开源模型"发布时间-上下文长度-参数量"散点图。
数据全部来自本章 7.2/7.3/7.4/7.6 各表（发布时官方声明的最大上下文，含外推配置；
MoE 模型取总参数）。运行：python gen_model_scatter.py，输出 model_scatter.png。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from datetime import date

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# (名称, 发布日期, 上下文/K tokens, 参数量/B, 家族, 是否MoE, 标签偏移px)
models = [
    ("LLaMA-1-7B",     date(2023, 2, 24), 2,    7,    "LLaMA 系", False, None),
    ("LLaMA-1-65B",    date(2023, 2, 24), 2,    65,   "LLaMA 系", False, None),
    ("LLaMA-2-7B",     date(2023, 7, 18), 4,    7,    "LLaMA 系", False, None),
    ("LLaMA-2-70B",    date(2023, 7, 18), 4,    70,   "LLaMA 系", False, None),
    ("Mistral-7B",     date(2023, 9, 27), 8,    7.3,  "Mistral 系", False, None),
    ("Qwen-72B",       date(2023, 11, 29), 8,   72,   "Qwen 系", False, None),
    ("Mixtral-8x7B",   date(2024, 1, 10), 32,   46.7, "Mistral 系", True, (6, 6)),
    ("LLaMA-3-8B",     date(2024, 4, 18), 8,    8,    "LLaMA 系", False, None),
    ("LLaMA-3-70B",    date(2024, 4, 18), 8,    70,   "LLaMA 系", False, None),
    ("DeepSeek-V2",    date(2024, 5, 6),  128,  236,  "DeepSeek 系", True, (6, -16)),
    ("Qwen2-72B",      date(2024, 6, 6),  32,   72,   "Qwen 系", False, None),
    ("Gemma-2-27B",    date(2024, 6, 27), 8,    27,   "Gemma 系", False, None),
    ("LLaMA-3.1-405B", date(2024, 7, 23), 128,  405,  "LLaMA 系", False, (-10, 9)),
    ("Qwen2.5-72B",    date(2024, 9, 19), 128,  72,   "Qwen 系", False, None),
    ("DeepSeek-V3",    date(2024, 12, 26), 128, 671,  "DeepSeek 系", True, (-30, 9)),
    ("Qwen3-235B-A22B", date(2025, 4, 29), 128, 235,  "Qwen 系", True, (-120, -19)),
    ("GLM-4.5",        date(2025, 7, 28), 128,  355,  "GLM 系", True, (8, 7)),
    ("Kimi K2",        date(2025, 7, 11), 128,  1040, "GLM 系", True, (-28, -20)),
]

family_color = {
    "LLaMA 系": "#4C72B0", "Qwen 系": "#DD8452", "DeepSeek 系": "#55A868",
    "Mistral 系": "#C44E52", "Gemma 系": "#8172B3", "GLM 系": "#937860",
}

fig, ax = plt.subplots(figsize=(10, 5.6), dpi=140)
for name, d, ctx, params, fam, is_moe, offset in models:
    ax.scatter(d, ctx, s=28 + 90 * (params / 1040) ** 0.5 * 4,
               color=family_color[fam], marker="*" if is_moe else "o",
               edgecolors="black", linewidths=0.5, zorder=3, alpha=0.9)
    if offset:
        ax.annotate(name, (d, ctx), textcoords="offset points", xytext=offset,
                    fontsize=8.5, color="#222222", zorder=4)

ax.set_yscale("log")
ax.set_yticks([2, 4, 8, 32, 128])
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_ylabel("上下文长度（K tokens，对数轴）")
ax.set_xlabel("发布时间")
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.set_ylim(1.3, 400)
ax.grid(True, which="both", axis="y", ls="--", lw=0.4, alpha=0.5)
ax.set_title("主流开源模型：发布时间 × 上下文长度 × 参数量（点越大参数越多，星号 = MoE）",
             fontsize=11.5)

legend_items = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                       markersize=8, label=f) for f, c in family_color.items()]
legend_items += [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#bbbbbb",
           markeredgecolor="black", markersize=7, label="稠密模型"),
    Line2D([0], [0], marker="*", color="w", markerfacecolor="#bbbbbb",
           markeredgecolor="black", markersize=11, label="MoE（总参数）"),
]
ax.legend(handles=legend_items, loc="upper left", fontsize=8, ncol=2, framealpha=0.9)

fig.tight_layout()
fig.savefig("model_scatter.png", dpi=140)
print("已保存 model_scatter.png")

# -*- coding: utf-8 -*-
"""
同一 7B 模型（n=8 卡）每卡每步通信字节数对比（按本章 12.2/12.5/12.10 公式真实计算）：
  环形 AllReduce: 2(n-1)/n * S；AllGather / ReduceScatter 各 (n-1)/n * S
  DP 系：对 2N 字节梯度做规约；ZeRO-3：3 次（前向 AG + 反向 AG + 梯度 RS）
  TP：每层 4 次 AllReduce，每次 S = 2BTd（激活），共 4L 次
运行：python gen_comm_7b.py  （依赖 matplotlib）
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

N, n = 7e9, 8
T, B, d, L = 2048, 1, 4096, 32
G = 1e9

ar = 2*(n-1)/n          # AllReduce 每卡系数 1.75
ag = (n-1)/n            # AllGather / ReduceScatter 每卡系数 0.875
grad = 2*N              # bf16 梯度字节数
tp_once = ar * 2*B*T*d  # 一次激活 AllReduce 每卡字节

configs = [
    ("朴素 DP\n1 次梯度 AllReduce",        ar*grad,                "1 次梯度 AllReduce"),
    ("ZeRO-1\n梯度分桶 RS+AG",             ar*grad,                "梯度分桶 RS+AG，总量不变"),
    ("ZeRO-2\n梯度 RS+权重 AG",            ar*grad,                "RS 梯度 + AG 更新后权重"),
    ("ZeRO-3\n2×AG+1×RS（+50%）",          3*ag*grad,              "前向 AG + 反向 AG + 梯度 RS"),
    ("TP8\n每层 4 次小 AllReduce",         4*L*tp_once,            "每层 4 次小 AllReduce（激活）"),
    ("TP8 + ZeRO-1\n激活通信 + 梯度通信",  4*L*tp_once + ar*grad,  "TP 激活 + ZeRO-1 梯度"),
]
names = [c[0] for c in configs]
vals  = [c[1]/G for c in configs]

fig, ax = plt.subplots(figsize=(11, 6.2))
bars = ax.bar(range(len(names)), vals, 0.55,
              color=["#9bb8d8", "#1f77b4", "#1f77b4", "#d62728", "#2ca02c", "#7f4fc9"])
for i, (r, v) in enumerate(zip(bars, vals)):
    ax.text(r.get_x()+r.get_width()/2, v + 0.6, f"{v:.1f} GB",
            ha="center", fontsize=11.5, fontweight="bold")

ax.annotate("总量 +50%：3×0.875N 对比 2×0.875N", xy=(3, 36.75), xytext=(1.7, 41),
            fontsize=10.5, color="#d62728", arrowprops=dict(arrowstyle="->", color="#d62728"))

ax.set_ylabel("每卡每步通信量（GB，bf16 计价）", fontsize=12)
ax.set_ylim(0, 48)
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, fontsize=10.5)
ax.set_title(f"7B 模型 n={n} 卡一步训练的每卡通信量（按本章公式计算）",
             fontsize=12.5)
ax.grid(axis="y", alpha=0.3)
fig.text(0.5, -0.015,
         "解读：TP 通信量随激活（B、T、d、L）不随 N，小包高频、延迟敏感 → 只放机内 NVLink；"
         "ZeRO 系通信量随 N，流式大包可预取/分桶重叠 → 可跨机。",
         ha="center", fontsize=10, color="#444444")
fig.tight_layout()
fig.savefig("gen_comm_7b.png", dpi=150, bbox_inches="tight")
for nm, v, note in zip(names, vals, configs):
    print(f"{nm}: {v:.2f} GB  ({note[1]})")
print("已保存 gen_comm_7b.png")

# -*- coding: utf-8 -*-
"""
7B 模型在各并行方案下每卡显存柱状图（按本章公式真实计算，n=8 卡）：
  模型状态：混合精度 AdamW 16 字节/参数（ZeRO 按 12.3 的公式切分，TP 按卡数切分）
  激活：每层 34TBd + 5hT^2B 字节（Korthikanti et al., 2022），TP 方案按 SP 再除以 TP 度数
运行：python gen_mem_7b.py  （依赖 matplotlib）
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

N = 7e9
n = 8
T, B, d, h, L = 2048, 1, 4096, 32, 32
act_per_layer = 34*T*B*d + 5*h*T*T*B          # 字节/层（bf16）
act_full = L * act_per_layer / 1e9            # GB
act_sp = act_full / n                          # Megatron SP 下除以 TP 度数

configs = [
    ("朴素 DP",            16*N/n**0, act_full,  False),   # 每卡全量
    ("ZeRO-1",             4*N + 12*N/n, act_full, False),
    ("ZeRO-2",             2*N + 14*N/n, act_full, False),
    ("ZeRO-3 / FSDP",      16*N/n, act_full,  False),
    ("TP8 + ZeRO1 + SP",   16*N/n, act_sp,    True),
]
names  = [c[0] for c in configs]
states = [c[1]/1e9 for c in configs]
acts   = [c[2] for c in configs]

fig, ax = plt.subplots(figsize=(10.5, 6))
x = range(len(names))
bars1 = ax.bar(x, states, 0.55, label="模型状态（16 字节/参数，按方案切分）", color="#1f77b4")
bars2 = ax.bar(x, acts, 0.55, bottom=states, label=f"激活（$34TBd+5hT^2B$/层，T={T},B={B}）", color="#ffA15c")
for i, (s, a) in enumerate(zip(states, acts)):
    ax.text(i, s + a + 2.5, f"{s+a:.1f} GB", ha="center", fontsize=11.5, fontweight="bold")
    ax.text(i, s/2, f"{s:.1f}", ha="center", fontsize=9.5, color="white")
    ax.text(i, s + a/2, f"{a:.1f}", ha="center", fontsize=9.5)

ax.axhline(80, color="#d62728", ls="--", lw=1.3)
ax.text(4.42, 82, "H100 80GB", color="#d62728", fontsize=10, ha="right")
ax.axhline(12, color="#2ca02c", ls="--", lw=1.3)
ax.text(4.42, 14, "RTX 4070Ti 12GB", color="#2ca02c", fontsize=10, ha="right")

ax.set_xticks(list(x))
ax.set_xticklabels(names, fontsize=11.5)
ax.set_ylabel("每卡显存（GB）", fontsize=12)
ax.set_ylim(0, 165)
ax.set_title("7B 模型 8 卡各并行方案的每卡显存（由代码按本章公式计算，脚本同目录可复现）",
             fontsize=12.5)
ax.legend(fontsize=10.5, loc="upper right")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig("gen_mem_7b.png", dpi=150, bbox_inches="tight")
print(f"激活（不开 SP/检查点）每卡 {act_full:.1f} GB；SP 后 {act_sp:.1f} GB")
for nm, s, a in zip(names, states, acts):
    print(f"{nm}: 模型状态 {s:.1f} + 激活 {a:.1f} = {s+a:.1f} GB")
print("已保存 gen_mem_7b.png")

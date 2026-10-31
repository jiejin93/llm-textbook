# -*- coding: utf-8 -*-
"""图：GPU 存储层级的容量与带宽（RTX 4090 vs H100 SXM，对数轴双系列）。

数据来源：
- 寄存器/共享内存容量：CUDA C Programming Guide（计算能力 8.9 / 9.0 每 SM 上限）；
- L2 / HBM 容量与带宽：NVIDIA 官方白皮书（GeForce RTX 4090、H100 SXM）；
- 共享内存带宽 = SM 数 x 128 B/clk x 频率（4090: 128 SM @2.52 GHz, H100: 132 SM @1.98 GHz）；
- 寄存器带宽 = FP32 FMA 峰值 x 2 个源操作数 x 4 B（吞吐下限折算）；
- 主机内存按 DDR5 双通道（消费平台）/ 服务器 8 通道折算。
运行：python gen_memory_hierarchy.py（依赖 numpy + matplotlib，无需 GPU）
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

KB, MB, GB, TB = 2**10, 2**20, 2**30, 2**40


def fmt(v):
    if v >= TB:
        return f"{v/TB:.0f} TB"
    if v >= GB:
        return f"{v/GB:.0f} GB"
    if v >= MB:
        return f"{v/MB:.0f} MB"
    if v >= KB:
        return f"{v/KB:.0f} KB"
    return f"{v} B"


def fmt_bw(v):  # v 单位 GB/s
    return f"{v/1000:.0f} TB/s" if v >= 1000 else f"{v} GB/s"


levels = ["寄存器\n(每SM)", "共享内存/L1\n(每SM)", "L2 缓存\n(全片)", "HBM 显存\n(全片)", "主机内存 DDR\n(系统)"]
# 容量（字节）
cap_4090 = [256 * KB, 128 * KB, 72 * MB, 24 * GB, 64 * GB]
cap_h100 = [256 * KB, 228 * KB, 50 * MB, 80 * GB, 2 * TB]
# 带宽（GB/s）：寄存器/共享内存为折算值，L2 为公开量级，HBM 为官方值
bw_4090 = [330_000, 41_000, 10_000, 1_008, 96]
bw_h100 = [268_000, 33_000, 10_000, 3_350, 307]

x = np.arange(len(levels))
w = 0.38
c1, c2 = "#1976d2", "#e65100"

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), dpi=150)

for ax, va, vb, unit, note in [
    (axes[0], cap_4090, cap_h100, "字节", "寄存器/共享内存为每 SM 数值"),
    (axes[1], bw_4090, bw_h100, "GB/s", "寄存器/共享内存为聚合折算值"),
]:
    labeler = fmt if unit == "字节" else fmt_bw
    ax.bar(x - w / 2, va, w, label="RTX 4090", color=c1, zorder=3)
    ax.bar(x + w / 2, vb, w, label="H100 SXM", color=c2, zorder=3)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(levels, fontsize=9)
    ax.set_title(f"{'容量' if unit == '字节' else '带宽'}（{unit}，对数轴）\n{note}", fontsize=11)
    ax.grid(True, axis="y", which="both", alpha=0.3, zorder=0)
    ax.set_ylabel(unit)
    for xi, (a, b) in enumerate(zip(va, vb)):
        ax.text(xi - w / 2, a * 1.18, labeler(a), ha="right", fontsize=7.5, color=c1)
        ax.text(xi + w / 2, b * 1.18, labeler(b), ha="left", fontsize=7.5, color=c2)

axes[1].annotate("HBM→DDR 跨 PCIe：再差 1–2 个数量级",
                 xy=(3.62, 96 * 1.3), xytext=(2.0, 40_000), fontsize=9,
                 arrowprops=dict(arrowstyle="->", color="gray"))
axes[0].legend(loc="upper left", fontsize=10)

fig.suptitle("GPU 存储层级：越往上越快、也越小（RTX 4090 vs H100 SXM）", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.93))
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory_hierarchy.png")
fig.savefig(out, dpi=150)
print("saved:", out)

# 打印折算口径，便于核对
print("共享内存带宽 4090 = 128 SM x 128 B/clk x 2.52 GHz = %.1f TB/s" % (128 * 128 * 2.52e9 / 1e12))
print("共享内存带宽 H100 = 132 SM x 128 B/clk x 1.98 GHz = %.1f TB/s" % (132 * 128 * 1.98e9 / 1e12))
print("寄存器带宽 4090 = (82.6/2) T FMA/s x 2 操作数 x 4B = %.0f TB/s" % (8 * 82.6e12 / 2 / 1e12))
print("寄存器带宽 H100 = (66.9/2) T FMA/s x 2 操作数 x 4B = %.0f TB/s" % (8 * 66.9e12 / 2 / 1e12))

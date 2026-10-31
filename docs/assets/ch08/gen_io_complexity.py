# -*- coding: utf-8 -*-
"""左图：标准注意力 vs FlashAttention 分块 online softmax 的 HBM 读写量随 T 的增长
（按 FlashAttention 论文的 IO 复杂度公式真实计算）；
右图：softmax 注意力 O(T^2 d) 与线性注意力/SSM O(T d^2) 的全序列计算量对比（对数轴）。
运行：python gen_io_complexity.py，输出 io_complexity.png。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# ---------- 左图：HBM 流量（元素访问次数 × 2 字节，fp16） ----------
# 朴素实现：S、P 矩阵各"写一次 + 读一次" -> 4*T^2；Q/K/V/O 往返 -> 4*T*d
# FlashAttention：Q/O 各一次 -> 2*T*d；K/V 分块按 IO 复杂度公式 T^2*d^2/M（M 为 SRAM 元素数）
D = 128
M = 192 * 1024            # A100 一组 SM 的 SRAM 约 192 KiB（按 fp16 元素计）
T = np.logspace(9, 16.2, 60, base=2)          # T: 512 ~ ~9 万

naive_bytes = (4 * T**2 + 4 * T * D) * 2
flash_bytes = (T**2 * D**2 / M + 2 * T * D) * 2

# ---------- 右图：全序列计算量 ----------
D2 = 512                                     # 取常见隐藏维
T2 = np.logspace(9, 16.41, 60, base=2)       # T: 512 ~ ~9 万
flops_softmax = 2 * T2**2 * D2               # 每头 QK^T 与 PV 各 2*T^2*d 量级
flops_linear = 2 * T2 * D2**2                # 线性注意力/SSM：状态更新 O(T*d^2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.8), dpi=140)

ax1.loglog(T, naive_bytes / 2**20, lw=2.2, color="#C44E52",
           label="朴素实现 $\\Theta(T^2 + Td)$")
ax1.loglog(T, flash_bytes / 2**20, lw=2.2, color="#4C72B0",
           label="FlashAttention $\\Theta(T^2 d^2/M + Td)$")
ax1.axvline(65536, color="gray", ls=":", lw=1)
i64 = np.argmin(np.abs(T - 65536))
ax1.annotate(f"T=64K：朴素 {naive_bytes[i64]/2**20:.0f} MiB vs FA "
             f"{flash_bytes[i64]/2**20:.0f} MiB（约 "
             f"{naive_bytes[i64]/flash_bytes[i64]:.0f} 倍）",
             xy=(65536, naive_bytes[i64] / 2**20),
             xytext=(700, naive_bytes[i64] / 2**20 * 0.25), fontsize=8.5,
             arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))
ax1.set_xlabel("序列长度 T（对数轴）")
ax1.set_ylabel("单头 HBM 读写字节数（MiB，对数轴）")
ax1.set_title("(a) HBM 读写量：分块 online softmax 不物化 $T\\times T$ 矩阵", fontsize=10.5)
ax1.legend(fontsize=9, loc="upper left")
ax1.grid(True, which="both", ls="--", lw=0.35, alpha=0.45)
ax1.set_title("(a) HBM 读写量：朴素 vs 分块（d=128，M=192KiB）", fontsize=10.5)

ax2.loglog(T2, flops_softmax, lw=2.2, color="#C44E52",
           label="softmax 注意力 $2T^2 d$")
ax2.loglog(T2, flops_linear, lw=2.2, color="#55A868",
           label="线性注意力 / SSM $2Td^2$")
ax2.axvline(D2, color="gray", ls=":", lw=1)
ax2.annotate("交叉点 T = d = 512", xy=(D2, 2 * D2**3), xytext=(900, 2 * D2**3),
             fontsize=9, arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))
ax2.annotate("T=64K 时相差\nd/T = 128 倍", xy=(65536, 2 * 65536**2 * D2),
             xytext=(9000, 2 * 65536**2 * D2 * 0.35), fontsize=9,
             arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))
ax2.set_xlabel("序列长度 T（对数轴）")
ax2.set_ylabel("全序列计算量（FLOPs，对数轴）")
ax2.set_title("(b) 计算复杂度：$O(T^2 d)$ vs $O(Td^2)$（d=512）", fontsize=10.5)
ax2.legend(fontsize=9, loc="upper left")
ax2.grid(True, which="both", ls="--", lw=0.35, alpha=0.45)

fig.tight_layout()
fig.savefig("io_complexity.png", dpi=140)
for t_query in (4096, 65536):
    i = np.argmin(np.abs(T - t_query))
    j = np.argmin(np.abs(T2 - t_query))
    print(f"T={t_query}: 朴素 {naive_bytes[i]/2**20:10.1f} MiB, "
          f"FA {flash_bytes[i]/2**20:8.3f} MiB, "
          f"比值 {naive_bytes[i]/flash_bytes[i]:8.1f}x | "
          f"FLOPs 比 {flops_softmax[j]/flops_linear[j]:6.1f}x")
print("已保存 io_complexity.png")

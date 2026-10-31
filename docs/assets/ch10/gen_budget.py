# -*- coding: utf-8 -*-
"""
6ND 预算算例可视化：固定数据量 D = 1T token 时，训练不同规模模型需要的 GPU·天
（H100 SXM bf16 峰值 989 TFLOPS，三档 MFU），并在 MFU=40% 线上标注 7B / 13B / 70B
三点及其云端费用；同时画出 Chinchilla 最优（D = 20N）曲线做"过训练"对照。
公式：GPU·天 = 6ND / (MFU × 峰值FLOPS × 86400)
运行：python gen_budget.py  （依赖 numpy + matplotlib）
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

PEAK = 989e12          # H100 SXM bf16 稠密峰值 FLOPS
D_FIX = 1e12           # 固定 1T token
PRICE = 2.5            # 美元 / 卡·时（云端参考价）

def gpu_days(N, D, mfu):
    return 6 * N * D / (mfu * PEAK * 86400)

Ns = np.logspace(9, 11.15, 200)
for mfu, style in [(0.3, ":"), (0.4, "-"), (0.5, "--")]:
    days = gpu_days(Ns, D_FIX, mfu)
    lw = 2.6 if mfu == 0.4 else 1.3
    plt.plot(Ns, days, style, lw=lw, color="#1f77b4" if mfu == 0.4 else "#9bb8d8",
             label=f"1T token，MFU={int(mfu*100)}%")
    if mfu == 0.4:
        label_pos = {7e9: (1.1e9, 1750), 1.3e10: (2.1e10, 3600), 7e10: (3.6e10, 24000)}
        for N, name in [(7e9, "7B"), (1.3e10, "13B"), (7e10, "70B")]:
            d = gpu_days(np.array([N]), D_FIX, 0.4)[0]
            cost = d * 24 * PRICE / 1e4
            plt.scatter([N], [d], s=70, color="#d62728", zorder=5)
            plt.annotate(f"{name}：{d:,.0f} GPU·天\n≈ {cost:.1f} 万美元",
                         xy=(N, d), xytext=label_pos[N],
                         fontsize=10.5, color="#333333")

# Chinchilla 最优对照线：D = 20N（MFU=40%）
days_opt = gpu_days(Ns, 20 * Ns, 0.4)
plt.plot(Ns, days_opt, "-", lw=2, color="#2ca02c",
         label="Chinchilla 最优（D=20N），MFU=40%")
d7 = gpu_days(np.array([7e9]), 20 * 7e9, 0.4)[0]
plt.scatter([7e9], [d7], s=70, color="#2ca02c", zorder=5)
plt.annotate(f"7B 计算最优：140B token\n仅 {d7:.0f} GPU·天",
             xy=(7e9, d7), xytext=(1.7e10, 42), fontsize=10.5, color="#2ca02c")

plt.axhline(8, color="#888888", ls=":", lw=1)
plt.text(1.05e9, 11, "8 GPU·天 = 1 台 8×H100 跑 1 天", fontsize=9.5, color="#666666")

plt.xscale("log"); plt.yscale("log")
plt.xlabel("模型参数量 N", fontsize=12)
plt.ylabel("所需 GPU·天（H100，对数轴）", fontsize=12)
plt.title("训练预算计算器：$6ND$ 公式在不同规模/数据量下的 GPU·天\n（由代码生成，脚本同目录可复现）",
          fontsize=13)
plt.legend(fontsize=10, loc="upper left")
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig("gen_budget.png", dpi=150, bbox_inches="tight")
print("已保存 gen_budget.png")
print(f"7B×1T @MFU40% = {gpu_days(np.array([7e9]), 1e12, 0.4)[0]:,.0f} GPU·天，"
      f"8×H100 单机需 {gpu_days(np.array([7e9]), 1e12, 0.4)[0]/8:,.0f} 天")

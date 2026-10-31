# -*- coding: utf-8 -*-
"""
用 Chinchilla 论文的公开数据点（Table 3，Approach 1：固定模型规模、扫 token 得到的
计算最优点）真实拟合幂律 N*(C) ∝ C^a、D*(C) ∝ C^b，并外推到本章的两个算例预算：
  - C = 4.2e22 FLOPs（7B × 1T token 的预算）
  - C = 5.88e23 FLOPs（Chinchilla 70B × 1.4T 的预算，作交叉验证）
数据来源：Hoffmann et al., "Training Compute-Optimal Large Language Models",
arXiv:2203.15556, Table 3（FLOPs 列 / Tokens 列）。
运行：python gen_scaling_fit.py  （依赖 numpy + matplotlib）
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]  # 中文字体
plt.rcParams["axes.unicode_minus"] = False

# ---- Chinchilla 论文 Table 3 公开数据点：(参数量 N, 算力预算 C) ----
N_pts = np.array([4e8, 1e9, 1e10, 6.7e10, 1.75e11, 2.8e11, 5.2e11, 1e12, 1e13])
C_pts = np.array([1.92e19, 1.21e20, 1.23e22, 5.76e23, 3.85e24,
                  9.90e24, 3.43e25, 1.27e26, 1.30e28])
D_pts = C_pts / (6 * N_pts)          # 训练 token 数 D* = C/(6N)

# ---- log-log 最小二乘拟合：ln y = a * ln C + k ----
def fit_powerlaw(y):
    A = np.vstack([np.log(C_pts), np.ones_like(C_pts)]).T
    coef, *_ = np.linalg.lstsq(A, np.log(y), rcond=None)
    return coef[0], np.exp(coef[1])  # 指数 a、系数 k（y = k * C^a）

a_N, k_N = fit_powerlaw(N_pts)
a_D, k_D = fit_powerlaw(D_pts)

# ---- 外推两个算例预算 ----
def extrap(c, a, k):
    return k * c ** a
N_7b = extrap(4.2e22, a_N, k_N)      # 7B×1T 预算的最优规模
N_ch  = extrap(5.88e23, a_N, k_N)    # 交叉验证：应落在 ~70B 附近
ratio = D_pts / N_pts

print(f"拟合结果: N* = {k_N:.4g} * C^{a_N:.3f}   (论文 Approach 1 给 a=0.50)")
print(f"          D* = {k_D:.4g} * C^{a_D:.3f}   (论文 Approach 1 给 b=0.50)")
print(f"D*/N* 比值范围: {ratio.min():.1f} ~ {ratio.max():.1f}")
print(f"外推 C=4.2e22 (7B×1T 预算): 最优 N* ≈ {N_7b:.2e} 参数, D* = C/(6N*) ≈ {4.2e22/(6*N_7b):.2e} token")
print(f"外推 C=5.88e23 (Chinchilla 预算): 最优 N* ≈ {N_ch:.2e} 参数 (论文实际取 70B)")

# ---- 画图 ----
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
C_line = np.logspace(np.log10(C_pts.min()) * 0.92, np.log10(6.4e23), 200)

ax = axes[0]
ax.scatter(C_pts, N_pts, s=55, color="#1f77b4", zorder=3, label="论文 Table 3 数据点（9 个）")
ax.plot(C_line, extrap(C_line, a_N, k_N), color="#d62728", lw=2,
        label=f"拟合：$N^* = {k_N:.3g}\\,C^{{{a_N:.2f}}}$")
ax.scatter([5.88e23], [7e10], marker="*", s=320, color="#2ca02c", zorder=4,
           label=f"Chinchilla 实训点 70B（外推预测 {N_ch/1e9:.0f}B，命中）")
ax.scatter([4.2e22], [N_7b], marker="D", s=90, color="#ff7f0e", zorder=4,
           label=f"外推 7B×1T 预算 → 最优 ≈ {N_7b/1e9:.0f}B")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("算力预算 C（FLOPs）", fontsize=12)
ax.set_ylabel("计算最优参数量 $N^*$", fontsize=12)
ax.set_title("计算最优模型规模随算力的幂律增长", fontsize=13)
ax.legend(fontsize=10, loc="upper left")
ax.grid(True, which="both", alpha=0.3)

ax = axes[1]
ax.scatter(C_pts, D_pts, s=55, color="#1f77b4", zorder=3, label="论文数据点 $D^* = C/(6N)$")
ax.plot(C_line, extrap(C_line, a_D, k_D), color="#d62728", lw=2,
        label=f"拟合：$D^* = {k_D:.3g}\\,C^{{{a_D:.2f}}}$")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("算力预算 C（FLOPs）", fontsize=12)
ax.set_ylabel("计算最优数据量 $D^*$（token）", fontsize=12)
ax.set_title(f"数据量同步增长，$D^*/N^* \\approx {ratio.min():.0f}\\sim{ratio.max():.0f}$（约 20）",
             fontsize=13)
ax.legend(fontsize=10, loc="upper left")
ax.grid(True, which="both", alpha=0.3)

fig.suptitle("用 Chinchilla 论文公开数据点真实拟合计算最优前沿（由代码生成，脚本同目录可复现）",
             fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig("gen_scaling_fit.png", dpi=150, bbox_inches="tight")
print("已保存 gen_scaling_fit.png")

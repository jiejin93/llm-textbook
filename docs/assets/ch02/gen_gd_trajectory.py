# -*- coding: utf-8 -*-
"""不同学习率下梯度下降在 2D 峡谷损失面上的轨迹（等高线 + 三色轨迹）。
损失面：f(x) = 0.5 x^T A x，A 的特征值为 1 与 60（旋转 45 度），形成"峡谷"。
由代码生成，脚本同目录可复现：python gen_gd_trajectory.py
依赖：matplotlib, numpy
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ---------- 损失面：旋转的各向异性二次函数（峡谷） ----------
theta = np.pi / 4
R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
eigvals = np.array([1.0, 60.0])
A = R @ np.diag(eigvals) @ R.T

def f(x):
    return 0.5 * x @ A @ x

def grad(x):
    return A @ x

# ---------- 三种学习率的 GD 轨迹 ----------
x0 = np.array([4.0, 6.0])
runs = [
    ("过小 lr=0.004：200 步仍在谷里爬", 0.004, 200, "#2e6da4"),
    ("合适 lr=0.030：60 步稳定收敛", 0.030, 60, "#3a9d5d"),
    ("过大 lr=0.0345：震荡放大发散", 0.0345, 16, "#c0504d"),
]

# ---------- 画图 ----------
fig, ax = plt.subplots(figsize=(8.2, 6.2), dpi=150)
gx, gy = np.meshgrid(np.linspace(-8, 8, 400), np.linspace(-8, 8, 400))
Z = 0.5 * (A[0, 0] * gx ** 2 + 2 * A[0, 1] * gx * gy + A[1, 1] * gy ** 2)
levels = np.logspace(-1, np.log10(Z.max()), 18)
cs = ax.contour(gx, gy, Z, levels=levels, cmap="Blues", alpha=0.75, linewidths=0.9)
ax.clabel(cs, levels=levels[::4], fmt="%.0f", fontsize=7.5, colors="#556688")

for name, lr, n_steps, color in runs:
    x = x0.copy()
    traj = [x.copy()]
    for _ in range(n_steps):
        x = x - lr * grad(x)
        traj.append(x.copy())
    traj = np.array(traj)
    ax.plot(traj[:, 0], traj[:, 1], "-o", color=color, ms=3.5, lw=1.5,
            label=name, zorder=5)

ax.scatter([0], [0], marker="*", s=260, color="#222222", zorder=6)
ax.annotate("全局最优（谷底）", (0.3, -0.55), fontsize=9.5, fontweight="bold")
ax.set_xlabel("参数 $x_1$", fontsize=11)
ax.set_ylabel("参数 $x_2$", fontsize=11)
ax.set_title("梯度下降轨迹：同一峡谷损失面、同一起点，三种学习率三种命运\n"
             "（等高线越密 = 越陡；陡方向震荡、平方向缓慢，是动量与 Adam 的动机）",
             fontsize=11.5)
ax.legend(fontsize=9.5, loc="upper left")
ax.set_xlim(-8, 8)
ax.set_ylim(-8, 8)
ax.grid(ls="--", alpha=0.3)
fig.tight_layout()
fig.savefig(__file__.replace("gen_gd_trajectory.py", "gd_trajectory.png"),
            bbox_inches="tight")
print("saved: gd_trajectory.png")

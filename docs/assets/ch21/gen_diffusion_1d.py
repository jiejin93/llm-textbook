# -*- coding: utf-8 -*-
"""图②：一维扩散过程的真实轨迹——正弦曲线逐步加噪（DDPM 闭式解）+ DDIM 确定性去噪回路。

全部由真实公式计算：前向用 x_t = sqrt(abar_t) x0 + sqrt(1-abar_t) eps 一步加噪；
DDIM 去噪用 x_{t-1} = sqrt(abar_{t-1}) x0_hat + sqrt(1-abar_{t-1}) eps_hat，
其中 eps_hat 用"完美噪声预测器"（加噪时真实混入的那个 eps）代替训练好的网络，
以展示采样器本身的机械过程。脚本同目录可复现：python gen_diffusion_1d.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(21)
T = 1000
x = np.linspace(0, 1, 128)
x0 = np.sin(2 * np.pi * x) * 0.8                       # 干净数据（一维“信号”）

# ---- 1) DDPM 线性调度与闭式解所需量 ----------------------------------------
betas = np.linspace(1e-4, 0.02, T)
alphas = 1 - betas
abar = np.cumprod(alphas)                              # alpha_bar_t

# ---- 2) 前向：固定一个 eps，闭式一步加噪 ------------------------------------
eps = rng.normal(size=x0.shape)
def forward(t):
    """闭式解：任意时刻 t（1..T）一步算出 x_t，无需逐步迭代。"""
    return np.sqrt(abar[t - 1]) * x0 + np.sqrt(1 - abar[t - 1]) * eps

fwd_ts = [1, 300, 700, 1000]

# ---- 3) DDIM 确定性去噪回路（eta=0，跳步 10 步） ----------------------------
def ddim_sample(eps_hat, steps):
    """从 x_T 出发的 DDIM 确定性采样；steps 为降序时间步序列。

    起点是前向闭式解在 t=T 处的样本：q(x_T|x_0)=N(sqrt(abar_T)x0,(1-abar_T)I)≈N(0,I)，
    这正是扩散采样"从纯高斯噪声开始"的严格含义。
    """
    traj = {}
    x_t = np.sqrt(abar[steps[0] - 1]) * x0 + np.sqrt(1 - abar[steps[0] - 1]) * eps_hat
    traj[steps[0]] = x_t.copy()
    for t, t_prev in zip(steps[:-1], steps[1:]):
        x0_hat = (x_t - np.sqrt(1 - abar[t - 1]) * eps_hat) / np.sqrt(abar[t - 1])
        if t_prev == 0:                      # 最后一步直接输出预测的干净数据
            x_t = x0_hat
        else:
            x_t = (np.sqrt(abar[t_prev - 1]) * x0_hat
                   + np.sqrt(1 - abar[t_prev - 1]) * eps_hat)
        traj[t_prev] = x_t.copy()
    return traj

steps = np.arange(1000, 0, -100)                         # 1000, 900, ..., 100
steps = np.append(steps, 0)                              # 最后一跳落到 t=0
traj = ddim_sample(eps, steps)                          # 完美预测器
x0_rec = traj[steps[-1]]
mse = float(np.mean((x0_rec - x0) ** 2))

# ---- 4) 画图 ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), dpi=150, sharey=True)
cmap = plt.get_cmap("Blues")

ax = axes[0]
for i, t in enumerate(fwd_ts):
    ax.plot(x, forward(t), color=cmap(0.3 + 0.65 * i / (len(fwd_ts) - 1)), lw=1.5,
            label=f"$t$={t}" + ("（≈纯噪声）" if t == 1000 else ""))
ax.plot(x, x0, color="crimson", lw=2.2, label="数据 $x_0$（正弦）")
ax.set_title("前向过程：$x_t=\\sqrt{\\bar{\\alpha}_t}\\,x_0+\\sqrt{1-\\bar{\\alpha}_t}\\,\\epsilon$\n"
             "数据逐步被噪声淹没（闭式解一步算出）", fontsize=10.5)
ax.set_xlabel("空间坐标")
ax.set_ylabel("信号值")
ax.legend(fontsize=8.5, loc="upper right")
ax.grid(alpha=0.3)

ax = axes[1]
plot_ts = [1000, 800, 600, 400, 200, 0]                 # 展示去噪的渐进轨迹
for i, t in enumerate(plot_ts):
    c = cmap(0.30 + 0.65 * i / (len(plot_ts) - 1))
    lw = 2.2 if t in (1000, 0) else 1.3
    lab = (f"$t$={t}（≈纯噪声起点）" if t == 1000
           else (f"$t$={t}（输出 $\\hat{{x}}_0$）" if t == 0 else None))
    ax.plot(x, traj[t], color=c, lw=lw, label=lab)
ax.plot(x, x0, "--", color="crimson", lw=1.8, label="目标 $x_0$")
ax.set_title("DDIM 去噪回路（$\\eta=0$，10 步跳步）：\n"
             f"从纯噪声恢复数据，重建 MSE={mse:.2e}（完美 $\\hat\\epsilon$）", fontsize=10.5)
ax.set_xlabel("空间坐标")
ax.legend(fontsize=8, loc="upper right", ncols=2)
ax.grid(alpha=0.3)

fig.suptitle("一维扩散：训练学会预测 $\\epsilon$ 之后，采样就是反复“预测 $\\hat{x}_0$ → 沿确定方向回退一步”",
             fontsize=11.5, y=1.02)
fig.tight_layout()
fig.savefig(__file__.replace("gen_diffusion_1d.py", "diffusion_1d.png"),
            dpi=150, bbox_inches="tight")

print("DDIM 重建 MSE（完美噪声预测器）:", mse)
print("abar[0], abar[500], abar[1000] =",
      round(float(abar[0]), 4), round(float(abar[499]), 4), round(float(abar[999]), 4))
print("已保存 diffusion_1d.png")

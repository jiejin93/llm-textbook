# -*- coding: utf-8 -*-
"""SGD / 动量 / AdamW 在 toy 凸面上的收敛速度对比。
凸面：L(w) = 0.5*(s1*w1^2 + s2*w2^2)，s1=1、s2=1e-3——两个坐标的梯度尺度差
1000 倍，是 LLM 中"不同参数梯度尺度高度异质"的最小缩影。
由代码生成，脚本同目录可复现：python gen_optim_compare.py
依赖：torch >= 2.0 (CPU 即可), matplotlib
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

s = torch.tensor([1.0, 1e-3])     # 两个坐标的曲率：差 1000 倍
x0 = torch.tensor([2.0, 5.0])
N = 300

def run(make_opt):
    x = x0.clone().requires_grad_(True)
    opt = make_opt([x])
    curve = []
    for _ in range(N):
        opt.zero_grad()
        loss = (0.5 * s * x ** 2).sum()
        loss.backward()
        opt.step()
        curve.append((0.5 * s * x.detach() ** 2).sum().item())
    return curve

curves = [
    ("SGD        lr=0.9（受大梯度坐标限制，不能再大）", run(lambda p: torch.optim.SGD(p, lr=0.9)), "#c0504d"),
    ("动量 0.9   lr=0.6", run(lambda p: torch.optim.SGD(p, lr=0.6, momentum=0.9)), "#e8a33d"),
    ("AdamW     lr=0.05", run(lambda p: torch.optim.AdamW(p, lr=0.05, weight_decay=0.0)), "#3a9d5d"),
]

fig, ax = plt.subplots(figsize=(8.6, 5.4), dpi=150)
for name, curve, color in curves:
    ax.semilogy(curve, lw=2.0, color=color, label=name)
ax.set_xlabel("训练步数", fontsize=11)
ax.set_ylabel("损失（log 尺度）", fontsize=11)
ax.set_title("梯度尺度异质的凸面（两坐标曲率差 1000 倍）上的收敛速度\n"
             "SGD/动量被大梯度坐标锁死学习率，小梯度坐标几乎不动 → 平台期；"
             "AdamW 逐坐标自适应步长，全坐标推进",
             fontsize=11.5)
ax.legend(fontsize=9.5, loc="upper right")
ax.grid(ls="--", alpha=0.4, which="both")
print("最终损失：", {n.split()[0]: f"{c[-1]:.2e}" for n, c, _ in curves})
fig.tight_layout()
fig.savefig(__file__.replace("gen_optim_compare.py", "optim_compare.png"),
            bbox_inches="tight")
print("saved: optim_compare.png")

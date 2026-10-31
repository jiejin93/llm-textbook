# -*- coding: utf-8 -*-
"""真实运行实验：toy 分类任务上的 4 专家 MoE，带 / 不带辅助损失各训练一次，
画出训练结束后各专家被分配的 token 数直方图（路由塌缩 vs 均衡，一眼可见）。
- 数据：4 类高斯簇，簇大小刻意不均（70%/10%/10%/10%），模拟真实数据的分布倾斜；
- 模型：router(Linear 2->4, top-1) + 4 个可训练专家线性层(Linear 2->2) + 线性分类头；
- 两次训练唯一区别：alpha=0（无辅助损失）与 alpha=0.01（Switch 式辅助损失）。
运行：python gen_moe_balance.py（CPU 数秒），输出 moe_balance.png 并打印负载直方图。
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # Anaconda + torch 常见的 OpenMP 冲突规避

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

N_EXPERTS, N_CLASSES, D = 4, 4, 2
ALPHA = 0.01                                   # Switch Transformer 的系数

# ---------- toy 分类数据：4 个高斯簇，样本数刻意不均 ----------
centers = torch.tensor([[2.0, 2.0], [-2.0, 2.0], [-2.0, -2.0], [2.0, -2.0]])
sizes = [1400, 200, 200, 200]                  # 70% / 10% / 10% / 10%
xs, ys = [], []
for c, n in enumerate(sizes):
    xs.append(centers[c] + 0.6 * torch.randn(n, D))
    ys.append(torch.full((n,), c))
X, Y = torch.cat(xs), torch.cat(ys)

class MoEClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.router = nn.Linear(D, N_EXPERTS)
        self.experts = nn.ModuleList([nn.Linear(D, D) for _ in range(N_EXPERTS)])
        self.head = nn.Linear(D, N_CLASSES)

    def forward(self, x):
        scores = self.router(x)                            # (B, N)
        top1 = scores.argmax(dim=-1)                       # top-1 硬路由
        h = torch.stack([e(x) for e in self.experts], dim=1)   # (B, N, D)
        h = h[torch.arange(len(x)), top1]                  # 只取被选中专家的输出
        return self.head(h), scores

def aux_loss(scores):
    """Switch 式辅助损失 alpha * N * sum_i f_i * P_i（9.3 节公式）。"""
    P = scores.softmax(-1).mean(0)
    f = torch.bincount(scores.argmax(-1), minlength=N_EXPERTS).float() / len(scores)
    return ALPHA * N_EXPERTS * (f * P).sum()

def train(alpha):
    torch.manual_seed(7)                                   # 两次训练同一初始化
    model = MoEClassifier()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for step in range(300):
        opt.zero_grad()
        logits, scores = model(X)
        loss = F.cross_entropy(logits, Y)
        if alpha > 0:
            loss = loss + alpha * aux_loss(scores)
        loss.backward()
        opt.step()
    with torch.no_grad():
        assign = model.router(X).argmax(-1)
        counts = torch.bincount(assign, minlength=N_EXPERTS).tolist()
        acc = (logits.argmax(-1) == Y).float().mean().item()
    return counts, acc

results = {}
for alpha in (0.0, ALPHA):
    counts, acc = train(alpha)
    results[alpha] = counts
    print(f"alpha={alpha}: 专家负载(个数) = {counts}   测试精度 = {acc:.3f}")

# ---------- 直方图 ----------
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=140, sharey=True)
titles = ["无辅助损失（alpha=0）：路由塌缩", "带辅助损失（alpha=0.01）：负载被拉平"]
for ax, alpha, title in zip(axes, (0.0, ALPHA), titles):
    counts = results[alpha]
    bars = ax.bar(range(N_EXPERTS), counts, color="#4C72B0",
                  edgecolor="black", linewidth=0.5, tick_label=[f"专家{i}" for i in range(N_EXPERTS)])
    n_total = sum(counts)
    ax.axhline(n_total / N_EXPERTS, color="#C44E52", ls="--", lw=1.6,
               label=f"理想均衡 {n_total}/{N_EXPERTS} = {n_total // N_EXPERTS}")
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, c + 25, f"{c}\n({c / n_total:.0%})",
                ha="center", fontsize=9)
    ax.set_title(title, fontsize=10.5)
    ax.set_xlabel("专家编号")
    ax.set_ylim(0, n_total * 1.15)
axes[0].set_ylabel("被分配的 token 数")
axes[0].legend(fontsize=9)
fig.suptitle("同一个 4 专家 MoE、同一份倾斜数据：辅助损失决定路由是否塌缩（top-1 路由，训练后统计）",
             fontsize=11.5)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig("moe_balance.png", dpi=140)
print("已保存 moe_balance.png")

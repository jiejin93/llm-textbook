# gen_softmax_scaling.py —— 缩放因子对 softmax 的影响（真实数值实验）
# 复现：python gen_softmax_scaling.py   （依赖 numpy matplotlib）
# 产物：
#   ch06_gen_softmax_scaling.png —— (a) softmax 熵 vs d_k；(b) 最大概率 vs d_k，
#                                   三种缩放因子对比，标注 d_k=64
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(0)
N_TRIAL, T = 4000, 128          # 每个点 4000 组随机 q,k；softmax 长度 128
D_KS = np.unique(np.concatenate([
    np.round(np.logspace(np.log10(4), np.log10(512), 16)).astype(int), [64]]))

def stats(d_k, divisor_mode):
    """对给定 d_k 与缩放方式，返回 softmax 的平均熵与平均最大概率。
    q,k 各分量独立标准正态——这是 6.3.3 节方差推导的假设本身。"""
    q = rng.standard_normal((N_TRIAL, d_k))
    k = rng.standard_normal((N_TRIAL, T, d_k))
    logits = np.einsum("nd,ntd->nt", q, k)          # 未缩放点积
    if divisor_mode == "none":      div = 1.0
    elif divisor_mode == "sqrt":    div = np.sqrt(d_k)
    elif divisor_mode == "dk":      div = float(d_k)
    logits = logits / div
    logits -= logits.max(axis=1, keepdims=True)      # 数值稳定的 softmax
    p = np.exp(logits); p /= p.sum(axis=1, keepdims=True)
    ent = -(p * np.log(p + 1e-12)).sum(axis=1).mean()
    return ent, p.max(axis=1).mean()

modes = [("none", "不缩放（除以 1）", "#d62728"),
         ("sqrt", "除以 $\\sqrt{d_k}$（标准）", "#1f6fd6"),
         ("dk",   "除以 $d_k$（过度缩放）", "#2ca02c")]

ents = {m: [] for m, _, _ in modes}
maxs = {m: [] for m, _, _ in modes}
for d_k in D_KS:
    for m, _, _ in modes:
        e, mx = stats(int(d_k), m)
        ents[m].append(e); maxs[m].append(mx)

fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), dpi=150)
for m, label, color in modes:
    axes[0].plot(D_KS, ents[m], "o-", ms=4, color=color, lw=1.8, label=label)
    axes[1].plot(D_KS, maxs[m], "o-", ms=4, color=color, lw=1.8, label=label)
for ax in axes:
    ax.axvline(64, color="gray", ls="--", lw=1.2)
    ax.annotate("d_k = 64", (66, ax.get_ylim()[1] * 0.92), color="gray", fontsize=10)
    ax.set_xscale("log", base=2)
    ax.set_xticks([4, 8, 16, 32, 64, 128, 256, 512])
    ax.set_xticklabels(["4", "8", "16", "32", "64", "128", "256", "512"])
    ax.set_xlabel("键/查询维度 $d_k$（softmax 长度固定为 128）")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
axes[0].set_ylabel("softmax 平均熵（nat）")
axes[0].set_title("(a) 熵：不缩放时随 $d_k$ 塌缩到 0\n= 分布退化为 one-hot")
axes[1].set_ylabel("softmax 平均最大概率")
axes[1].set_title("(b) 最大概率：不缩放时逼近 1\n= 注意力\u201c提前定死\u201d，梯度趋零")
fig.suptitle("缩放因子实验：$q,k$ 各分量独立标准正态，4000 组采样平均（由代码生成）",
             fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig("ch06_gen_softmax_scaling.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# 打印 d_k=64 处的数值供正文引用
for i, dk in enumerate(D_KS):
    if dk == 64:
        for m, label, _ in modes:
            print(f"d_k=64 | {label}: 熵={ents[m][i]:.3f} nat, "
                  f"最大概率={maxs[m][i]:.3f}")
print("图已生成：ch06_gen_softmax_scaling.png")

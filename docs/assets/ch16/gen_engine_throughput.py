# -*- coding: utf-8 -*-
"""图 16-2：vLLM vs SGLang 在不同负载特征下的相对吞吐（解析模型估算，非实测）。

模型（正文 16.3/16.4 给出全部假设）：
    相对吞吐 R = (T_in + T_out) / ((1 - eta_eff) * T_in + T_out)，无缓存基线 R=1
    eta_eff（有效前缀命中率）：
      SGLang RadixAttention：token 级匹配 → eta_eff = eta
      vLLM APC：块级（block_size=16）整块匹配 → 可复用 token 数向下取整到块边界
    多副本（右图，长多轮负载）：
      vLLM + 轮询负载均衡：前缀散落在 n 个副本 → eta_eff = eta / n
      SGLang + cache-aware 路由（官方 router）：eta_eff ≈ 0.9 * eta
注意：这是按命中率公式建模的估算，用于解释差距来源；引擎内核层面的快慢
（kernel、调度器实现）不在本模型内。实测数据见本章延伸阅读。
由代码生成，脚本同目录可复现：python gen_engine_throughput.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

BLOCK = 16  # vLLM block_size

def rel_tp(T_in, T_out, eta_eff):
    """相对无缓存基线的吞吐倍率（只建模 prefill 计算量的节省）。"""
    return (T_in + T_out) / ((1 - eta_eff) * T_in + T_out)

def eta_vllm(T_in, eta):
    """vLLM APC：只有完整块可复用 → 可复用 token 向下取整到 16 的倍数。"""
    shared = int(eta * T_in)
    usable = (shared // BLOCK) * BLOCK
    return usable / T_in

# 三种负载特征（假设值，正文有推导）
loads = [
    ("短问答\nT_in=128, T_out=256", 128, 256, 0.02),
    ("RAG 前缀复用\nT_in=4096, T_out=128", 4096, 128, 0.60),
    ("长多轮对话\nT_in=2048, T_out=256", 2048, 256, 0.85),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.6))

# ---- 左图：单副本，三种负载 ----
x = np.arange(len(loads))
w = 0.36
r_vllm = [rel_tp(T, O, eta_vllm(T, e)) for _, T, O, e in loads]
r_sg = [rel_tp(T, O, e) for _, T, O, e in loads]
b1 = ax1.bar(x - w / 2, r_vllm, w, color="tab:blue", label="vLLM（APC，块哈希前缀缓存）")
b2 = ax1.bar(x + w / 2, r_sg, w, color="tab:red", label="SGLang（RadixAttention，前缀树）")
for b in list(b1) + list(b2):
    ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.06,
             f"{b.get_height():.2f}", ha="center", fontsize=10)
ax1.axhline(1.0, color="gray", ls="--", lw=1.2)
ax1.text(-0.42, 1.08, "基线=1", fontsize=9, color="gray")
ax1.set_xticks(x)
ax1.set_xticklabels([l[0] for l in loads], fontsize=10)
ax1.set_ylabel("相对吞吐（无缓存基线 = 1）")
ax1.set_ylim(0, 4.8)
ax1.set_title("单副本、缓存空闲：两引擎几乎打平\n（差距只来自块对齐损失，<3%）")
ax1.legend(fontsize=9.5, loc="upper left")
ax1.grid(axis="y", alpha=0.3)

# ---- 右图：多副本（长多轮负载），路由策略决定胜负 ----
n_rep = np.array([1, 2, 4, 8, 16])
eta, T_in, T_out = 0.85, 2048, 256
r_poll = np.array([rel_tp(T_in, T_out, eta / n) for n in n_rep])
r_router = np.array([rel_tp(T_in, T_out, (0.9 * eta) if n > 1 else eta) for n in n_rep])
ax2.plot(n_rep, r_poll, "o-", color="tab:blue", lw=2.2, label="vLLM APC + 轮询负载均衡（$\\eta_{eff}=\\eta/n$）")
ax2.plot(n_rep, r_router, "s-", color="tab:red", lw=2.2, label="SGLang Radix + cache-aware 路由（$\\eta_{eff}\\approx0.9\\eta$）")
for n, r in zip(n_rep, r_poll):
    if n == 1:
        continue  # n=1 两线重合，只标红点，避免叠字
    ax2.annotate(f"{r:.2f}", (n, r), xytext=(0, -16), textcoords="offset points",
                 ha="center", fontsize=9, color="tab:blue")
for n, r in zip(n_rep, r_router):
    ax2.annotate(f"{r:.2f}", (n, r), xytext=(0, 8), textcoords="offset points",
                 ha="center", fontsize=9, color="tab:red")
ax2.axhline(1.0, color="gray", ls="--", lw=1.2)
ax2.annotate(f"8 副本时相差 {r_router[3] / r_poll[3]:.1f} 倍：\n命中被轮询打散，路由救回来",
             xy=(8, r_poll[3]), xytext=(9.5, 2.4), fontsize=10, color="purple",
             arrowprops=dict(arrowstyle="->", color="purple", lw=1.1))
ax2.set_xscale("log", base=2)
ax2.set_xticks(n_rep)
ax2.set_xticklabels(n_rep)
ax2.set_xlabel("副本数 n（同一模型、同一批卡）")
ax2.set_ylabel("相对吞吐（无缓存基线 = 1）")
ax2.set_ylim(0, 4.4)
ax2.set_title("长多轮负载（$\\eta$=0.85）：\n命中率 × 路由才是吞吐胜负手")
ax2.legend(fontsize=9.5, loc="upper right")
ax2.grid(alpha=0.3, which="both")

fig.suptitle("vLLM vs SGLang 相对吞吐：命中率解析模型（模型化估算，非实测；实测见本章延伸阅读）",
             fontsize=12.5, y=1.03)
fig.tight_layout()
fig.savefig("engine_relative_throughput.png", dpi=140, bbox_inches="tight")
print("saved engine_relative_throughput.png")
for (name, T, O, e), rv, rs in zip(loads, r_vllm, r_sg):
    print(f"{name!r}: vLLM={rv:.3f}, SGLang={rs:.3f}, 比值={rs/rv:.3f}")
print(f"n=8: poll={r_poll[3]:.2f}, router={r_router[3]:.2f}, 比={r_router[3]/r_poll[3]:.2f}")

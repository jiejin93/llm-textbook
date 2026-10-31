# -*- coding: utf-8 -*-
"""图 15-6：KV Cache 显存 vs 上下文长度曲线族（7B/13B 级/70B × MHA/GQA-8）。

按第 8 章 KV 显存公式真实计算：
    M_KV(T) = 2 * L * T * d_kv * b_param  字节/单条序列
其中因子 2 = K 与 V 两份，d_kv = h_kv * d_k，b_param = 2（fp16）。
水平参考线 12/24/80 GB 对应 12GB 消费级卡 / RTX 4090 / A100-H100 80GB。
由代码生成，脚本同目录可复现：python gen_kv_memory.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

GB = 1024 ** 3          # 与显卡标称一致的 GiB 口径
B_PARAM = 2             # fp16 每参数 2 字节
KI = 1024

# (标签, 层数 L, d_kv = h_kv*d_k, 颜色, 线型)
models = [
    ("7B  MHA      （L=32, d_kv=4096）", 32, 32 * 128, "tab:red", "-"),
    ("7B  GQA-8    （L=32, d_kv=1024）", 32, 8 * 128, "tab:orange", "-"),
    ("14B 级 GQA-8 （L=48, d_kv=1024）", 48, 8 * 128, "tab:green", "-"),
    ("70B GQA-8    （L=80, d_kv=1024）", 80, 8 * 128, "tab:blue", "-"),
    ("70B 若用 MHA （L=80, d_kv=8192）", 80, 64 * 128, "tab:purple", "--"),
]

ctx = np.linspace(1024, 131072, 400)      # 1K ~ 128K 上下文

fig, ax = plt.subplots(figsize=(10.5, 6.2))
for label, L, d_kv, color, ls in models:
    bytes_per_tok = 2 * L * d_kv * B_PARAM
    y = ctx * bytes_per_tok / GB
    ax.plot(ctx / 1024, y, color=color, ls=ls, lw=2.2, label=label)

# 参考横线：整机显存上限（未扣权重）
for mem, name in [(12, "12GB 卡（3060/4070 级）"), (24, "24GB（RTX 4090）"), (80, "80GB（A100/H100）")]:
    ax.axhline(mem, color="gray", ls=":", lw=1.2)
    ax.text(131, mem + 1.2, name, ha="right", fontsize=9.5, color="dimgray")

# 标注关键交点：M_KV(T*)=显存上限 → T* = 上限字节 / 每 token 字节
marks = [
    (32, 32 * 128, 12, "7B MHA 爆 12GB：T≈24K", "tab:red", (-38, 8)),
    (48, 8 * 128, 12, "14B 级 GQA 爆 12GB：T≈64K", "tab:green", (-55, 14)),
    (80, 8 * 128, 24, "70B GQA 爆 24GB：T≈75K", "tab:blue", (-58, 12)),
]
for L, d_kv, cap, text, color, (dx, dy) in marks:
    per_tok = 2 * L * d_kv * B_PARAM
    T_star = cap * GB / per_tok / 1024      # 以 K 为单位
    ax.plot([T_star], [cap], "o", ms=6, color=color, zorder=5)
    ax.annotate(text, xy=(T_star, cap), xytext=(T_star + dx, cap + dy),
                fontsize=9, color=color,
                arrowprops=dict(arrowstyle="->", color=color, lw=1))

ax.set_xlim(0, 132)
ax.set_ylim(0, 92)
ax.set_xticks(range(0, 140, 16))
ax.set_xlabel("上下文长度（千 token，单条序列，batch=1）")
ax.set_ylabel("KV Cache 显存（GB，1 GB = $2^{30}$ 字节）")
ax.set_title("KV Cache 显存 vs 上下文长度（fp16，不含权重）\n"
             r"$M_{KV}=2\,L\,T\,d_{kv}\,b_{param}$：GQA-8 把每 token KV 压到 MHA 的 1/4~1/8")
ax.grid(alpha=0.3)
ax.legend(loc="upper left", fontsize=9.5, title="模型配置（每 token KV）", title_fontsize=9.5)
ax.text(0.985, 0.03,
        "注意：本图只是 KV 预算；fp16 权重还要再占 2N 字节\n"
        "（70B fp16 权重 140GB，只能进 80GB×2 卡）",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5, color="dimgray")

fig.tight_layout()
fig.savefig("kv_memory_vs_ctx.png", dpi=140)
print("saved kv_memory_vs_ctx.png")

# 控制台输出每 token KV，便于正文引用
for label, L, d_kv, *_ in models:
    per_tok = 2 * L * d_kv * B_PARAM
    print(f"{label}: {per_tok/KI:.0f} KiB/token;  4K 上下文单条 = {4096*per_tok/GB:.2f} GB")

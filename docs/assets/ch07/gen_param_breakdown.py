# -*- coding: utf-8 -*-
"""用第 6/7 章参数量公式，按真实 config 计算各模型"注意力/FFN/词嵌入"参数分布。
公式：注意力投影 = 2*d^2 + 2*d*h_kv*d_k；SwiGLU FFN = 3*d*d_ff；词嵌入 = V*d（tie 只计一次）。
所有超参取自各模型 HF config.json（模型名在 configs 中注明）。
运行：python gen_param_breakdown.py，输出 param_breakdown.png。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# 模型名 → (d, L, h_kv, d_k, d_ff, V, tie, 展示名)
configs = {
    "meta-llama/Meta-Llama-3-8B":      (4096, 32, 8,  128, 14336, 128256, False, "LLaMA-3-8B"),
    "mistralai/Mistral-7B-v0.1":       (4096, 32, 8,  128, 14336, 32000,  False, "Mistral-7B"),
    "Qwen/Qwen3-8B":                   (4096, 36, 8,  128, 12288, 151936, False, "Qwen3-8B"),
    "Qwen/Qwen3-1.7B":                 (2048, 28, 8,  128, 6144,  151936, True,  "Qwen3-1.7B"),
    "google/gemma-2-2b":               (2304, 26, 1,  256, 9216,  256128, True,  "Gemma-2-2B(MQA)"),
}

names, attn_share, ffn_share, emb_share, totals = [], [], [], [], []
for model_id, (d, L, h_kv, dk, d_ff, V, tie, show) in configs.items():
    attn = L * (2 * d * d + 2 * d * h_kv * dk)      # W_Q,W_O + GQA 的 W_K,W_V
    ffn = L * (3 * d * d_ff)                        # SwiGLU gate/up/down 三门
    emb = V * d * (1 if tie else 2)                 # tie 共享计 1 份，否则嵌入+head 共 2 份
    total = attn + ffn + emb
    names.append(show)
    attn_share.append(attn / total * 100)
    ffn_share.append(ffn / total * 100)
    emb_share.append(emb / total * 100)
    totals.append(total / 1e9)

y = np.arange(len(names))[::-1]
fig, ax = plt.subplots(figsize=(9, 4.6), dpi=140)
b1 = ax.barh(y, attn_share, color="#4C72B0", label="注意力投影（GQA）")
b2 = ax.barh(y, ffn_share, left=attn_share, color="#DD8452", label="SwiGLU FFN")
left2 = [a + f for a, f in zip(attn_share, ffn_share)]
b3 = ax.barh(y, emb_share, left=left2, color="#55A868", label="词嵌入矩阵")

for i, (yy, t) in enumerate(zip(y, totals)):
    ax.text(101, yy, f"{t:.2f} B", va="center", fontsize=9, color="#333333")
    ax.text(attn_share[i] / 2, yy, f"{attn_share[i]:.0f}%", va="center",
            ha="center", fontsize=8, color="white")
    ax.text(attn_share[i] + ffn_share[i] / 2, yy, f"{ffn_share[i]:.0f}%",
            va="center", ha="center", fontsize=8, color="white")
    if emb_share[i] > 6:
        ax.text(left2[i] + emb_share[i] / 2, yy, f"{emb_share[i]:.0f}%",
                va="center", ha="center", fontsize=8, color="white")

ax.set_yticks(y)
ax.set_yticklabels(names, fontsize=9.5)
ax.set_xlim(0, 112)
ax.set_xlabel("占全部参数比例（%）")
ax.set_title("参数都花在哪：按真实 config 计算的参数分布（右列为总参数量）", fontsize=11.5)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=9, framealpha=0.95)
ax.grid(True, axis="x", ls="--", lw=0.4, alpha=0.5)

fig.tight_layout()
fig.subplots_adjust(bottom=0.22)
fig.savefig("param_breakdown.png", dpi=140)
for n, t in zip(names, totals):
    print(f"{n:16s} 总参数 {t:6.2f} B")
print("已保存 param_breakdown.png")

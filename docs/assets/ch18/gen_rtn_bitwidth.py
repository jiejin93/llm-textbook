# -*- coding: utf-8 -*-
"""图①：位宽 vs 量化误差——真实 GPT-2 权重上的 RTN 量化实验（CPU 即可复现）。

数据：openai-community/gpt2 的两个真实权重矩阵（fp32 原值，未改动）：
- h.10.mlp.c_fc.weight   [768, 3072]，分布温和（std=0.128，absmax=2.55）
- h.11.attn.c_proj.weight [768, 768]，重尾（std=0.182，absmax=8.88，约 50 倍 std）
二者已抽取到同目录 gpt2_weights_sample.npz（约 10.5 MB），抽取命令：
  python -c "from safetensors.numpy import load_file; import numpy as np; \
  d=load_file('gpt2_model.safetensors'); \
  np.savez_compressed('gpt2_weights_sample.npz', \
      mlp_c_fc_L10=d['h.10.mlp.c_fc.weight'], attn_c_proj_L11=d['h.11.attn.c_proj.weight'])"
其中 gpt2_model.safetensors 可从 HF 下载（国内可用 hf-mirror.com 镜像）：
  curl -L -o gpt2_model.safetensors https://hf-mirror.com/openai-community/gpt2/resolve/main/model.safetensors
若两者都不可得，脚本回退到“标准正态+真实离群权重”混合分布并在输出中注明（此时图注须注明合成数据）。

实验：对称 RTN（per-group g=128），位宽 8/6/4/3/2，看两件事：
1. 权重相对 MSE；2. 模拟前向（线性层 y = W @ x，x 为高斯激活）输出的余弦相似度。
运行：python gen_rtn_bitwidth.py
"""
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))


def load_weights():
    """优先真实 GPT-2 权重；不可得则回退合成分布并注明。返回 [out, in]（GPT-2 Conv1D 原始存储为 [in, out]，此处转置）。"""
    npz = os.path.join(HERE, "gpt2_weights_sample.npz")
    if os.path.exists(npz):
        d = np.load(npz)
        return d["mlp_c_fc_L10"].T, d["attn_c_proj_L11"].T, "真实权重：openai-community/gpt2 第 10 层 MLP.c_fc 与第 11 层 attn.c_proj"
    env = os.environ.get("GPT2_SAFETENSORS")
    if env and os.path.exists(env):
        from safetensors.numpy import load_file
        d = load_file(env)
        return d["h.10.mlp.c_fc.weight"].T, d["h.11.attn.c_proj.weight"].T, "真实权重：openai-community/gpt2"
    rng = np.random.default_rng(0)
    w1 = rng.normal(0, 0.128, size=(3072, 768))
    w2 = rng.normal(0, 0.182, size=(768, 768))
    idx = rng.choice(w2.size, size=int(w2.size * 0.005), replace=False)  # 0.5% 离群
    w2.flat[idx] *= rng.uniform(20, 50, size=idx.size)
    return w1, w2, "合成分布：标准正态 + 0.5% 大离群（非真实模型权重）"


def rtn_quantize(w, bits, group=128):
    """分组对称 RTN：返回反量化权重。"""
    qmax = 2 ** (bits - 1) - 1
    g = w.reshape(-1, group)
    s = np.abs(g).max(axis=1, keepdims=True) / qmax
    q = np.clip(np.round(g / s), -qmax - 1, qmax)
    return (q * s).reshape(w.shape)


W1, W2, source = load_weights()
print("数据源：", source)
bits_list = [8, 6, 4, 3, 2]

mse1, mse2, cos1, cos2 = [], [], [], []
rng = np.random.default_rng(42)
X = rng.normal(0, 1.0, size=(768, 512))  # 模拟激活：768 维 x 512 个样本
Y1_fp, Y2_fp = W1 @ X, W2 @ X

for b in bits_list:
    for W, ref, mse_l, cos_l in [(W1, Y1_fp, mse1, cos1), (W2, Y2_fp, mse2, cos2)]:
        Wq = rtn_quantize(W, b, group=128)
        mse_l.append(np.mean((Wq - W) ** 2) / np.mean(W ** 2))  # 相对 MSE
        Yq = Wq @ X
        num = np.sum(Yq * ref, axis=0)
        den = np.linalg.norm(Yq, axis=0) * np.linalg.norm(ref, axis=0)
        cos_l.append(np.mean(num / den))

for b, m1, m2, c1, c2 in zip(bits_list, mse1, mse2, cos1, cos2):
    print(f"int{b}:  相对MSE  c_fc={m1:.2e}  c_proj={m2:.2e} |  余弦相似度  c_fc={c1:.5f}  c_proj={c2:.5f}")

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), dpi=150)
c_a, c_b = "#1976d2", "#e65100"

ax = axes[0]
ax.plot(bits_list, mse1, marker="o", color=c_a, label="MLP.c_fc（分布温和）")
ax.plot(bits_list, mse2, marker="s", color=c_b, label="attn.c_proj（重尾权重）")
ax.annotate("两条曲线几乎重合：\nper-group scale 抹平了重尾差异", xy=(3, 0.08), xytext=(4.2, 0.28),
            fontsize=9, arrowprops=dict(arrowstyle="->", color="dimgray"))
ax.set_yscale("log")
ax.set_xticks(bits_list)
ax.set_xlabel("位宽（bit）", fontsize=11)
ax.set_ylabel("权重相对 MSE = $\\Vert\\hat{W}-W\\Vert^2 / \\Vert W\\Vert^2$", fontsize=11)
ax.set_title("位宽 vs 权重误差：每降 1 bit 约放大 4 倍", fontsize=12)
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=10, loc="lower left")

ax = axes[1]
ax.plot(bits_list, cos1, marker="o", color=c_a, label="MLP.c_fc（分布温和）")
ax.plot(bits_list, cos2, marker="s", color=c_b, label="attn.c_proj（重尾权重）")
ax.set_xticks(bits_list)
ax.set_ylim(min(min(cos1), min(cos2)) - 0.03, 1.005)
ax.set_xlabel("位宽（bit）", fontsize=11)
ax.set_ylabel("模拟前向输出余弦相似度（512 个高斯输入取平均）", fontsize=11)
ax.set_title("位宽 vs 输出保真度：int6 起几乎无损，int2 崩塌", fontsize=12)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10, loc="center right")

fig.suptitle(f"RTN 量化误差随位宽的变化（{source}；激活为高斯模拟）", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.93))
out = os.path.join(HERE, "rtn_bitwidth.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved:", out)

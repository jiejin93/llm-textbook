# -*- coding: utf-8 -*-
"""图：方阵 GEMM 的算术强度与可达算力（RTX 4090，建模估算，非实测）。

模型口径（全部公开参数，可在本章正文核对）：
- fp16 输入/输出，方阵 n x n：FLOPs = 2n^3，HBM 流量 = 3n^2 x 2 B（A、B 各读一次，C 写一次）
  => 算术强度 AI(n) = n/3 FLOP/Byte；
- 带宽上限（TFLOPS）= 1008 GB/s x AI(n)；Tensor Core fp16 峰值取 330 TFLOPS（fp16 累加口径，
  fp32 累加减半），脊点在 AI = 330/1.008 ≈ 327，即 n ≈ 1000；
- "cuBLAS 实际达成（建模）"：效率锚点取公开基准的典型值（fp32 累加路径，大矩阵约 50%--
  55% 峰值），锚点间在 log2(n) 上线性插值。建模估算，非本地实测。
运行：python gen_gemm_ai.py（依赖 numpy + matplotlib，无需 GPU）
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

BW_TFLOPS_PER_FLOPB = 1.008      # 1008 GB/s -> 每单位 AI 贡献 1.008 TFLOPS
PEAK = 330.0                     # Tensor Core fp16（fp16 累加口径），TFLOPS

ns = np.array([64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536,
               2048, 3072, 4096, 6144, 8192, 12288, 16384], dtype=float)
ai = ns / 3.0                                    # FLOP/Byte
bw_ceiling = BW_TFLOPS_PER_FLOPB * ai            # 带宽决定的可达 TFLOPS

# cuBLAS 典型效率锚点（相对 330 TFLOPS 峰值；fp32 累加路径的公开基准量级）
anchor_n = np.array([64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384], dtype=float)
anchor_eff = np.array([0.05, 0.09, 0.16, 0.25, 0.38, 0.48, 0.52, 0.55, 0.56])
eff = np.interp(np.log2(ns), np.log2(anchor_n), anchor_eff)
achieved = PEAK * eff

fig, ax = plt.subplots(figsize=(10.5, 6.2), dpi=150)
ax.axhline(PEAK, color="gray", ls="--", lw=1.2)
ax.text(70, PEAK * 1.22, "算力屋顶：330 TFLOPS（fp16 Tensor Core，fp16 累加口径）", fontsize=9, color="dimgray")
ax.plot(ns, bw_ceiling, color="#2e7d32", ls="-.", lw=1.6,
        label="HBM 带宽上限 = 1008 GB/s × n/3（AI = n/3 FLOP/Byte）")
ax.plot(ns, achieved, color="#c62828", lw=2.4, marker="o", ms=4.5,
        label="cuBLAS 实际达成（建模估算：效率锚点插值，fp32 累加）")

ridge_n = PEAK / BW_TFLOPS_PER_FLOPB * 3
ax.axvline(ridge_n, color="#2e7d32", ls=":", lw=1.2)
ax.annotate(f"n≈{ridge_n:.0f}（AI≈327 ≈ 脊点）：\n左边的 GEMM 就算效率 100% 也摸不到算力屋顶",
            xy=(ridge_n, PEAK), xytext=(1900, 34), fontsize=9,
            arrowprops=dict(arrowstyle="->", color="dimgray"))
ax.annotate("decode GEMV（n=1）：AI≈1 FLOP/Byte，\n带宽上限只有 ≈1 TFLOPS——离屋顶 300 倍",
            xy=(64, bw_ceiling[0]), xytext=(64, 90), fontsize=9,
            arrowprops=dict(arrowstyle="->", color="dimgray"))
for n_ in (512, 2048):
    i = list(ns).index(n_)
    ax.text(ns[i] * 1.10, achieved[i] * 0.60, f"{achieved[i]:.0f} TFLOPS\n(效率{eff[i]*100:.0f}%)",
            fontsize=8.5, color="#c62828")

ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xticks([64, 256, 1024, 4096, 16384])
ax.set_xticklabels(["64", "256", "1024", "4096", "16384"])
ax.set_xlabel("方阵尺寸 n（A、B、C 均为 n×n，fp16）", fontsize=11)
ax.set_ylabel("可达算力（TFLOPS，对数轴）", fontsize=11)
ax.set_ylim(1, 900)
ax.set_title("GEMM 的算术强度随矩阵尺寸增长：小矩阵乘是“带宽的囚徒”（RTX 4090，建模估算）", fontsize=12)
ax.legend(loc="lower right", fontsize=9.5)
ax.grid(True, which="both", alpha=0.3)

fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gemm_ai_curve.png")
fig.savefig(out, dpi=150)
print("saved:", out)
for n_, e_, a_, b_ in zip(ns, eff, achieved, bw_ceiling):
    print(f"n={n_:>6.0f}  AI={n_/3:>7.1f} FLOP/B  cuBLAS建模={a_:>6.1f} TFLOPS ({e_*100:>4.1f}%)  带宽上限={min(b_, PEAK):>6.1f} TFLOPS")

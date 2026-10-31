# -*- coding: utf-8 -*-
"""图 15-8：decode 每步耗时 / 吞吐随 batch size 的 roofline 解释曲线。

解析模型（常数取 RTX 4090 + 7B fp16 的真实量级，与第 17 章 roofline 一致）：
    权重搬运时间   t_w    = N*b_param / BW = 14GB / 1008GB/s ≈ 13.9 ms（与 B 无关）
    计算时间       t_c(B) = 2*N*B / F_peak ≈ 0.0424*B ms（斜率 = 算力屋顶）
    KV 读取时间    t_kv(B) = B * T_kv * kv_bytes_per_tok / BW（随 B 线性增长）
    每步耗时       t(B) = max(t_w, t_c(B), t_kv(B))
右图吞吐 = B / t(B)：带宽受限区吞吐 ∝ B（平缓区），随后被 KV 流量或算力压平。
由代码生成，脚本同目录可复现：python gen_roofline_batch.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

# ---- 硬件与模型常数（RTX 4090，7B fp16）----
BW = 1008e9          # 显存带宽 B/s
F = 330e12           # FP16 Tensor 峰值 FLOPS（与第 17 章口径一致）
N = 7e9              # 参数量
WEIGHT_BYTES = N * 2        # fp16 权重 14 GB
T_KV = 2048                 # 假设 KV 长度
KV_MHA = 512 * 1024         # MHA 每 token KV 字节（512 KiB）
KV_GQA = 128 * 1024         # GQA-8 每 token KV 字节（128 KiB）

t_w = WEIGHT_BYTES / BW * 1000                       # ms
kv_slope = lambda kv_per_tok: T_KV * kv_per_tok / BW * 1000   # ms / batch

B = np.logspace(0, np.log10(2048), 400)              # batch 1 ~ 2048

t_w_line = np.full_like(B, t_w)
t_comp = 2 * N * B / F * 1000                        # ms
t_mha = t_w + B * kv_slope(KV_MHA)
t_gqa = t_w + B * kv_slope(KV_GQA)
t_ideal = np.maximum(t_w_line, t_comp)               # 只算权重：经典 roofline
t_real_mha = np.maximum(t_ideal, t_mha)
t_real_gqa = np.maximum(t_ideal, t_gqa)

B_STAR = t_w / (2 * N / F * 1000)                    # 理想 crossover ≈ 328
thr = lambda t: B / t * 1000                         # tok/s（t 单位 ms）

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.8))

# ---- 左图：每步耗时 ----
ax1.loglog(B, t_ideal, ls="--", color="gray", lw=2,
           label="理想 roofline：max(t_w, 2NB/F)（只算权重）")
ax1.loglog(B, t_real_mha, color="tab:red", lw=2.4,
           label=f"现实 MHA：t_w + B·T_kv·KV/BW（T_kv={T_KV}）")
ax1.loglog(B, t_real_gqa, color="tab:blue", lw=2.4,
           label="现实 GQA-8（KV 压到 1/4）")
ax1.axvline(B_STAR, color="gray", ls=":", lw=1.2)
ax1.annotate(f"理想 crossover B*≈{B_STAR:.0f}\n（B<B* 加批近乎免费）",
             xy=(B_STAR, t_w), xytext=(B_STAR * 1.15, t_w * 3.2),
             fontsize=9.5, color="dimgray",
             arrowprops=dict(arrowstyle="->", color="dimgray", lw=1))
ax1.annotate("带宽受限平台：t ≈ 13.9 ms\n（一批只搬一次权重）",
             xy=(8, t_w), xytext=(1.6, t_w * 6), fontsize=10, color="tab:red",
             arrowprops=dict(arrowstyle="->", color="tab:red", lw=1))
ax1.annotate("KV 流量随 B 线性增长\n→ 每步耗时爬坡（TPOT 变差）",
             xy=(180, t_real_mha[np.searchsorted(B, 180)]), xytext=(14, 480),
             fontsize=10, color="firebrick",
             arrowprops=dict(arrowstyle="->", color="firebrick", lw=1))
bbox_w = dict(facecolor="white", alpha=0.82, edgecolor="none", pad=1.5)
ax1.annotate("算力屋顶：只有 KV 很短、\nGQA 压缩后才逼近的陡增段",
             xy=(1500, t_ideal[np.searchsorted(B, 1500)]), xytext=(300, 22),
             fontsize=9.5, color="dimgray", bbox=bbox_w,
             arrowprops=dict(arrowstyle="->", color="dimgray", lw=1))
ax1.set_xlabel("batch size B（同批并发序列数）")
ax1.set_ylabel("单个 decode 步耗时（ms）")
ax1.set_title("每步耗时：max(搬权重, KV 流量, 算力)")
ax1.grid(alpha=0.3, which="both")
ax1.legend(loc="upper left", fontsize=9)

# ---- 右图：吞吐 ----
ax2.loglog(B, thr(t_ideal), ls="--", color="gray", lw=2, label="理想 roofline（只算权重）")
ax2.loglog(B, thr(t_real_mha), color="tab:red", lw=2.4, label="现实 MHA（T_kv=2048）")
ax2.loglog(B, thr(t_real_gqa), color="tab:blue", lw=2.4, label="现实 GQA-8")
ax2.axhline(1000 / kv_slope(KV_MHA), color="tab:red", ls=":", lw=1.2)
ax2.axhline(1000 / kv_slope(KV_GQA), color="tab:blue", ls=":", lw=1.2)
ax2.text(60, 1000 / kv_slope(KV_MHA) * 1.2,
         f"MHA 吞吐上限 ≈ {1000 / kv_slope(KV_MHA):.0f} tok/s（KV 带宽封顶）",
         fontsize=9.5, color="tab:red")
ax2.text(2.2, 1000 / kv_slope(KV_GQA) * 1.16,
         f"GQA 上限 ≈ {1000 / kv_slope(KV_GQA):.0f} tok/s",
         fontsize=9.5, color="tab:blue")
ax2.annotate("带宽受限区：吞吐 ∝ B\n（批处理的红利区）", xy=(6, thr(t_real_mha)[np.searchsorted(B, 6)]),
             xytext=(1.15, 330), fontsize=10, color="firebrick", bbox=bbox_w,
             arrowprops=dict(arrowstyle="->", color="firebrick", lw=1))
ax2.annotate("吞吐饱和：加批不再变快\n（batch 翻倍 ≠ 吞吐翻倍）",
             xy=(1000, thr(t_real_gqa)[np.searchsorted(B, 1000)]),
             xytext=(170, 70), fontsize=10, color="tab:blue",
             arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1))
ax2.set_xlabel("batch size B")
ax2.set_ylabel("系统吞吐（token/s）")
ax2.set_title("吞吐：先 ∝ B，后被带宽压平")
ax2.grid(alpha=0.3, which="both")
ax2.legend(loc="upper left", fontsize=9)

fig.suptitle("decode 时间 / 吞吐 vs batch size：roofline 解析曲线（7B fp16，RTX 4090：BW=1008 GB/s，"
             "F$_{peak}$=330 TFLOPS，t$_w$≈13.9 ms）", fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig("decode_vs_batch.png", dpi=140, bbox_inches="tight")
print("saved decode_vs_batch.png")
print(f"t_w={t_w:.2f} ms; 理想 crossover B*={B_STAR:.0f}; "
      f"MHA 上限={1000/kv_slope(KV_MHA):.0f} tok/s; GQA 上限={1000/kv_slope(KV_GQA):.0f} tok/s; "
      f"算力上限={F/(2*N):.0f} tok/s")

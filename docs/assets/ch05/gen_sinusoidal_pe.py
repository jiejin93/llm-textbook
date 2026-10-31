# gen_sinusoidal_pe.py —— 正弦位置编码的两张真实可视化
# 复现：python gen_sinusoidal_pe.py   （依赖 numpy matplotlib）
# 产物：
#   ch05_gen_sinusoidal_heatmap.png —— max_len=64, d_model=128 的正弦编码矩阵热力图
#   ch05_gen_sinusoidal_slices.png  —— 固定位置沿维度"切一刀"的曲线族
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]   # 中文坐标轴
plt.rcParams["axes.unicode_minus"] = False

MAX_LEN, D_MODEL = 64, 128

def sinusoidal_pe(max_len, d_model):
    """Transformer 原论文的正弦位置编码：
    PE(pos,2i)=sin(pos/10000^(2i/d))，PE(pos,2i+1)=cos(pos/10000^(2i/d))。"""
    pos = np.arange(max_len)[:, None]                      # (max_len, 1)
    i = np.arange(0, d_model, 2)[None, :]                  # (1, d/2)
    omega = 1.0 / (10000 ** (i / d_model))                 # 频率，几何级数
    angles = pos * omega                                   # (max_len, d/2)
    pe = np.zeros((max_len, d_model))
    pe[:, 0::2] = np.sin(angles)                           # 偶数维 sin
    pe[:, 1::2] = np.cos(angles)                           # 奇数维 cos
    return pe

pe = sinusoidal_pe(MAX_LEN, D_MODEL)

# ---- 图 1：编码矩阵热力图（横轴维度，纵轴位置） --------------------------
fig, ax = plt.subplots(figsize=(7.6, 5.4), dpi=150)
im = ax.imshow(pe, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1,
               origin="lower", interpolation="nearest")
cbar = fig.colorbar(im, ax=ax, shrink=0.9)
cbar.set_label("编码值（无量纲，值域 [-1, 1]）")
ax.set_xlabel("维度下标（左端 i=0 频率最高、周期约 6 个 token；右端频率最低、周期约 5 万 token；\n偶数维 sin 与奇数维 cos 交错形成镜像条纹）")
ax.set_ylabel("位置 pos（token 序号）")
ax.set_title(f"正弦位置编码矩阵热力图（max_len={MAX_LEN}, d_model={D_MODEL}）\n"
             "低维高频条纹区分相邻位置，高维低频色块承载长程结构")
ax.set_xticks([0, 16, 32, 48, 64, 80, 96, 112, 127])
fig.tight_layout()
fig.savefig("ch05_gen_sinusoidal_heatmap.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---- 图 2：固定位置沿维度切一刀的曲线族 ----------------------------------
fig, ax = plt.subplots(figsize=(7.6, 4.6), dpi=150)
for p in [0, 8, 16, 32, 48, 63]:
    ax.plot(np.arange(D_MODEL), pe[p], lw=1.4, label=f"pos = {p}")
ax.set_xlabel("维度下标")
ax.set_ylabel("编码值")
ax.set_title(f"固定位置的编码向量沿维度的取值（max_len={MAX_LEN}, d_model={D_MODEL}）\n"
             "同一位置在不同维度上是不同频率的 sin/cos 采样——多尺度\u201c钟摆指纹\u201d")
ax.set_xlim(0, D_MODEL - 1)
ax.grid(alpha=0.3)
ax.legend(ncol=2, fontsize=9)
fig.tight_layout()
fig.savefig("ch05_gen_sinusoidal_slices.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# 顺带打印核心性质验证：内积只依赖偏移量 k（供正文叙述核对）
dot = lambda a, b: float(a @ b)
print("PE(5)·PE(15) =", round(dot(pe[5], pe[15]), 4),
      "| PE(40)·PE(50) =", round(dot(pe[40], pe[50]), 4), "（偏移均为 10）")
print("PE(5)·PE(25) =", round(dot(pe[5], pe[25]), 4), "（偏移 20）")
print("两张图已生成：ch05_gen_sinusoidal_heatmap.png, ch05_gen_sinusoidal_slices.png")

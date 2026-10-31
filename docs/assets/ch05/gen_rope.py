# gen_rope.py —— RoPE 的两张真实可视化
# 复现：python gen_rope.py   （依赖 numpy matplotlib）
# 产物：
#   ch05_gen_rope_circle.png        —— 二维单位圆上 q,k 旋转示意（箭头 + 旋转弧）
#   ch05_gen_rope_inner_heatmap.png —— RoPE 下 q·k 内积随 (query pos, key pos) 的热力矩阵
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]   # 中文坐标轴
plt.rcParams["axes.unicode_minus"] = False

# ---- 图 3：二维单位圆上的旋转 --------------------------------------------
# 取 theta = 2*pi/24（即波长 24 个 token），m=5, n=14 → 夹角 = (n-m)*theta = 9 步
THETA = 2 * np.pi / 24
M, N = 5, 14

def rot(phi):
    return np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])

# 随机固定内容向量（不与位置对齐，图更真实）
rng = np.random.default_rng(7)
q2 = rng.normal(size=2); q2 /= np.linalg.norm(q2)
k2 = rng.normal(size=2); k2 /= np.linalg.norm(k2)
qm, kn = rot(M * THETA) @ q2, rot(N * THETA) @ k2

fig, ax = plt.subplots(figsize=(6.4, 6.4), dpi=150)
circ = np.linspace(0, 2 * np.pi, 200)
ax.plot(np.cos(circ), np.sin(circ), color="lightgray", lw=1, zorder=1)
# 原始向量（浅色虚线）与旋转后向量（实色箭头）
for v, c, ls in [(q2, "#bbbbbb", "--"), (k2, "#dddddd", ":")]:
    ax.annotate("", xy=v, xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=c, lw=1.4, linestyle=ls))
ax.annotate("", xy=qm, xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color="#1f6fd6", lw=2.6))
ax.annotate("", xy=kn, xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color="#d62728", lw=2.6))
ax.text(qm[0] * 1.12, qm[1] * 1.12, f"$R(m\\theta)q$\n（m={M}）",
        color="#1f6fd6", fontsize=12, ha="center")
ax.text(kn[0] * 1.12, kn[1] * 1.12, f"$R(n\\theta)k$\n（n={N}）",
        color="#d62728", fontsize=12, ha="center")
# 旋转弧：从原向量转到旋转后向量
ax.add_patch(Arc((0, 0), 0.9, 0.9, angle=0,
                 theta1=np.degrees(np.arctan2(q2[1], q2[0])),
                 theta2=np.degrees(np.arctan2(qm[1], qm[0])),
                 color="#1f6fd6", lw=1.2, linestyle="--"))
ax.add_patch(Arc((0, 0), 0.72, 0.72, angle=0,
                 theta1=np.degrees(np.arctan2(k2[1], k2[0])),
                 theta2=np.degrees(np.arctan2(kn[1], kn[0])),
                 color="#d62728", lw=1.2, linestyle="--"))
# q 与 k 之间的夹角弧（相对位置 (n-m)*theta）
a_q, a_k = np.arctan2(qm[1], qm[0]), np.arctan2(kn[1], kn[0])
ax.add_patch(Arc((0, 0), 1.08, 1.08, angle=0,
                 theta1=np.degrees(min(a_q, a_k)),
                 theta2=np.degrees(max(a_q, a_k)), color="#2ca02c", lw=2.2))
mid = (min(a_q, a_k) + max(a_q, a_k)) / 2
ax.text(1.18 * np.cos(mid), 1.18 * np.sin(mid),
        f"夹角 = (n-m)\u03b8 = {N-M}\u03b8", color="#2ca02c", fontsize=12, ha="center")
ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
ax.set_xlim(-1.45, 1.45); ax.set_ylim(-1.45, 1.45)
ax.set_aspect("equal")
ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
ax.set_title(f"RoPE 二维核心：位置 = 旋转角（\u03b8={THETA:.3f} rad/步）\n"
             "内积只依赖夹角 (n-m)\u03b8，绝对位置被旋转吸收", fontsize=12)
fig.tight_layout()
fig.savefig("ch05_gen_rope_circle.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---- 图 4：内积随相对位置的热力矩阵 --------------------------------------
D, T = 64, 48
inv_freq = 1.0 / (10000 ** (np.arange(0, D, 2) / D))     # (D/2,) 频率谱
pos = np.arange(T)
angles = pos[:, None] * inv_freq[None, :]                # (T, D/2)
cos_m, sin_m = np.cos(angles), np.sin(angles)            # 各 (T, D/2)

def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return np.concatenate((-x2, x1), axis=-1)

rng = np.random.default_rng(0)
q_vec = rng.normal(size=D)                               # 固定内容 q
k_vec = rng.normal(size=D)                               # 固定内容 k
qq = q_vec[None, :] * np.concatenate([cos_m, cos_m], axis=1) \
     + rotate_half(q_vec[None, :]) * np.concatenate([sin_m, sin_m], axis=1)
kk = k_vec[None, :] * np.concatenate([cos_m, cos_m], axis=1) \
     + rotate_half(k_vec[None, :]) * np.concatenate([sin_m, sin_m], axis=1)
score = qq @ kk.T                                        # (T, T) 内积矩阵

fig, ax = plt.subplots(figsize=(7.2, 5.6), dpi=150)
im = ax.imshow(score, cmap="RdBu_r", origin="lower", interpolation="nearest")
cbar = fig.colorbar(im, ax=ax, shrink=0.9)
cbar.set_label("旋转后内积 $\\langle R_m q,\\ R_n k\\rangle$")
ax.set_xlabel("key 位置 n")
ax.set_ylabel("query 位置 m")
ax.set_title(f"RoPE 下固定 (q,k) 的注意力分数矩阵（d={D}, T={T}）\n"
             "数值沿反对角线恒定 = 只依赖相对距离 m-n（对角带状结构）")
fig.tight_layout()
fig.savefig("ch05_gen_rope_inner_heatmap.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# 数值验证：同一条反对角线上的内积应几乎相等（平移不变）
d1 = [score[i, i + 7] for i in range(0, 20)]
print("相对距离 7 的内积（m=0..19）标准差:", float(np.std(d1)))
print("相对距离 7 vs 15 的内积均值:", float(np.mean([score[i, i+7] for i in range(40)])),
      float(np.mean([score[i, i+15] for i in range(32)])))
print("两张图已生成：ch05_gen_rope_circle.png, ch05_gen_rope_inner_heatmap.png")

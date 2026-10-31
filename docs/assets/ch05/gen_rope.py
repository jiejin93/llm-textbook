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
# 取 theta = 2*pi/24（即波长 24 个 token），m=5, n=14 → 夹角 = (n-m)*theta = 9 步 = 135°。
# 取 q、k 同向（同一内容方向），旋转后两向量的夹角才恰好等于位置角之差 (n-m)θ。
THETA = 2 * np.pi / 24
M, N = 5, 14
ALPHA0 = np.radians(-60)          # q、k 共同的原始方向

def rot(phi):
    return np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])

x0 = np.array([np.cos(ALPHA0), np.sin(ALPHA0)])
qm, kn = rot(M * THETA) @ x0, rot(N * THETA) @ x0

fig, ax = plt.subplots(figsize=(6.4, 6.4), dpi=150)
circ = np.linspace(0, 2 * np.pi, 200)
ax.plot(np.cos(circ), np.sin(circ), color="lightgray", lw=1, zorder=1)
# 原始向量（灰色虚线，q、k 同向）与旋转后向量（实色箭头）
ax.annotate("", xy=x0, xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="#999999", lw=1.6, linestyle="--"))
ax.annotate("", xy=qm, xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color="#1f6fd6", lw=2.6))
ax.annotate("", xy=kn, xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color="#d62728", lw=2.6))
ax.text(x0[0] * 1.16, x0[1] * 1.16, "原始向量 x\n（q、k 同向）",
        color="#777777", fontsize=11, ha="center")
ax.text(qm[0] * 1.14, qm[1] * 1.14, f"$R(m\\theta)q$\n（m={M}）",
        color="#1f6fd6", fontsize=12, ha="center")
ax.text(kn[0] * 1.14, kn[1] * 1.14, f"$R(n\\theta)k$\n（n={N}）",
        color="#d62728", fontsize=12, ha="center")
# 旋转弧：从原始方向分别转到 R(mθ)x、R(nθ)x（即位置角 mθ、nθ）
ax.add_patch(Arc((0, 0), 0.62, 0.62, angle=0,
                 theta1=np.degrees(ALPHA0), theta2=np.degrees(M * THETA),
                 color="#1f6fd6", lw=1.2, linestyle="--"))
ax.text(0.36 * np.cos((ALPHA0 + M * THETA) / 2), 0.36 * np.sin((ALPHA0 + M * THETA) / 2),
        f"$m\\theta$", color="#1f6fd6", fontsize=11, ha="center")
ax.add_patch(Arc((0, 0), 0.9, 0.9, angle=0,
                 theta1=np.degrees(ALPHA0), theta2=np.degrees(N * THETA),
                 color="#d62728", lw=1.2, linestyle="--"))
ax.text(0.52 * np.cos((ALPHA0 + N * THETA) / 2), 0.52 * np.sin((ALPHA0 + N * THETA) / 2),
        f"$n\\theta$", color="#d62728", fontsize=11, ha="center")
# 旋转后两向量之间的夹角弧（相对位置 (n-m)*theta = 135°）
ax.add_patch(Arc((0, 0), 1.1, 1.1, angle=0,
                 theta1=np.degrees(M * THETA), theta2=np.degrees(N * THETA),
                 color="#2ca02c", lw=2.2))
mid = (M + N) / 2 * THETA
ax.text(1.28 * np.cos(mid), 1.28 * np.sin(mid),
        f"夹角 = (n-m)\u03b8 = {N-M}\u03b8 = {int(np.degrees((N-M)*THETA))}\u00b0",
        color="#2ca02c", fontsize=12, ha="center")
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
             "等 m-n 斜线上数值恒定 = 只依赖相对距离（对角带状结构）")
fig.tight_layout()
fig.savefig("ch05_gen_rope_inner_heatmap.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# 数值验证：同一条 m-n 恒定斜线上的内积应几乎相等（平移不变）
d1 = [score[i, i + 7] for i in range(0, 20)]
print("相对距离 7 的内积（m=0..19）标准差:", float(np.std(d1)))
print("相对距离 7 vs 15 的内积均值:", float(np.mean([score[i, i+7] for i in range(40)])),
      float(np.mean([score[i, i+15] for i in range(32)])))
print("两张图已生成：ch05_gen_rope_circle.png, ch05_gen_rope_inner_heatmap.png")

# -*- coding: utf-8 -*-
"""图 15-7：投机解码期望加速比 vs 接受率 α（γ=4/6/8）。

按第 15 章推导的期望加速比公式真实绘制：
    S(α, γ, c) = (1 - α^(γ+1)) / ((1 - α) * (γ*c + 1))
α：草稿 token 被目标模型接受的概率；γ：每轮草稿长度；c：草稿单步开销
相对 target 单步的比例（7B 草稿 / 70B 目标约 0.1）。
阴影标出 α < 0.5 的"低接受率区"：γ 越大越容易得不偿失（S < 1）。
由代码生成，脚本同目录可复现：python gen_spec_speedup.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

C_DRAFT = 0.1          # 草稿单步开销 / target 单步开销（7B/70B 约 0.1）
GAMMAS = [4, 6, 8]
COLORS = ["tab:blue", "tab:green", "tab:red"]

alpha = np.linspace(0.02, 0.98, 400)

def speedup(a, g, c=C_DRAFT):
    return (1 - a ** (g + 1)) / ((1 - a) * (g * c + 1))

fig, ax = plt.subplots(figsize=(9.8, 6.0))

# 低接受率区（α<0.5）阴影
ax.axvspan(0, 0.5, color="mistyrose", zorder=0)
ax.text(0.25, 4.35, "低接受率区 α < 0.5\n草稿开销 γc 收不回来\n（γ 越大越先跌破 S=1）",
        ha="center", fontsize=10, color="firebrick")

for g, c in zip(GAMMAS, COLORS):
    ax.plot(alpha, speedup(alpha, g), color=c, lw=2.4, label=f"γ={g}（c={C_DRAFT}）")
    # 标出 S=1 的跌破点
    below = alpha[speedup(alpha, g) < 1.0]
    if len(below) > 0:
        a_star = below[-1]
        ax.plot(a_star, 1.0, "o", color=c, ms=7, zorder=5)

# 三个临界点标注错开排布，箭头分别指向各自交点
cross_pts = {}
for g, c in zip(GAMMAS, COLORS):
    below = alpha[speedup(alpha, g) < 1.0]
    if len(below):
        cross_pts[g] = below[-1]
text_pos = {4: (0.56, 0.98), 6: (0.56, 0.80), 8: (0.56, 0.62)}
for g, c in zip(GAMMAS, COLORS):
    if g in cross_pts:
        ax.annotate(f"γ={g}：α<{cross_pts[g]:.2f} 变慢",
                    xy=(cross_pts[g], 1.0), xytext=text_pos[g],
                    fontsize=10, color=c,
                    arrowprops=dict(arrowstyle="->", color=c, lw=1.1))

ax.axhline(1.0, color="gray", ls="--", lw=1.2)
ax.text(0.975, 1.06, "S=1：与普通 decode 打平", ha="right", fontsize=9.5, color="gray")

# 甜蜜点标注
a_sweet = 0.9
ax.annotate(f"甜蜜点：α=0.9、γ=8 时\nS≈{speedup(a_sweet, 8):.1f}×",
            xy=(a_sweet, speedup(a_sweet, 8)), xytext=(0.70, 4.15),
            fontsize=10, color="tab:red",
            arrowprops=dict(arrowstyle="->", color="tab:red", lw=1))

ax.set_xlim(0, 1.0)
ax.set_ylim(0.42, 4.8)
ax.set_xlabel("接受率 α（草稿 token 被 target 接受的概率）")
ax.set_ylabel("期望加速比 S（相对普通 decode）")
ax.set_title("投机解码期望加速比 vs 接受率 α\n"
             r"$S=\dfrac{1-\alpha^{\gamma+1}}{(1-\alpha)(\gamma c+1)}$"
             f"，草稿开销 c={C_DRAFT}（由代码按公式真实绘制）")
ax.grid(alpha=0.3)
ax.legend(loc="upper left", fontsize=10)

fig.tight_layout()
fig.savefig("spec_speedup_vs_alpha.png", dpi=140)
print("saved spec_speedup_vs_alpha.png")
for g in GAMMAS:
    below = alpha[speedup(alpha, g) < 1.0]
    a_star = below[-1] if len(below) else float("nan")
    print(f"γ={g}: S<1 的临界 α ≈ {a_star:.3f}; S(α=0.5)={speedup(0.5, g):.2f}; "
          f"S(α=0.8)={speedup(0.8, g):.2f}; S(α=0.9)={speedup(0.9, g):.2f}")

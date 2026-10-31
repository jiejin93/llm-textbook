# -*- coding: utf-8 -*-
"""图①：ViT 的 patch 切分可视化——读入一张真实照片，中心裁方后按 14×14 叠加网格。

左：整图 + 14×14 网格与按行编号的几个 patch；右：把其中一个 patch 展平成
16×16×3=768 维向量（线性投影后即 1 个视觉 token）。
照片来源：skimage.data.astronaut()（NASA，公有领域）。
脚本同目录可复现：python gen_patch_grid.py
"""
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage import data

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

P, S = 16, 224                     # patch 大小 16、输入分辨率 224（ViT-B/16）；14×14=196 个 patch

# ---- 1) 读入真实照片：中心裁方 + 缩放到 224×224 ------------------------------
img = Image.fromarray(data.astronaut()).convert("RGB")
w, h = img.size
side = min(w, h)
img = img.crop(((w - side) // 2, (h - side) // 2,
                (w + side) // 2, (h + side) // 2)).resize((S, S))
arr = np.asarray(img)
n = S // P                         # 14

# ---- 2) 左图：叠加网格，高亮并编号部分 patch --------------------------------
fig = plt.figure(figsize=(10.6, 5.2), dpi=150)
gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 0.72, 0.16], wspace=0.28)
ax = fig.add_subplot(gs[0])
ax.imshow(arr)
for k in range(1, n):
    ax.axhline(k * P, color="yellow", lw=0.7, alpha=0.85)
    ax.axvline(k * P, color="yellow", lw=0.7, alpha=0.85)
ax.set_xticks([]);
ax.set_yticks([])
# 按行优先顺序给 patch 标号（展示“图像 → 序列”的排布方式），只标一部分避免拥挤
shown = [0, 5, 27, 47, 55, 98, 130, 180, 195]
for idx in shown:
    r, c = divmod(idx, n)
    ax.add_patch(plt.Rectangle((c * P, r * P), P, P, fill=False,
                               color="red", lw=1.6))
    ax.text(c * P + P / 2, r * P + P / 2, str(idx), color="red", fontsize=8.5,
            ha="center", va="center", weight="bold",
            bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.6))
ax.set_title(f"输入 224×224，切成 {n}×{n}={n*n} 个 16×16 patch\n"
             "（黄色网格；红框 patch 按行优先编号 → 一维序列）", fontsize=10.5)

# ---- 3) 中图：选中 patch；右图：展平成 768 维向量条 --------------------------
sel = 47
r, c = divmod(sel, n)
patch = arr[r * P:(r + 1) * P, c * P:(c + 1) * P]          # 16×16×3
vec = patch.reshape(-1).astype(float)                       # 768 维

ax = fig.add_subplot(gs[1])
ax.imshow(patch, interpolation="nearest")
ax.set_xticks([]);
ax.set_yticks([])
ax.set_title(f"第 {sel} 号 patch\n16×16×3 = 768 维", fontsize=10.5)
ax.set_xlabel("线性投影后 = 1 个视觉 token", fontsize=10)

ax_bar = fig.add_subplot(gs[2])
ax_bar.imshow(vec.reshape(48, 16), cmap="viridis", aspect="auto")
ax_bar.set_xticks([]);
ax_bar.set_yticks([])
ax_bar.set_xlabel("展平\n$\in\mathbb{R}^{768}$", fontsize=9)

fig.suptitle("ViT：图像即序列——切 patch、展平、线性投影、加位置嵌入，交给标准 Transformer",
             fontsize=12, y=1.0)
fig.savefig(__file__.replace("gen_patch_grid.py", "patch_grid.png"),
            dpi=150, bbox_inches="tight")
print(f"图像尺寸: {w}x{h} -> 裁方缩放 {S}x{S}, patch {P}px, 网格 {n}x{n}")
print(f"第 {sel} 号 patch 展平维度: {vec.shape[0]}（=16*16*3）")
print("已保存 patch_grid.png")

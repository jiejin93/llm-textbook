# gen_multihead_toy.py —— 多头 vs 单头的真实训练对照实验（CPU 可跑，约 10 分钟）
# 复现：python gen_multihead_toy.py   （依赖 torch matplotlib numpy）
# 任务：莎士比亚字符级语言模型（与 6.18 节同一语料，取自 ../ch04/data_en.txt 前 40 万字符）
# 对照：两个小模型（2 层、d=128、T=64），一个 h=1（单头）、一个 h=4（多头），同种子同数据
# 产物：ch06_gen_multihead_patterns.png —— 层 2 各头注意力权重热图网格（每头一格）
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # Windows 下 torch/matplotlib 的 OpenMP 冲突规避
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
torch.manual_seed(0)

# ---------------- 数据：字符级莎士比亚 ----------------
text = open("../ch04/data_en.txt", encoding="utf-8").read()[:400_000]
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n_train = int(len(data) * 0.95)
T, BATCH, D, N_LAYER = 64, 32, 128, 2

def get_batch(split):
    d = data[:n_train] if split == "train" else data[n_train:]
    ix = torch.randint(0, len(d) - T - 1, (BATCH,))
    x = torch.stack([d[i:i + T] for i in ix])
    y = torch.stack([d[i + 1:i + T + 1] for i in ix])
    return x, y

# ---------------- 模型：本章 6.17 节 minigpt 的紧凑版 ----------------
class Attn(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.last_att = None                       # 记录 softmax 后权重供可视化

    def forward(self, x):
        B, Tt, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=2)
        q = q.view(B, Tt, self.h, d // self.h).transpose(1, 2)
        k = k.view(B, Tt, self.h, d // self.h).transpose(1, 2)
        v = v.view(B, Tt, self.h, d // self.h).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / (d // self.h) ** 0.5
        mask = torch.triu(torch.ones(Tt, Tt, dtype=torch.bool), 1)
        att = att.masked_fill(mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        self.last_att = att.detach()               # (B, h, T, T)
        y = (att @ v).transpose(1, 2).reshape(B, Tt, d)
        return self.proj(y)

class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn, self.mlp = Attn(d, h), nn.Sequential(
            nn.Linear(d, 4 * d, bias=False), nn.GELU(),
            nn.Linear(4 * d, d, bias=False))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))

class TinyGPT(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.emb = nn.Embedding(len(chars), D)
        self.pos = nn.Embedding(T, D)
        self.blocks = nn.ModuleList([Block(D, h) for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(D)
        self.head = nn.Linear(D, len(chars), bias=False)
        self.head.weight = self.emb.weight
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, std=0.02)

    def forward(self, idx, targets=None):
        x = self.emb(idx) + self.pos(torch.arange(T))
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.ln_f(x))
        loss = F.cross_entropy(logits.view(-1, len(chars)), targets.view(-1)) \
            if targets is not None else None
        return logits, loss

def train(model, steps=4000):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    for step in range(steps + 1):
        x, y = get_batch("train")
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 500 == 0:
            print(f"  step {step:5d} loss {loss.item():.3f}", flush=True)
    model.eval()
    with torch.no_grad():
        vx, vy = get_batch("val")
        vloss = model(vx, vy)[1].item()
    return vloss

print("训练单头模型 (h=1) ...", flush=True)
m1 = TinyGPT(h=1); loss1 = train(m1)
print("训练多头模型 (h=4) ...", flush=True)
m4 = TinyGPT(h=4); loss4 = train(m4)
print(f"最终 val loss: 单头 h=1 = {loss1:.3f} | 多头 h=4 = {loss4:.3f}", flush=True)

# ---------------- 可视化：同一段文本上各头的注意力 ----------------
# 选一段含重复子串的文本（诱导模式可见），从验证区找，长度必须等于 T
sample_raw = None
val_text = text[n_train:n_train + 6000]
for i in range(0, 2500, 4):
    seg = val_text[i:i + T]
    anchor = seg[:28]
    if len(anchor) >= 20 and val_text.find(anchor, i + 10) != -1:
        sample_raw = seg; break
if sample_raw is None:
    sample_raw = val_text[:T]
assert len(sample_raw) == T
sample = torch.tensor([stoi[c] for c in sample_raw])[None, :]

def head_maps(model, layer=1):
    with torch.no_grad():
        model(sample)
        att = model.blocks[layer].attn.last_att[0]     # (h, T, T)
    return att.numpy()

maps1 = head_maps(m1)
maps4 = head_maps(m4)
S = len(sample_raw)
fig, axes = plt.subplots(2, 4, figsize=(15.6, 8.4), dpi=150)

def draw(ax, amap, title):
    im = ax.imshow(amap, cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([0, S - 1]); ax.set_xticklabels([1, S], fontsize=8)
    ax.set_yticks([0, S - 1]); ax.set_yticklabels([1, S], fontsize=8)
    return im

off1 = lambda a: float(np.mean([a[t, t - 1] for t in range(1, S)]))
ent = lambda a: float(-(np.clip(a, 1e-9, 1) * np.log(np.clip(a, 1e-9, 1))).sum(-1).mean())
im = draw(axes[0, 0], maps1[0],
          f"单头模型 h=1（val loss={loss1:.3f}）\n前一词占比={off1(maps1[0]):.2f} "
          f"平均熵={ent(maps1[0]):.2f}")
for ax in axes[0, 1:]:
    ax.axis("off")
axes[0, 0].set_ylabel("query 位置（字符序号）")
for j in range(4):
    a = maps4[j]
    draw(axes[1, j], a, f"多头模型 h=4，头 {j}（val loss={loss4:.3f}）\n"
        f"前一词占比={off1(a):.2f} 平均熵={ent(a):.2f}")
axes[1, 0].set_ylabel("query 位置（字符序号）")
fig.suptitle("同一个字符级语言模型任务：单头只能形成一种混合模式，多头各头各司其职"
             "（横轴 key 位置；样本文本：" + repr(sample_raw[:24]) + "...）",
             fontsize=12)
fig.colorbar(im, ax=axes[:, 3], shrink=0.8, label="注意力权重")
fig.savefig("ch06_gen_multihead_patterns.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("图已生成：ch06_gen_multihead_patterns.png")

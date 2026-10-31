# -*- coding: utf-8 -*-
"""字符级 bigram 语言模型训练演示：loss 下降曲线 + 训练前后生成样本对比。
由代码生成，脚本同目录可复现：python gen_bigram_demo.py
依赖：torch >= 2.0 (CPU 即可), matplotlib
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

torch.manual_seed(0)

# ---------- 语料：句式多样的小语料（读者可换成任意 txt） ----------
sentences = [
    "语言模型预测下一个词。",
    "输入法联想的也是下一个词。",
    "预测得越准，模型越像会说话。",
    "训练就是把猜错的程度一点点压低。",
    "大模型读过的文本越多，句子越通顺。",
    "生成的每个字都来自一个概率分布。",
    "模型并不理解世界，它只是很会猜。",
    "数据的规模和质量决定能力的上限。",
    "注意力机制让模型看到更远的上下文。",
    "算力换能力，规模定律说的是这件事。",
    "对话模型要先听懂指令，再组织答案。",
    "一个好的分词器能省下大量算力。",
    "损失下降到很低，生成反而失去多样性。",
    "温度越高，采样出的句子越发散。",
    "工程师每天都要在速度和质量之间权衡。",
    "开源社区把大模型的能力带给了每个人。",
]
corpus = "".join(sentences * 12)
chars = sorted(set(corpus))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
V = len(chars)
data = torch.tensor([stoi[c] for c in corpus])
print(f"语料 {len(corpus)} 字符，词表 V={V}，ln(V)={torch.log(torch.tensor(float(V))):.3f}")

x = data[:-1]          # 每个位置的输入字符
y = data[1:]           # 标签 = 输入的下一个字符

logits_table = torch.zeros(V, V, requires_grad=True)   # bigram 打分表
opt = torch.optim.Adam([logits_table], lr=0.08)

losses = []
snap_steps = {1, 200, 6000}
snaps = {}

@torch.no_grad()
def generate(start="语", n=48, temp=1.0):
    out, cur = start, stoi[start]
    for _ in range(n):
        p = F.softmax(logits_table[cur] / temp, dim=-1)
        cur = torch.multinomial(p, 1).item()
        out += itos[cur]
    return out

for step in range(6001):
    loss = F.cross_entropy(logits_table[x], y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    losses.append(loss.item())
    if step in snap_steps:
        snaps[step] = (loss.item(), generate())

# ---------- 画图 ----------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=150,
                               gridspec_kw={"width_ratios": [1, 1.35]})
ax1.semilogx(range(1, len(losses) + 1), losses, color="#c0504d", lw=1.8)
lnv = torch.log(torch.tensor(float(V))).item()
ax1.axhline(lnv, color="#888888", ls="--", lw=1.2)
ax1.text(2.2, lnv + 0.10, f"瞎猜下限 ln(V)={lnv:.2f}", fontsize=9, color="#666666")
label_pos = {1: (32, -26), 200: (14, 12), 6000: (40, -26)}
for s in snap_steps:
    ax1.scatter([s], [losses[s]], color="#2e6da4", zorder=5)
    ax1.annotate(f"第{s}步\nloss={losses[s]:.2f}", (s, losses[s]),
                 textcoords="offset points", xytext=label_pos[s], fontsize=8.5,
                 color="#2e6da4", ha="center")
ax1.set_xlim(0.8, 12000)
ax1.set_ylim(0.15, 5.7)
ax1.set_xlabel("训练步数（log 尺度）", fontsize=10.5)
ax1.set_ylabel("交叉熵损失（nat）", fontsize=10.5)
ax1.set_title("(a) bigram 训练：损失从瞎猜水平持续下降", fontsize=11)
ax1.grid(ls="--", alpha=0.4)

ax2.axis("off")
ax2.set_title("(b) 同一模型不同训练程度下的续写（起点均为“语”）", fontsize=11)
lines = [f"第 {s:>4} 步  loss={snaps[s][0]:.2f}\n{snaps[s][1]}\n" for s in sorted(snaps)]
ax2.text(0.0, 0.92, "\n".join(lines), fontsize=10.5,
         va="top", linespacing=1.7)

fig.suptitle("字符级 bigram 语言模型：训练让“胡话”逐渐变成“像话”",
             fontsize=13, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(__file__.replace("gen_bigram_demo.py", "bigram_demo.png"),
            bbox_inches="tight")
print("saved: bigram_demo.png")
for s in sorted(snaps):
    print(f"step {s:>4}  loss={snaps[s][0]:.3f}  样本: {snaps[s][1]}")

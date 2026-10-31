# -*- coding: utf-8 -*-
"""
去重前后文档间 n-gram 重合度分布对比（toy 语料上的真实计算）：
1) 构造 80 篇文档：40 篇原创 + 20 篇"换 5% 词"的近重复 + 4 篇逐字副本 +
   8 篇短正文长页脚模板页 + 8 篇"正文+同款页脚"文档；
2) 对全部 C(80,2)=3160 个文档对，精确计算字符 8-gram 集合的 Jaccard 相似度；
3) 执行两步去重：精确哈希去重 + 贪心近重复删除（J >= 0.6 的后出现者删除）；
4) 对比重去重前后全部文档对的相似度直方图。
运行：python gen_dedup_ngram.py  （依赖 numpy + matplotlib）
"""
import hashlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
rng = np.random.default_rng(1)

CHARS = "的一是不了在有人我他这个们中来上大为和国地到以说时要就出会可也你对生能而子那得于着下自之年过发后作里用道行所然家种事成方多经么去法学如都同现当没动面起看定天分还进好小部其些主样理心她本前开但因只从想实日军者意无力它与长把机十民第公此已工使情明性知全三又关点正业外两高间问理领白更几文立空手电门公持天气网络数据模型训练学习系统"
WORDS = [CHARS[i:i+2] for i in range(0, len(CHARS) - 1, 2)]
FOOTER = "本站版权所有转载必究联系方式见页面底部免责声明使用前必读同意条款方可继续访问本站"

def rand_doc(n):
    return "".join(rng.choice(WORDS, size=n))

def shingles(t, k=8):
    return set(t[i:i+k] for i in range(len(t) - k + 1)) if len(t) >= k else {t}

def jac(a, b):
    u = a | b
    return len(a & b) / len(u) if u else 0.0

# ---------- 1. 构造语料 ----------
docs = {}
for i in range(40):
    docs[f"o{i}"] = rand_doc(int(rng.integers(35, 55)))
for i in range(20):                                   # 近重复：换 2% 的词（≈1 个词，且确保真换）
    w = list(docs[f"o{i}"])
    pos = rng.choice(len(w), size=max(1, int(len(w)*0.02)), replace=False)
    for t in pos:
        old = w[t]
        while w[t] == old:
            w[t] = rng.choice(WORDS)
    docs[f"n{i}"] = "".join(w)
for i in range(30, 34):                               # 逐字副本
    docs[f"c{i}"] = docs[f"o{i}"]
for i in range(8):                                    # 模板页：短正文 + 长页脚
    docs[f"s{i}"] = rand_doc(6) + FOOTER
for i in range(8):                                    # 正文 + 同款页脚
    docs[f"f{i}"] = docs[f"o{i}"] + FOOTER
ids = sorted(docs)
print(f"语料共 {len(ids)} 篇")

# ---------- 2. 全对相似度（去重前） ----------
sh = {d: shingles(docs[d]) for d in ids}
pairs = [(a, b, jac(sh[a], sh[b])) for i, a in enumerate(ids) for b in ids[i+1:]]
J_before = np.array([p[2] for p in pairs])

# ---------- 3. 精确去重 + 贪心近重复删除（按随机流式顺序处理） ----------
stream = list(ids)
rng.shuffle(stream)                                   # 模拟真实流水线的流式到达顺序
seen_hash, kept, removed_exact, removed_near = set(), [], 0, 0
for d in stream:
    h = hashlib.md5(docs[d].encode()).hexdigest()
    if h in seen_hash:
        removed_exact += 1
        continue
    seen_hash.add(h)
    if any(jac(sh[d], sh[k]) >= 0.6 for k in kept):    # 贪心：与任一已保留文档高相似即删
        removed_near += 1
        continue
    kept.append(d)
J_after = np.array([jac(sh[a], sh[b]) for i, a in enumerate(kept) for b in kept[i+1:]])
dup_mass_before = (J_before >= 0.6).mean() * 100
dup_mass_after = (J_after >= 0.6).mean() * 100
print(f"精确去重删除 {removed_exact} 篇，近重复删除 {removed_near} 篇，保留 {len(kept)} 篇")
print(f"相似度>=0.6 的文档对占比：去重前 {dup_mass_before:.2f}% -> 去重后 {dup_mass_after:.3f}%")

# ---------- 4. 画分布对比 ----------
bins = np.arange(0, 1.02, 0.02)
fig, ax = plt.subplots(figsize=(9.5, 5.4))
ax.hist(J_before, bins=bins, alpha=0.65, label=f"去重前（{len(ids)} 篇，{len(J_before)} 对）",
        color="#d62728", log=True)
ax.hist(J_after, bins=bins, alpha=0.75, label=f"去重后（{len(kept)} 篇，{len(J_after)} 对）",
        color="#1f77b4", log=True)
ax.annotate("近重复/副本对：J≈0.8~1.0\n（20 近重复 + 4 逐字副本）",
            xy=(0.88, (J_before >= 0.8).sum()), xytext=(0.46, 300),
            fontsize=10.5, color="#d62728",
            arrowprops=dict(arrowstyle="->", color="#d62728"))
ax.annotate("页脚模板带来的中等相似对\n（8 篇模板页 + 8 篇带页脚正文）",
            xy=(0.17, (np.abs(J_before-0.17) < 0.02).sum()), xytext=(0.24, 1500),
            fontsize=10.5, color="#7f4fc9",
            arrowprops=dict(arrowstyle="->", color="#7f4fc9"))
ax.set_xlabel("文档对相似度：字符 8-gram 集合的 Jaccard", fontsize=12)
ax.set_ylabel("文档对数量（对数轴）", fontsize=12)
ax.set_title("去重前后文档间 n-gram 重合度分布：高相似峰消失，页脚中等峰仍在", fontsize=13)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("gen_dedup_ngram.png", dpi=150, bbox_inches="tight")
print("已保存 gen_dedup_ngram.png")

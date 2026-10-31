# -*- coding: utf-8 -*-
"""
MinHash + LSH 真实模拟实验：
1) 构造 200 篇模拟文档（20 个基底 + 180 个受控改写变体，覆盖不同真实 Jaccard 相似度）；
2) 对每一对"基底-变体"精确计算字符 5-gram 的 Jaccard 相似度作为真值；
3) 用仿射变换近似随机全序哈希，计算 k=100 维 MinHash 签名（b=20 个 band × r=5 行），
   独立重复 100 次（每次换哈希种子），统计每个相似度档被 LSH 检出为候选对的频率；
4) 与解析 S 曲线 1-(1-J^r)^b 对比，并画出 FineWeb / RefinedWeb 的公开参数曲线。
运行：python gen_minhash_lsh.py  （依赖 numpy + matplotlib，约 20 秒）
"""
import hashlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(0)

# ---------- 1. 构造 200 篇模拟文档 ----------
CHARS = "的一是不了在有人我他这个们中来上大为和国地到以说时要就出会可也你对生能而子那得于着下自之年过发后作里用道行所然家种事成方多经么去法学如都同现当没动面起看定天分还进好小部其些主样理心她本前开但因只从想实日军者意无力它与长把机十民第公此已工使情明性知全三又关点正业外两高间问理领白更几文立空手门公持而是天本研究"
WORDS = [CHARS[i:i+2] for i in range(0, len(CHARS) - 1, 2)]  # 伪词表：二字组合
def rand_doc(n_words):
    return "".join(rng.choice(WORDS, size=n_words))

def shingles(text, k=5):
    return set(text[i:i+k] for i in range(len(text) - k + 1))

n_base, n_var = 20, 9
docs = {}          # doc_id -> 文本
pairs_true = []    # (base_id, var_id, 真实 Jaccard)
base_word_lists = [list(rng.choice(WORDS, size=40)) for _ in range(n_base)]
for i, wl in enumerate(base_word_lists):
    docs[f"b{i}"] = "".join(wl)
for i, wl in enumerate(base_word_lists):
    for j, f in enumerate(np.linspace(0.05, 0.95, n_var)):   # 替换比例 → 相似度梯度
        v = wl.copy()
        idx = rng.choice(len(v), size=int(len(v) * f), replace=False)
        for t in idx:
            v[t] = rng.choice(WORDS)
        did = f"v{i}_{j}"
        docs[did] = "".join(v)
        J = len(shingles(docs[f"b{i}"]) & shingles(docs[did])) / len(shingles(docs[f"b{i}"]) | shingles(docs[did]))
        pairs_true.append((f"b{i}", did, J))
print(f"文档数 {len(docs)}，受控相似度对 {len(pairs_true)} 个；"
      f"真实 Jaccard 范围 {min(p[2] for p in pairs_true):.2f} ~ {max(p[2] for p in pairs_true):.2f}")

# ---------- 2. 精确 MinHash + LSH（仿射变换近似随机全序） ----------
P = (1 << 31) - 1            # 梅森素数 2^31-1，仿射哈希模数
K, B, R = 100, 20, 5         # 签名 100 维 = 20 band × 5 行；阈值 (1/20)^(1/5) ≈ 0.55

ids = list(docs.keys())
base_hashes = []             # 每个 shingle 一次 md5 → 稳定的基哈希
for d in ids:
    s = np.array([int(hashlib.md5(x.encode()).hexdigest()[:8], 16) % P
                  for x in shingles(docs[d])], dtype=np.int64)
    base_hashes.append(s)
Smax = max(len(s) for s in base_hashes)
X = np.full((len(ids), Smax), P, dtype=np.int64)   # 用 P 填充哨兵
for row, s in enumerate(base_hashes):
    X[row, :len(s)] = s
id2row = {d: i for i, d in enumerate(ids)}

def lsh_detect(trials=100):
    counts = np.zeros(len(pairs_true), dtype=np.int32)   # 每对被检出的次数
    for _ in range(trials):
        a = rng.integers(1, P, size=K, dtype=np.int64)
        b = rng.integers(0, P, size=K, dtype=np.int64)
        sig = ((X[None, :, :] * a[:, None, None] + b[:, None, None]) % P).min(axis=2)  # (K, n_docs)
        buckets = {}
        for j in range(B):
            key = j
            for col in range(sig.shape[1]):
                band = tuple(sig[j*R:(j+1)*R, col])
                buckets.setdefault((key, band), []).append(col)
        cand = set()
        for members in buckets.values():
            if len(members) > 1:
                for x in range(len(members)):
                    for y in range(x+1, len(members)):
                        cand.add((members[x], members[y]))
        for t, (d1, d2, _) in enumerate(pairs_true):
            if (id2row[d1], id2row[d2]) in cand:
                counts[t] += 1
    return counts

hit = lsh_detect(100)
per_pair = hit / 100.0                                # 每对的检出频率
Js = np.array([p[2] for p in pairs_true])

# ---------- 3. 按真实 Jaccard 分桶统计检出频率 ----------
edges = np.arange(0, 1.0001, 0.075)
centers, freq, err = [], [], []
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (Js >= lo) & (Js < hi)
    if m.sum() >= 3:
        centers.append(0.5*(lo+hi))
        freq.append(per_pair[m].mean())
        err.append(per_pair[m].std() / np.sqrt(m.sum()))
centers, freq, err = map(np.array, (centers, freq, err))

def s_curve(J, b, r):
    return 1 - (1 - J**r)**b
Jx = np.linspace(0, 1, 400)
t_thr = (1/B)**(1/R)

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
ax = axes[0]
ax.errorbar(centers, freq, yerr=err, fmt="o", ms=6, capsize=3, color="#1f77b4",
            label="模拟检出频率（100 次重复，2000 对样本）")
ax.plot(Jx, s_curve(Jx, B, R), "-", color="#d62728", lw=2,
        label=f"解析式 $1-(1-J^{{{R}}})^{{{B}}}$（$b={B},r={R}$）")
ax.axvline(t_thr, ls="--", color="#888888", lw=1.2)
ax.text(t_thr-0.33, 0.42, f"拐点阈值 $t\\approx(1/b)^{{1/r}}\\approx{t_thr:.2f}$", fontsize=11)
ax.set_xlabel("两文档真实 Jaccard 相似度（字符 5-gram）", fontsize=12)
ax.set_ylabel("被判为候选重复对的概率", fontsize=12)
ax.set_title("MinHash+LSH 检出概率：模拟 vs 解析", fontsize=13)
ax.legend(fontsize=10, loc="upper left"); ax.grid(alpha=0.3); ax.set_ylim(-0.03, 1.05)

ax = axes[1]
for b, r, c, name in [(20, 5, "#1f77b4", "本实验 b=20, r=5（k=100）"),
                      (14, 8, "#2ca02c", "FineWeb b=14, r=8（k=112）"),
                      (450, 20, "#ff7f0e", "RefinedWeb b=450, r=20（k=9000）")]:
    ax.plot(Jx, s_curve(Jx, b, r), color=c, lw=2, label=name)
ax.set_xlabel("两文档真实 Jaccard 相似度", fontsize=12)
ax.set_ylabel("检出概率", fontsize=12)
ax.set_title("不同 $(b,r)$ 配置的 S 曲线：调参=平移阈值", fontsize=13)
ax.legend(fontsize=10, loc="upper left"); ax.grid(alpha=0.3)

fig.suptitle("MinHash+LSH 候选对检出概率 S 曲线：模拟与解析一致，配置决定阈值", fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig("gen_minhash_lsh.png", dpi=150, bbox_inches="tight")
cross = np.where(freq >= 0.5)[0]
t_emp = float(np.interp(0.5, freq[cross[0]-1:cross[0]+1], centers[cross[0]-1:cross[0]+1])) if len(cross) else float("nan")
print(f"经验拐点（检出频率穿越 0.5）约 J={t_emp:.2f}（理论 {t_thr:.2f}）")
print("已保存 gen_minhash_lsh.png")

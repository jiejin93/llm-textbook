# gen_bpe_vocab_scaling.py —— 真实训练不同词表大小的 BPE，画压缩率曲线 + 词素比柱状图
# 复现：python gen_bpe_vocab_scaling.py   （依赖 pip install tokenizers matplotlib）
# 语料（均为公有领域文本，与本脚本同目录）：
#   data_en.txt —— tinyshakespeare 戏剧全集，约 111 万字符
#                   （https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt）
#   data_zh.txt —— 《孟子》全文，约 4.8 万字符（zh.wikisource.org，繁体转简体）
# 产物：
#   ch04_gen_vocab_compression.png —— 词表大小 vs 压缩率（token 数/字符数）
#   ch04_gen_fertility.png         —— 词素比 fertility（tok/词、tok/字）对比柱状图
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]   # 中文坐标轴
plt.rcParams["axes.unicode_minus"] = False

VOCAB_SIZES = [1000, 2000, 4000, 8000]

def train_bpe(lines, vocab_size):
    """在给定语料行上真实训练 byte-level BPE（HF tokenizers 的 Rust 实现）。"""
    tk = Tokenizer(models.BPE(unk_token=None))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # 256 字节全保留
    )
    tk.train_from_iterator(lines, trainer)
    return tk

def measure(tk, sample_lines, lang):
    """在 held-out 语料上测压缩率（token/字符）与词素比（token/词 或 token/字）。"""
    n_tok = sum(len(tk.encode(s).ids) for s in sample_lines)
    n_chr = sum(len(s) for s in sample_lines)
    text = "".join(sample_lines)
    if lang == "en":                       # 英文语言单元 = 词
        units = len(re.findall(r"[A-Za-z']+", text))
    else:                                  # 中文语言单元 = 汉字
        units = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    return n_tok / n_chr, n_tok / units

results = {}
for lang, fname in [("en", "data_en.txt"), ("zh", "data_zh.txt")]:
    text = open(fname, encoding="utf-8").read()
    lines = [ln for ln in text.splitlines() if ln.strip()]
    cut = int(len(lines) * 0.8)            # 前 80% 训练 / 后 20% held-out 评测
    train_lines, eval_lines = lines[:cut], lines[cut:]
    print(f"[{lang}] train lines={len(train_lines)}, eval chars="
          f"{sum(len(s) for s in eval_lines)}")
    results[lang] = {"cr": [], "fert": []}
    for V in VOCAB_SIZES:
        tk = train_bpe(train_lines, V)
        cr, fert = measure(tk, eval_lines, lang)
        results[lang]["cr"].append(cr)
        results[lang]["fert"].append(fert)
        print(f"[{lang}] V={V:5d}  压缩率={cr:.3f} tok/字符  词素比={fert:.2f}")

# ---- 图 1：词表大小 vs 压缩率 --------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
x = range(len(VOCAB_SIZES))
ax.plot(x, results["en"]["cr"], "o-", color="#1f6fd6", lw=2, label="英文（莎剧全集）")
ax.plot(x, results["zh"]["cr"], "s-", color="#d62728", lw=2, label="中文（《孟子》全文）")
for i, (en_v, zh_v) in enumerate(zip(results["en"]["cr"], results["zh"]["cr"])):
    ax.annotate(f"{en_v:.3f}", (i, en_v), textcoords="offset points",
                xytext=(0, -14), ha="center", color="#1f6fd6", fontsize=9)
    ax.annotate(f"{zh_v:.3f}", (i, zh_v), textcoords="offset points",
                xytext=(0, 8), ha="center", color="#d62728", fontsize=9)
ax.set_xticks(list(x))
ax.set_xticklabels([f"{V//1000}k" for V in VOCAB_SIZES])
ax.set_xlabel("词表大小 V")
ax.set_ylabel("压缩率（token 数 / 字符数，越低越省）")
ax.set_title("词表大小对压缩率的影响：真实训练 byte-level BPE（held-out 语料）")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig("ch04_gen_vocab_compression.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---- 图 2：词素比 fertility 柱状图 ---------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
w = 0.36
b1 = ax.bar([i - w / 2 for i in x], results["en"]["fert"], w,
            color="#1f6fd6", label="英文：平均每词 token 数")
b2 = ax.bar([i + w / 2 for i in x], results["zh"]["fert"], w,
            color="#d62728", label="中文：平均每汉字 token 数")
for bars in (b1, b2):
    for r in bars:
        ax.annotate(f"{r.get_height():.2f}", (r.get_x() + r.get_width() / 2,
                    r.get_height()), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=9)
ax.axhline(1.0, color="gray", ls="--", lw=1)
ax.annotate("fertility = 1：一词/一字一个 token（最省）", (0.02, 0.93),
            xycoords="axes fraction", color="gray", fontsize=9)
ax.set_xticks(list(x))
ax.set_xticklabels([f"{V//1000}k" for V in VOCAB_SIZES])
ax.set_xlabel("词表大小 V")
ax.set_ylabel("词素比 fertility（token 数 / 语言单元数）")
ax.set_title("词素比 fertility：中文对词表大小远比英文敏感（同一实验）")
ax.grid(alpha=0.3, axis="y")
ax.legend()
fig.tight_layout()
fig.savefig("ch04_gen_fertility.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("两张图已生成：ch04_gen_vocab_compression.png, ch04_gen_fertility.png")

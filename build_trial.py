# -*- coding: utf-8 -*-
"""Build the free-trial edition of the textbook site into llm-textbook/docs/ for GitHub Pages."""
import os, re, json, shutil, html
import markdown

ROOT = os.path.dirname(os.path.abspath(__file__))
BOOK = os.path.join(os.path.dirname(ROOT), "llm-book", "new")   # 最新校订稿
SRC = os.path.join(BOOK, "chapters")
RENDERED = os.path.join(BOOK, "chapters_rendered")
OUT = os.path.join(ROOT, "docs")

MATH_STORE = json.load(open(os.path.join(RENDERED, "_math.json"), encoding="utf-8"))
MANIFEST = [
    ("math.md", "第 0 章 数学基础"),
    ("ch00.md", "前言"),
    ("ch01.md", "第 1 章 绪论：大语言模型全景"),
    ("ch02.md", "第 2 章 语言模型基础"),
    ("ch03.md", "第 3 章 Transformer 详解"),
    ("ch04.md", "第 4 章 分词器"),
]
REPO = "https://github.com/jiejin93/llm-textbook"
QR = os.path.join(ROOT, "branding", "二维码.jpg")
QR_NAME = "wechat-qr.jpg"
HAS_QR = os.path.exists(QR)

def md2html(text, fname):
    math = MATH_STORE.get(fname, [])
    text = re.sub(r"\x00MATH(\d+)\x00", lambda m: f"MATHPH{m.group(1)}XHPHTAM", text)
    svgs = []
    def stash(m):
        svgs.append(m.group(0)); return f"\n\nSVGPLACEHOLDER{len(svgs)-1}ENDPLACEHOLDER\n\n"
    text = re.sub(r"<svg.*?</svg>", stash, text, flags=re.S)
    out = markdown.markdown(text, extensions=["tables", "fenced_code", "toc", "sane_lists", "smarty"])
    for i, s in enumerate(svgs):
        out = out.replace(f"<p>SVGPLACEHOLDER{i}ENDPLACEHOLDER</p>", s).replace(f"SVGPLACEHOLDER{i}ENDPLACEHOLDER", s)
    out = re.sub(r"<(?:em|strong)>(MATHPH(\d+)XHPHTAM)</(?:em|strong)>", r"\1", out)
    out = re.sub(r"<p>(MATHPH(\d+)XHPHTAM)</p>", r"\1", out)
    out = re.sub(r"MATHPH(\d+)XHPHTAM", lambda m: (math[int(m.group(1))] if int(m.group(1)) < len(math) else ""), out)
    return out

CSS = open(os.path.join(BOOK, "assets", "site.css"), encoding="utf-8").read()

def page(title, body, idx):
    prev_l, next_l = "", ""
    if idx > 0:
        prev_l = f'<a class="nav-prev" href="{os.path.splitext(MANIFEST[idx-1][0])[0]}.html">← 上一页</a>'
    if idx < len(MANIFEST) - 1:
        next_l = f'<a class="nav-next" href="{os.path.splitext(MANIFEST[idx+1][0])[0]}.html">下一页 →</a>'
    nav = "".join(f'<a href="{os.path.splitext(f)[0]}.html"{" class=active" if i==idx else ""}>{html.escape(t)}</a>' for i, (f, t) in enumerate(MANIFEST))
    return f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{html.escape(title)} · 大模型：从 Transformer 到部署（试读版）</title>
<link rel="stylesheet" href="katex/katex.min.css"><style>{CSS}</style></head><body>
<aside id="sidebar"><div class="brand"><a href="index.html">大模型<br><span>从 Transformer 到部署 · 免费试读</span></a></div>
<nav id="toc">{nav}<div class="part">完整版</div><a href="{REPO}">GitHub 仓库 / 获取完整版</a></nav></aside>
<main><div class="topbar"><button onclick="document.getElementById('sidebar').classList.toggle('open')">☰ 目录</button></div>
<article class="content"><div class="trial-banner">📚 当前为免费试读版 · 完整版（第 5–17 章 + 附录）获取方式见 <a href="{REPO}" target="_blank">GitHub 仓库</a></div>
{body}<div class="pagenav">{prev_l}{next_l}</div></article></main></body></html>"""

if __name__ == "__main__":
    shutil.rmtree(OUT, ignore_errors=True); os.makedirs(OUT)
    # local katex assets
    shutil.copytree(os.path.join(os.path.dirname(ROOT), "llm-book", "_site", "katex"), os.path.join(OUT, "katex"))  # katex 静态资源沿用主站
    for i, (f, title) in enumerate(MANIFEST):
        body = md2html(open(os.path.join(RENDERED, f), encoding="utf-8").read(), f)
        open(os.path.join(OUT, os.path.splitext(f)[0] + ".html"), "w", encoding="utf-8").write(page(title, body, i))
    # index: hero landing
    cover = open(os.path.join(BOOK, "assets", "cover.svg"), encoding="utf-8").read()
    if HAS_QR:
        shutil.copy(QR, os.path.join(OUT, QR_NAME))
    toc = "".join('<a class="toc-item" href="{}.html"><span class="toc-no">{:02d}</span>{}</a>'.format(
        os.path.splitext(f)[0], i + 1, html.escape(t)) for i, (f, t) in enumerate(MANIFEST))
    qr_html = ('<img class="qr" src="%s" alt="作者微信二维码">' % QR_NAME) if HAS_QR else ""
    LANDING_CSS = """
.hero{background:linear-gradient(160deg,#0f1f3d 0%,#1a3a6e 55%,#2455a4 100%);color:#fff;
  padding:4rem 1.5rem 3.2rem;text-align:center;position:relative;overflow:hidden}
.hero::after{content:"";position:absolute;inset:0;background:
  radial-gradient(600px 300px at 20% 10%,rgba(90,140,255,.18),transparent),
  radial-gradient(500px 260px at 85% 80%,rgba(212,175,55,.10),transparent);pointer-events:none}
.hero-inner{max-width:860px;margin:0 auto;position:relative;z-index:1}
.hero-badge{display:inline-block;background:rgba(255,255,255,.12);border:1px solid rgba(255,255,255,.25);
  border-radius:999px;padding:.35em 1.1em;font-size:.82em;letter-spacing:.06em;margin-bottom:1.2em}
.hero h1{font-size:2.6em;margin:.2em 0 .3em;letter-spacing:.02em}
.hero .sub{color:#c9d6ee;font-size:1.08em;max-width:640px;margin:0 auto 1.6em}
.stats{display:flex;justify-content:center;gap:2.6em;flex-wrap:wrap;margin:1.8em 0}
.stat b{display:block;font-size:1.7em;color:#ffd98a}
.stat span{font-size:.82em;color:#9fb2d6}
.cta-row .btn{background:#ffd98a;color:#1a2440!important;border:none;padding:.8em 2em;font-size:1.05em;box-shadow:0 6px 20px rgba(0,0,0,.25)}
.cta-row .btn.ghost{background:transparent;color:#e8eefb!important;border:1px solid rgba(255,255,255,.45);box-shadow:none}
.wrap{max-width:1060px;margin:0 auto;padding:0 1.5rem}
.sec{padding:3rem 0}
.sec h2{font-size:1.55em;text-align:center;margin-bottom:.4em}
.sec .lead{text-align:center;color:var(--muted);max-width:640px;margin:0 auto 2em}
.feat-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:1.2em}
.feat{background:#fff;border:1px solid #e3e9f2;border-radius:14px;padding:1.5em 1.4em;
  box-shadow:0 1px 4px rgba(20,35,70,.05);transition:transform .15s,box-shadow .15s}
.feat:hover{transform:translateY(-3px);box-shadow:0 10px 26px rgba(20,35,70,.10)}
.feat .ico{font-size:1.6em}
.feat h3{font-size:1.06em;margin:.45em 0 .35em}
.feat p{color:#5b6b7c;font-size:.92em;margin:0}
.gray{background:#f6f8fc}
.toc-grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:.8em}
.toc-item{display:flex;align-items:center;gap:.8em;background:#fff;border:1px solid #e3e9f2;
  border-radius:12px;padding:.9em 1.1em;font-weight:600;color:#2a3a55;text-decoration:none;transition:border-color .15s}
.toc-item:hover{border-color:#3b6fd4}
.toc-no{color:#fff;background:#3b6fd4;border-radius:8px;font-size:.78em;padding:.25em .55em;flex:none}
.price-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:1.4em;max-width:820px;margin:0 auto}
.price{background:#fff;border:1px solid #e3e9f2;border-radius:16px;padding:1.8em;text-align:left;position:relative}
.price.hot{border:2px solid #3b6fd4;box-shadow:0 12px 32px rgba(59,111,212,.16)}
.price .tag{position:absolute;top:-1em;left:1.4em;background:#d43b3b;color:#fff;font-size:.75em;
  padding:.3em .9em;border-radius:999px;letter-spacing:.05em}
.price h3{margin:.1em 0 .2em;font-size:1.1em}
.price .amt{font-size:2.2em;font-weight:800;color:#1a2440;margin:.35em 0 .1em}
.price .amt small{font-size:.45em;color:#8b98a8;font-weight:500}
.price ul{list-style:none;padding:0;margin:1em 0 0;color:#48586c;font-size:.93em}
.price li{padding:.32em 0;border-bottom:1px dashed #eef2f8}
.price li:last-child{border:none}
.price li::before{content:"✓";color:#2e9e5b;font-weight:700;margin-right:.6em}
.qr-card{display:flex;gap:1.6em;align-items:center;background:#fff;border:1px solid #e3e9f2;
  border-radius:16px;padding:1.6em;max-width:680px;margin:2.5em auto 0;text-align:left;box-shadow:0 2px 10px rgba(20,35,70,.06)}
.qr{width:150px;height:auto;border-radius:10px;border:1px solid #e4e8ee;flex:none}
.qr-card .t{font-weight:800;color:#1a56db;margin-bottom:.35em;font-size:1.15em}
.qr-card .d{color:#4a5a6c;font-size:.95em;line-height:1.7}
footer{background:#0f1f3d;color:#8fa2c8;text-align:center;padding:2.2rem 1.5rem;font-size:.88em}
footer a{color:#c9d6ee}
@media (max-width:600px){.hero h1{font-size:1.9em}.stats{gap:1.4em}.qr-card{flex-direction:column;text-align:center}}
"""
    idx = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>大模型：从 Transformer 到部署 | LLM 中文系统教材 · 免费在线试读</title>
<meta name="description" content="《大模型：从 Transformer 到部署》309 页中文 LLM 系统教材免费在线试读：数学基础、Transformer 逐模块实现、预训练、分布式训练、LoRA 微调、RLHF/DPO/GRPO、量化部署、视觉语言模型，92 个可运行代码练习。">
<meta name="keywords" content="大模型,LLM,Transformer,注意力机制,RoPE,预训练,分布式训练,LoRA,RLHF,DPO,GRPO,量化,vLLM,部署,视觉语言模型,中文教材,免费试读">
<meta property="og:type" content="book">
<meta property="og:title" content="大模型：从 Transformer 到部署（免费试读）">
<meta property="og:description" content="从注意力机制到 RLHF、量化部署与视觉语言模型的 309 页中文系统教材。">
<meta property="og:url" content="https://jiejin93.github.io/llm-textbook/">
<meta name="twitter:card" content="summary">
<link rel="canonical" href="https://jiejin93.github.io/llm-textbook/"><link rel="stylesheet" href="katex/katex.min.css"><style>{CSS}{LANDING_CSS}</style></head><body class="landing">
<header class="hero"><div class="hero-inner">
  <div class="hero-badge">309 页 · 17 章 + 数学基础章 + 3 附录 · 92 个可运行代码块</div>
  <h1>大模型：从 Transformer 到部署</h1>
  <p class="sub">一本面向工程师的中文系统教材 —— 从注意力公式的每一项，到预训练、RLHF 对齐、量化部署与视觉语言模型的完整工程链路。</p>
  <div class="cta-row"><a class="btn" href="math.html">📖 开始免费试读</a> <a class="btn ghost" href="{REPO}" target="_blank">GitHub 仓库</a></div>
  <div class="stats">
    <div class="stat"><b>309</b><span>页 A4 排版</span></div>
    <div class="stat"><b>17+1</b><span>章 + 数学基础</span></div>
    <div class="stat"><b>92</b><span>个可运行代码块</span></div>
    <div class="stat"><b>6</b><span>大部分 · 基础到部署</span></div>
  </div>
</div></header>

<section class="sec"><div class="wrap">
  <h2>这本书带你走完的链路</h2>
  <p class="lead">从"懂一点深度学习"起步，到独立完成大模型的设计、训练、微调、评测与上线。</p>
  <div class="feat-grid">
    <div class="feat"><div class="ico">📐</div><h3>数学不绕路</h3><p>第 0 章只讲四个工具（矩阵乘法 / 条件概率 / 链式法则 / 熵），每个概念标注"→ 后文哪里用到"，配浮点溢出、梯度震荡等真实实验。</p></div>
    <div class="feat"><div class="ico">🔍</div><h3>Transformer 逐项拆解</h3><p>缩放点积手推、√d_k 方差推导、RoPE 完整证明、因果掩码与位置编码的必然性 —— 以及 150 行可运行的最小 GPT。</p></div>
    <div class="feat"><div class="ico">⚙️</div><h3>预训练工程主线</h3><p>数据工程 → 训练稳定性 → Scaling Laws → 分布式（ZeRO / TP / PP）→ MoE，含 TinyStories 从零预训练实战。</p></div>
    <div class="feat"><div class="ico">🎯</div><h3>后训练全流程</h3><p>SFT / LoRA / QLoRA 微调、系统评测、RL 基础到 RLHF / DPO / GRPO / RLVR，DPO 损失三步完整推导。</p></div>
    <div class="feat"><div class="ico">🚀</div><h3>推理与部署</h3><p>KV cache 显存实算、FlashAttention、vLLM / PagedAttention、GPTQ / AWQ / FP8 量化、服务化压测。</p></div>
    <div class="feat"><div class="ico">👁</div><h3>多模态与前沿</h3><p>ViT / CLIP / LLaVA / Qwen-VL 架构剖析，含手算例与微调流程。</p></div>
  </div>
</div></section>

<section class="sec gray"><div class="wrap">
  <h2>免费试读 · 前言 + 第 0-4 章</h2>
  <p class="lead">无需注册，点击即读。数学地基与全书前四章，在线完整呈现。</p>
  <div class="toc-grid2">{toc}</div>
</div></section>

<section class="sec"><div class="wrap">
  <h2>获取完整版</h2>
  <p class="lead">完整版含第 5-17 章与全部附录（预训练 / 分布式 / MoE / 微调 / 评测 / RLHF-DPO-GRPO / 量化部署 / 多模态）。</p>
  <div class="price-grid">
    <div class="price">
      <h3>📘 教材完整版</h3>
      <div class="amt">¥59.9</div>
      <ul>
        <li>309 页 A4 排版 PDF 全书</li>
        <li>17 章 + 数学基础章 + 3 附录</li>
        <li>1 年内免费更新 2 次</li>
        <li>勘误与答疑</li>
      </ul>
    </div>
    <div class="price hot">
      <div class="tag">推荐 · 最超值</div>
      <h3>📦 教材 + 运行代码</h3>
      <div class="amt">¥99.9</div>
      <ul>
        <li>含左侧教材完整版全部内容</li>
        <li>92 个代码块独立仓库（按章组织）</li>
        <li>一键 conda 环境 + 章级运行器</li>
        <li>附 5 个开源实战项目指南</li>
        <li>重训练块参数缩放，低配显卡可跑</li>
      </ul>
    </div>
  </div>
  <div class="qr-card">{qr_html}
    <div><div class="t">扫码添加作者微信购买</div>
    <div class="d">备注请写 <b>LLM 教材</b>，会尽快回复。<br>
    也可到 <a href="{REPO}" target="_blank">GitHub 仓库</a> 查看试读章节源文件、完整目录与勘误。<br>
    姊妹篇《掩模版光学仿真与 die-to-database 缺陷检测》见 <a href="https://jiejin93.github.io/optic-textbook/" target="_blank">optic 教材站</a>。</div></div>
  </div>
</div></section>

<footer>《大模型：从 Transformer 到部署》 · 析微视觉实验室 · <a href="{REPO}" target="_blank">GitHub</a> · 试读内容 CC BY-NC 4.0</footer>
</body></html>"""
    open(os.path.join(OUT, "index.html"), "w", encoding="utf-8").write(idx)
    print("trial site built:", OUT)

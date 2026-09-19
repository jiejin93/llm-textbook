# -*- coding: utf-8 -*-
"""Build the free-trial edition of the textbook site into llm-textbook/docs/ for GitHub Pages."""
import os, re, json, shutil, html
import markdown

ROOT = os.path.dirname(os.path.abspath(__file__))
BOOK = os.path.join(os.path.dirname(ROOT), "llm-book")
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
    shutil.copytree(os.path.join(BOOK, "_site", "katex"), os.path.join(OUT, "katex"))
    for i, (f, title) in enumerate(MANIFEST):
        body = md2html(open(os.path.join(RENDERED, f), encoding="utf-8").read(), f)
        open(os.path.join(OUT, os.path.splitext(f)[0] + ".html"), "w", encoding="utf-8").write(page(title, body, i))
    # index: cover + intro
    cover = open(os.path.join(BOOK, "assets", "cover.svg"), encoding="utf-8").read()
    if HAS_QR:
        shutil.copy(QR, os.path.join(OUT, QR_NAME))
    qr_html = ("""<div class="pay-card">
  <img class="pay-qr" src="{qr}" alt="作者微信二维码">
  <div class="pay-info">
    <div class="pay-title">添加作者微信</div>
    <div>完整版（309 页 PDF + 实操代码包）· <b>¥39.9</b>，微信咨询<br>
    备注请写 <b>LLM 教材</b>，会尽快回复</div>
    <div class="pay-note">也可通过 <a href="{repo}" target="_blank">GitHub 仓库</a> 了解试读章节、完整目录与勘误。<br>
    姊妹篇《掩模版光学仿真与 die-to-database 缺陷检测》见 <a href="https://jiejin93.github.io/optic-textbook/" target="_blank"> optic 教材站</a>。</div>
  </div>
</div>""".format(repo=REPO, qr=QR_NAME)) if HAS_QR else ""
    toc = "".join(f'<a href="{os.path.splitext(f)[0]}.html">{html.escape(t)}</a>' for f, t in MANIFEST)
    idx = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>大模型：从 Transformer 到部署 | LLM 中文教程免费在线试读 · 预训练/RLHF/量化部署</title>
<meta name="description" content="《大模型：从 Transformer 到部署》309 页中文 LLM 系统教材免费在线试读：数学基础、Transformer 逐模块实现、预训练、分布式训练、LoRA 微调、RLHF/DPO/GRPO、量化部署、视觉语言模型，92 个可运行代码练习。">
<meta name="keywords" content="大模型,LLM,Transformer,注意力机制,RoPE,预训练,分布式训练,LoRA,RLHF,DPO,GRPO,量化,vLLM,部署,视觉语言模型,中文教材,免费试读">
<meta property="og:type" content="book">
<meta property="og:title" content="大模型：从 Transformer 到部署（免费试读）">
<meta property="og:description" content="从注意力机制到 RLHF、量化部署与视觉语言模型的 309 页中文系统教材。">
<meta property="og:url" content="https://jiejin93.github.io/llm-textbook/">
<meta name="twitter:card" content="summary">
<link rel="canonical" href="https://jiejin93.github.io/llm-textbook/"><link rel="stylesheet" href="katex/katex.min.css"><style>{CSS}
.trial-banner{{background:#fff7ed;border:1px solid #fdba74;border-radius:8px;padding:.7em 1em;margin-bottom:1.5em;font-size:.95em}}
.pay-card{{display:flex;gap:1.5em;align-items:center;background:#fff;border:1px solid #dbe3ee;border-radius:12px;padding:1.5em;max-width:640px;margin:2em auto;text-align:left;box-shadow:0 2px 8px rgba(20,30,50,.06)}}
.pay-qr{{width:170px;height:auto;max-height:240px;border-radius:8px;border:1px solid #e4e8ee;flex:none}}
.pay-title{{font-size:1.25em;font-weight:800;color:#1a56db;margin-bottom:.4em}}
.pay-note{{color:#5b6b76;font-size:.88em;margin-top:.6em}}
@media (max-width:600px){{.pay-card{{flex-direction:column;text-align:center}}}}
</style></head><body class="landing"><main class="landing-main"><div class="cover-wrap">{cover}</div>
<h1>大模型：从 Transformer 到部署</h1>
<p class="subtitle">免费试读版 · 第 0 章数学基础 + 前言 + 第 1–4 章</p>
<p><a class="btn" href="math.html">开始阅读</a> <a class="btn ghost" href="{REPO}" target="_blank">GitHub 仓库</a></p>
{qr_html}
<div class="toc-grid"><h3>试读目录</h3><div class="col">{toc}</div></div>
<p style="color:#5b6b76;margin-top:2em">完整版共 17 章 + 数学基础 + 3 附录（309 页 PDF）：预训练、分布式训练、MoE、微调、评测、RLHF/DPO/GRPO、量化与部署、视觉语言模型等。获取方式见上方付款入口或 GitHub 仓库 README。</p>
</main></body></html>"""
    open(os.path.join(OUT, "index.html"), "w", encoding="utf-8").write(idx)
    print("trial site built:", OUT)

<div align="center">

# 《大模型：从入门到精通》

**理论 · 公式 · 代码 · 生态 —— 一本大而全的中文 LLM 修炼手册**

[在线阅读](https://jiejin93.github.io/llm-textbook/) ·
[术语表](https://jiejin93.github.io/llm-textbook/appendix_a_glossary/) ·
[论文书单](https://jiejin93.github.io/llm-textbook/appendix_b_papers/)

从 BPE 的每一步合并，到注意力公式的每一项，再到一行请求如何被 SGLang 调度、
被 Triton 内核执行、被量化引擎压进 12GB 显存——每个你能叫出名字的东西，
都能在这里回答三件事：**数学定义是什么？解决什么问题？开源实现长什么样？**

</div>

---

## 本书特色

- **公式不跳步**：BPE 手算示例、RoPE 完整推导、在线 softmax、ZeRO 显存公式、LoRA/QLoRA、
  DPO/GRPO 闭式推导、投机解码接受率与加速比——全部保留中间步骤
- **代码可运行**：从零实现优先（150 行最小 GPT、BPE、Triton 内核、RAG、Agent loop），
  实战以 8–12GB 消费级显卡 + Windows/WSL 为基线
- **图文并茂**：60+ 张配图，论文官方原图与自绘 SVG 均注明出处
- **一词一段的术语详解**：每章末尾逐术语讲透"是什么→解决什么问题→工程中哪里碰到"，
  另有 350+ 条术语速查表
- **链接可追溯**：130+ 条论文/博客/仓库链接（arXiv/官方文档），经过人工核验

## 目录

| 部分 | 章节 |
|---|---|
| **A 基础篇** | 大模型全景 · 数学与优化 · 神经网络与反向传播 · 分词器与 BPE · 词嵌入与位置编码 · Transformer 全解 |
| **B 架构篇** | 主流模型架构剖析（LLaMA/Qwen/DeepSeek） · 注意力演进（GQA→MLA→FlashAttention→Mamba） · 混合专家 MoE |
| **C 训练篇** | 预训练与 Scaling Law · 数据工程 · 分布式训练 · SFT 与参数高效微调 · 对齐（RLHF/DPO/GRPO/RLVR） |
| **D 系统篇** | 推理优化 · 推理引擎实战（vLLM/SGLang/llama.cpp） · GPU/Triton 与编译器栈（LLVM/MLIR） · 量化与压缩 |
| **E 应用篇** | Prompt/RAG · Agent 与 MCP · 多模态与扩散模型 · 评测与安全 |
| **附录** | 术语表（350+） · 论文书单（94 篇分级） · 环境与硬件指南 |

## 本地构建

```bash
pip install mkdocs mkdocs-material
cd book
mkdocs serve        # 开发预览（全站内容可见，仅用于写作）
mkdocs build        # 输出到 ../docs
python tools/lock_build.py   # 付费章节加密锁定（必须，见下）
```

**付费墙说明**：`tools/lock_build.py` 会把非试读章节的正文加密进页面
（公开 HTML 只含导读与目录），读者在页面输入解锁码即可全站解锁。
解锁码从环境变量 `LLM_BOOK_CODE` 或本地文件 `book/tools/secret_code.txt`
读取，**该文件已被 .gitignore 排除，严禁提交或公开**。换码 = 改文件后重新
build + lock，旧码自动失效。

## 仓库结构

```
docs/          构建产物（GitHub Pages 从 main 分支 /docs 发布，付费章节已加密）
```

> 书籍源码（book/）存放在私有仓库，不随本公开仓库分发。

## 致谢

本书大量参考并链接了社区公开的论文、博客与开源实现（Karpathy、Lilian Weng、Jay Alammar、
苏剑林、Sebastian Raschka、Hugging Face、Datawhale 等），均以链接形式注明出处，版权归原作者所有。
内容采用 CC BY-NC-SA 4.0，代码示例可自由使用。

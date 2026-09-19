<div align="center">

# 《大模型：从 Transformer 到部署》

**一本面向工程师的中文系统教材：从注意力机制到 RLHF、量化部署与视觉语言模型**

[在线试读](https://jiejin93.github.io/llm-textbook/) · [获取完整版](#获取完整版) · [试读章节源文件](./trial/)

</div>

---

## 这本书讲什么

从"懂深度学习但 Transformer 不深"起步，到能**独立完成** LLM 的设计、预训练、微调、评测、量化、部署，以及视觉语言模型与 RLHF/GRPO 后训练。

**全书 309 页 / 17 章 + 数学基础章 + 3 个附录，包含：**

- 📐 **完整数学基础**（第 0 章）：线代/概率/信息论/矩阵求导/优化/浮点数，面向小白，每个概念标注"→第 N 章哪里用到"
- 🔍 **Transformer 详解**（第 3 章）：缩放点积手推、RoPE 推导、150 行可运行最小 GPT
- ⚙️ **工程主线**：分词器 BPE → 数据工程 → 训练稳定性 → Scaling Laws → 分布式（ZeRO/TP/PP）→ MoE
- 🎯 **后训练**：SFT/LoRA/QLoRA → 评测 → RL 基础 → RLHF/DPO/GRPO/RLVR（DPO 三步完整推导）
- 🚀 **推理与部署**：KV cache 显存实算、FlashAttention、vLLM/PagedAttention、GPTQ/AWQ/FP8 量化、服务化压测
- 👁 **多模态**：ViT/CLIP/LLaVA/Qwen-VL，含 2×2 手算例与微调流程
- 💻 **92 个可运行 Python 代码块 + 每章实操练习**（与 PyTorch 官方实现逐位对账）

> 📚 姊妹篇：[《掩模版光学仿真与 die-to-database 缺陷检测》](https://github.com/jiejin93/optic-textbook)（波动光学 → Abbe/Hopkins 成像仿真 → 缺陷检测，同系列中文教材）

## 免费试读

官网可免费阅读全文前三部分预览（第 0 章数学基础、第 1–4 章）：
👉 **https://jiejin93.github.io/llm-textbook/**

## 目录

<details><summary>点击展开完整目录</summary>

- 第 0 章 面向大模型的数学基础（线性代数·概率·微积分·优化）
- 第一部分 基础：绪论 / 语言模型基础 / Transformer 详解 / 分词器
- 第二部分 预训练：数据工程 / 训练方法与稳定性 / Scaling Laws / 分布式训练 / MoE
- 第三部分 微调与评测：SFT 与参数高效微调 / 模型评测
- 第四部分 后训练：强化学习基础 / RLHF、DPO 与 GRPO
- 第五部分 推理与部署：推理优化与模型量化 / 部署与服务化
- 第六部分 多模态与前沿：视觉语言模型 / 前沿专题与学习路线
- 附录：数学符号表 / 开源仓库清单 / 术语中英对照

</details>

## 获取完整版

| 版本 | 内容 | 价格 |
|---|---|---|
| PDF 完整版 | 309 页全书，1 年内 2 次免费更新 | ¥39（早鸟）/ ¥59 |
| PDF + 代码包 | 附全部实操练习仓库（含环境锁定） | ¥99 |

👉 购买：**https://github.com/jiejin93/llm-textbook#获取完整版**

## 本书如何写成的

本书在写作中使用了 AI 辅助（结构设计、初稿与代码生成），并经过多轮人工审校流程：全部公式逐条重推复核（修正过 9 处初稿错误）、92 个代码块通过语法与运行校验、插图几何与数值一致性检查。

写作参考了国内外公开课程、开源社区资料以及书中各章“延伸阅读”引用的论文与项目。

## License

- 试读章节与代码：CC BY-NC 4.0（署名-非商业）
- 完整版内容：保留所有权利，禁止转载/售卖

## Star History

（上线后由 star-history.com 生成嵌入）

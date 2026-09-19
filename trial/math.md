# 第 0 章 面向大模型的数学基础（线性代数·概率·微积分·优化）

这一章是为"只学过大学高数上册"的读者准备的前置数学补丁。后面 17 章里出现的几乎所有数学——从第 3 章注意力的点积、到第 6 章混合精度的浮点溢出、再到第 13 章 RLHF 里的 KL 约束——都建立在本章的概念之上。我们把它们按"依赖顺序"一次性讲清：先学会看符号（0.1），再学线性代数（0.2）和概率（0.3），它们合起来支撑信息论（0.4）与微积分（0.5），最后汇入优化（0.6）和数值表示（0.7）。

每一节的结构都是固定的：**直觉 → 形式定义 → 后文哪里用到 → 可运行代码 → 延伸阅读**。你不必先读完本章再读第 1 章——它更像一本随查字典，但从头到尾读一遍大约需要 4~6 小时，读完后再看第 3 章、第 12–13 章会顺畅得多。

## 0.1 这一章怎么读：数学符号速查

数学是一种"压缩格式"：符号写起来短，但每个符号背后都藏着一段话。下面这张表把全书高频符号先"解压"一遍，读后面任何章节卡在符号上时，先回来查这张表。

| 符号 | 读法 | 含义 | 例子 |
|---|---|---|---|
| $\sum_{i=1}^{n} a_i$ | 西格玛，i 从 1 到 n 求和 | 把 $a_1+a_2+\cdots+a_n$ 压成一行 | $\sum_{i=1}^{3} i = 1+2+3 = 6$ |
| $\prod_{i=1}^{n} p_i$ | 派，连乘 | 把 $p_1 \times p_2 \times \cdots \times p_n$ 压成一行 | $\prod_{i=1}^{3} i = 6$ |
| $P(A)$ | 事件 A 的概率 | 0 到 1 之间的数 | $P(\text{下雨}) = 0.3$ |
| $P(A \mid B)$ | 给定 B 时 A 的条件概率 | 已知 B 发生后，A 的可能性 | $P(\text{带伞}\mid\text{下雨}) = 0.9$ |
| $\mathbb{E}[X]$ | X 的期望 | 随机变量的"平均值" | 掷骰子 $\mathbb{E}[X]=3.5$ |
| $\arg\max_x f(x)$ | 使 f 取最大值的 x | 注意：返回的是 **x 本身**，不是最大值 | $\arg\max_x (-x^2) = 0$ |
| $\log x$ | 自然对数（底为 e） | $e^{\log x} = x$；本书所有 $\log$ 默认底 $e\approx 2.718$ | $\log e = 1$ |
| $\nabla f$ | f 的梯度（nabla） | 所有偏导数排成的向量 | 见 0.5 节 |
| $\|\mathbf{x}\|$ | 向量 x 的范数（长度） | $\sqrt{\sum_i x_i^2}$ | $\|(3,4)\| = 5$ |
| $\approx$ | 约等于 | 近似相等 | $e \approx 2.718$ |
| $:=$ | 定义为 | 左边定义为右边的值 | $\eta := 0.001$ |

三个最容易被小白忽略的细节：

1. **$\arg\max$ 与 $\max$ 的区别**：$\max_x f(x)$ 是"最高山峰的高度"，$\arg\max_x f(x)$ 是"最高山峰在哪"。第 2 章的 greedy 解码（贪心采样）用的就是 argmax——我们要的是"哪个词"，不是"概率多大"。
2. **对数把乘变加**：$\log(ab) = \log a + \log b$。语言模型一句话的概率是几百个条件概率的连乘，每个都小于 1，连乘会数值下溢（变成 $10^{-300}$ 这种机器几乎表示不了的数）；取对数后变成连加，既安全又好算。这是为什么训练目标几乎总写对数形式。
3. **对数求导**：若 $y = \log x$，则 $\frac{dy}{dx} = \frac{1}{x}$。推导只要把 $x = e^y$ 两边对 $x$ 求导：$1 = e^y \frac{dy}{dx}$，于是 $\frac{dy}{dx} = \frac{1}{e^y} = \frac{1}{x}$。这个公式在 softmax + 交叉熵的反向传播推导（0.5.4 节）里是关键一步。

> 延伸阅读：Khan Academy 的[代数与求和记号复习](https://www.khanacademy.org/math/algebra-home)；维基百科[数学符号表](https://en.wikipedia.org/wiki/List_of_mathematical_symbols)。

## 0.2 线性代数：向量、矩阵与张量

### 0.2.1 向量：一串数字，也是一个点

**直觉**：向量（vector）就是一列有序的数，比如 $\mathbf{x} = (x_1, x_2, \ldots, x_d) \in \mathbb{R}^d$。它有两种等价的看法：一是一支从原点出发的箭头（有方向有长度），二是 $d$ 维空间里的一个点。大模型里，"猫"这个词在模型眼中就是一个几千维的向量——语义被编码成了空间中的位置。

**定义**：$d$ 维实数向量空间 $\mathbb{R}^d$ 中的元素。向量可以相加（对应分量相加）、可以数乘（每个分量乘同一个数）。

### 0.2.2 点积：相似度的度量

两个同维向量 $\mathbf{q}$ 和 $\mathbf{k}$ 的点积（dot product，内积）：

$$
\mathbf{q} \cdot \mathbf{k} = \sum_{i=1}^{d} q_i k_i = \|\mathbf{q}\| \, \|\mathbf{k}\| \cos\theta
$$

其中 $\theta$ 是两向量的夹角。**直觉**：点积衡量"两个向量有多一致"。方向相同 → 点积大且为正；垂直（无关）→ 为 0；相反 → 为负。在词向量空间里，"国王 · 女王" 远大于 "国王 · 洗衣机"。

**→ 后文哪里用到**：这就是第 3 章注意力机制的引擎——注意力分数 $\text{score}(q, k) = \frac{q \cdot k}{\sqrt{d}}$，本质是"查询词 q 与每个词的键 k 做点积，看谁跟我最相关"。第 16 章 CLIP 的图文相似度也是点积。看懂本小节，第 3 章的公式就懂了一半。

### 0.2.3 矩阵乘法及其复杂度

**直觉**：矩阵（matrix）是一个 $m \times n$ 的数表。矩阵乘法 $C = AB$（$A$ 是 $m \times k$，$B$ 是 $k \times n$）可以理解为"对 B 的每一列分别做一次线性组合"，也可以理解为"批量做点积"：$c_{ij} = \sum_{l=1}^{k} a_{il} b_{lj}$，即 $C$ 的第 $i$ 行第 $j$ 列是 $A$ 的第 $i$ 行与 $B$ 的第 $j$ 列的点积。

$$
C_{ij} = \sum_{l=1}^{k} A_{il} B_{lj}, \qquad C \in \mathbb{R}^{m \times n}
$$

注意形状规则：$(m \times k) \times (k \times n) = (m \times n)$，内侧两个维度必须相等——写代码报维度错误时，先检查这个。

**复杂度**：每个 $c_{ij}$ 要做 $k$ 次乘加，共 $m \times n$ 个元素，所以是 $O(m \cdot k \cdot n)$。特例：两个 $n \times n$ 方阵相乘是 $O(n^3)$。**→ 第 3 章**：注意力中 $QK^\top$ 是 $(n \times d) \times (d \times n)$，复杂度 $O(n^2 d)$，$n$ 是序列长度——这个平方就是"长上下文贵"的根源，也是第 14 章 FlashAttention 要优化的对象。

### 0.2.4 转置、范数、单位矩阵与逆、特征值（速览）

- **转置** $A^\top$：行列互换，$A^\top_{ij} = A_{ji}$。性质 $(AB)^\top = B^\top A^\top$（顺序翻转！）。→ 第 3 章 $QK^\top$、第 10 章 LoRA 的 $B A$。
- **范数** $\|\mathbf{x}\|_p = \left(\sum_i |x_i|^p\right)^{1/p}$。最常用 $p=2$（欧氏长度）和 $p=1$（绝对值之和，第 6 章/第 10 章正则化里限制参数大小用它）。归一化 = 除以范数，让长度变成 1。
- **单位矩阵** $I$：对角线为 1 其余为 0，满足 $AI = A$，相当于数的 1。**逆矩阵** $A^{-1}$ 满足 $A^{-1}A = I$，相当于"除法"，概念了解即可——深度学习几乎从不显式求逆（数值不稳定且 $O(n^3)$），总是解方程或用梯度。
- **特征值**（一句话）：$A\mathbf{v} = \lambda \mathbf{v}$，$\lambda$ 是"被这个矩阵拉伸后方向不变只变长度"的倍数；它刻画矩阵的"最大拉伸能力"。→ 第 3 章位置编码、第 10 章初始化分析中会再次露面，届时再展开。

### 0.2.5 PyTorch 张量与广播

张量（tensor）是向量的推广：0 阶=标量，1 阶=向量，2 阶=矩阵，3 阶以上=张量。一批句子的词向量堆在一起就是形状 `(batch, seq_len, d_model)` 的 3 阶张量。**广播**（broadcasting）是 numpy/PyTorch 的对齐规则：形状不同的张量运算时，维度从右往左对齐，长度为 1 的维度自动"复制"到与另一方相同。→ 第 3 章多头注意力的 `mask`、第 8 章分布式里的 per-device 缩放，全靠广播。

```python
import torch

# 向量、点积、矩阵乘法
q = torch.tensor([1.0, 2.0, 3.0])
k = torch.tensor([4.0, 0.0, 1.0])
print(torch.dot(q, k))            # 7.0  = 1*4 + 2*0 + 3*1

# 注意力的雏形：4 个 token，每个 3 维
Q = torch.randn(4, 3)             # 形状 (n=4, d=3)
K = torch.randn(4, 3)
scores = Q @ K.transpose(-2, -1)  # (4,3) @ (3,4) -> (4,4)，token 两两点积
print(scores.shape)               # torch.Size([4, 4])

# 范数与归一化（第 3 章注意力里 Softmax 前的缩放、第 13 章都有出现）
print(torch.norm(q))              # sqrt(1+4+9) = 3.7417

# 广播：(4,3) + (3,) -> (3,) 被复制 4 份
print((Q + q).shape)              # torch.Size([4, 3])
```

> 延伸阅读：3Blue1Brown [Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra)（强烈推荐前三集：向量、线性组合、矩阵乘法的几何意义）；维基百科 [Matrix multiplication](https://en.wikipedia.org/wiki/Matrix_multiplication)；[PyTorch 广播语义](https://pytorch.org/docs/stable/notes/broadcasting.html)。

## 0.3 概率论：不确定性的语言

### 0.3.1 随机变量与概率分布

**直觉**：随机变量（random variable）是"还没揭晓的数"——掷骰子的点数、模型下一个要生成的词。概率分布（probability distribution）给每个可能的结果分配一个 0 到 1 的权重，所有权重加起来等于 1。语言模型本质上就是一台机器：输入前文，输出"下一个词的概率分布"。

**定义**：离散随机变量 $X$ 取值 $x$ 的概率记 $P(X = x)$（简写 $p(x)$），满足 $p(x) \ge 0$ 且 $\sum_x p(x) = 1$。

### 0.3.2 联合、条件概率与贝叶斯

- **联合概率** $P(A, B)$：两件事同时发生的概率。
- **条件概率**：$P(A \mid B) = \dfrac{P(A, B)}{P(B)}$，即"已知 B 发生，A 的概率"。读作"在 B 条件下 A"。
- 由此得**乘法法则** $P(A, B) = P(B)\,P(A \mid B)$。
- **贝叶斯公式**：

$$
P(A \mid B) = \frac{P(B \mid A)\, P(A)}{P(B)}
$$

**直觉**：贝叶斯是"事后修正"的公式——先有信念 $P(A)$（先验），看到证据 B，按"该证据在 A 下有多常见 $P(B \mid A)$"修正成 $P(A \mid B)$（后验）。**→ 第 4 章** Unigram 分词器的迭代就依赖它；更广泛地，"模型参数给定数据"的机器学习视角本身就是贝叶斯式的。

### 0.3.3 期望与方差、独立

- **期望**（expected value）：加权平均，$\mathbb{E}[X] = \sum_x x\, p(x)$。线性性：$\mathbb{E}[aX + bY] = a\,\mathbb{E}[X] + b\,\mathbb{E}[Y]$（无需独立！）。**→ 第 12 章**策略梯度、第 13 章 GRPO 的"组相对优势"全是对期望求导/估计。
- **方差**（variance）：$\mathrm{Var}(X) = \mathbb{E}\big[(X - \mathbb{E}[X])^2\big]$，衡量波动大小，方差小 = 估计稳。→ 第 12 章讲"为什么要用基线降低方差"。
- **独立**（independent）：$P(A, B) = P(A)P(B)$，互不影响。注意语言模型里相邻词**不**独立——这正是语言建模有意思的地方。

### 0.3.4 链式法则：语言模型的灵魂公式

把 0.3.2 的乘法法则反复使用，任意联合概率可以拆成一串条件概率的连乘：

$$
P(w_1, w_2, \ldots, w_n) = \prod_{i=1}^{n} P(w_i \mid w_1, \ldots, w_{i-1}) = \prod_{i=1}^{n} P(w_i \mid w_{<i})
$$

推导（对 $n=3$ 验证）：$P(w_1, w_2, w_3) = P(w_1)\,P(w_2, w_3 \mid w_1) = P(w_1)\,P(w_2 \mid w_1)\,P(w_3 \mid w_1, w_2)$。归纳即得一般形式。

**直觉**：一句话出现的概率 = "第 1 个词的概率 × 已知第 1 个词后第 2 个词的概率 × ……"。**→ 第 2 章**整章建立在这条公式上：语言模型训练就是最大化它（的对数），生成就是沿着它自回归地往前走；第 14 章的 KV Cache、投机解码也是在加速这个逐词过程。

### 0.3.5 采样

**直觉**：从分布里"抽签"。greedy 是永远抽最大那个（会啰嗦重复），temperature 是先把 logits 除以 $T$ 再 softmax（$T$ 小 → 分布尖，$T$ 大 → 分布平），top-k/top-p 是先砍掉小概率再抽。**→ 第 2 章采样策略一节详细展开，那里会逐个实现**。

```python
import torch

# 一个"下一个词"的分布：模拟 P(w | "今天天气真")
logits = torch.tensor([2.0, 1.0, 0.5, -1.0])
probs = torch.softmax(logits, dim=-1)
print(probs)                       # tensor([0.61, 0.22, 0.14, 0.03]) 附近

# 链式法则演示：P(今天, 天气) = P(今天) * P(天气|今天)
p_today = torch.tensor(0.01)
p_weather_given_today = torch.tensor(0.05)
print((p_today * p_weather_given_today).item())   # 0.0005，连乘多了就会下溢 -> 这就是为什么要取 log

# 从分布中采样（第 2 章 temperature 采样的雏形）
samples = torch.multinomial(probs, num_samples=1000, replacement=True)
print(torch.bincount(samples) / 1000.0)           # 频率近似概率，越接近 probs 越好
```

> 延伸阅读：3Blue1Brown [概率论系列](https://www.3blue1brown.com/topics/probability)；Khan Academy [Probability & Statistics](https://www.khanacademy.org/math/statistics-probability)；维基百科 [Bayes' theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem)。

## 0.4 信息论：熵、交叉熵、KL 散度

信息论回答一个问题："一条消息里到底含有多少信息？"

### 0.4.1 自信息与熵

**直觉**：越意外的事，信息量越大。太阳从东边升起（概率≈1）信息量≈0；今天下雪（小概率）信息量很大。把"意外程度"量化：

$$
\text{自信息}(x) = -\log p(x)
$$

概率 1 → 自信息 0；概率减半 → 自信息加 1（$\log 2 \approx 0.693$，以 2 为底时恰好是 1 比特）。取对数还保证了独立事件信息量可加。

**熵**（entropy）是自信息的期望，即"平均意外程度"：

$$
H(P) = \mathbb{E}_{x \sim P}[-\log p(x)] = -\sum_x p(x) \log p(x)
$$

分布越"确定"（一个概率接近 1）熵越低；分布越均匀熵越高。均匀分布 over $V$ 个词的熵是 $\log V$（取 2 为底即 $\log_2 V$ 比特）。**→ 第 7 章**讨论模型能力上限、第 2 章困惑度的定义，都以熵为出发点。

### 0.4.2 交叉熵：用错误的分布给真实数据编码

**定义**：真实分布 $P$，我们的模型给出分布 $Q$，交叉熵为

$$
H(P, Q) = -\sum_x p(x) \log q(x)
$$

**直觉**：数据其实来自 $P$，但我们用 $Q$ 当"编码本"去编码它。如果 $Q$ 把概率给了错误的地方（$p$ 大的地方 $q$ 小），$-\log q$ 就很大，交叉熵就高。**最重要的恒等式**（展开即得）：

$$
H(P, Q) = H(P) + D_{\mathrm{KL}}(P \,\|\, Q) \ge H(P)
$$

交叉熵 = 数据本身的熵（不可约的部分）+ 我们犯的错（KL 散度）。训练语言模型时 $H(P)$ 是常数，所以**最小化交叉熵 = 最小化 KL = 让模型分布贴近真实分布**。**→ 第 2 章**训练目标 `loss = -log P(下一个词 | 前文)` 正是交叉熵在"每步只有一个正确词"情形下的特例（此时 $H(P,Q)$ 退化为 $-\log q(w^*)$）；第 13 章奖励模型排名损失也是其变体。

### 0.4.3 KL 散度：两个分布差多远，且不对称

$$
D_{\mathrm{KL}}(P \,\|\, Q) = \sum_x p(x) \log \frac{p(x)}{q(x)} \ge 0
$$

两个关键性质：

1. **非负**（吉布斯不等式），且当且仅当 $P = Q$ 处处相等时取 0。直觉：平均对数似然比 $\mathbb{E}_P[\log(p/q)]$，$p>q$ 的地方被 $p$ 加权放大。
2. **不对称**：$D_{\mathrm{KL}}(P\|Q) \ne D_{\mathrm{KL}}(Q\|P)$！直觉：$D_{\mathrm{KL}}(P\|Q)$（前向 KL）惩罚"真实分布有概率、模型却给零概率"的地方——于是 $Q$ 被迫覆盖 $P$ 的每个角落（mass-covering）；$D_{\mathrm{KL}}(Q\|P)$（反向 KL）只惩罚"模型给了概率而真实分布没有"的地方——于是 $Q$ 倾向收缩到 $P$ 的某个峰上（mode-seeking）。

**→ 第 13 章** RLHF 的 PPO 目标里有一项 $\beta\, D_{\mathrm{KL}}(\pi_\theta \| \pi_{\text{ref}})$：允许策略向高奖励移动，但每个 token 都不能偏离参考模型太远，否则奖励被劫持。第 9 章 MoE 的负载均衡损失也是"让专家分布接近均匀"的 KL 形式。第 16 章 CLIP/InfoNCE 里也有它的身影。

### 0.4.4 困惑度

困惑度（perplexity, PPL）是交叉熵的指数：

$$
\mathrm{PPL} = 2^{H} \quad (\text{以 2 为底}) \qquad \text{或} \qquad \mathrm{PPL} = e^{L} \; \text{，其中 } L = -\frac{1}{n}\sum_{i=1}^{n} \log p(w_i \mid w_{<i})
$$

**直觉**：困惑度 = "模型每一步在多少个词之间犹豫"。PPL = 1 是完美预测；PPL = 100 意味着平均而言像在 100 个词里均匀猜。因为指数是单调的，**最小化交叉熵损失 ⇔ 最小化困惑度**。→ 第 2 章正式定义、第 11 章评测、第 7 章 Scaling Laws 的 $\hat L$ 全用它。

```python
import torch
import torch.nn.functional as F

# 手写交叉熵：模拟 3 个样本、词表 5，正确词 id 分别是 [0, 2, 4]
logits = torch.randn(3, 5)
target = torch.tensor([0, 2, 4])

log_probs = F.log_softmax(logits, dim=-1)          # log q(x)，数值稳定的做法（见 0.7 节）
mine = -log_probs[torch.arange(3), target].mean()  # H = -mean log q(正确词)
ref = F.cross_entropy(logits, target)              # PyTorch 官方实现
print(mine.item(), ref.item())                     # 两者应完全一致

# KL 散度的不对称性
p = torch.tensor([0.9, 0.1])
q = torch.tensor([0.5, 0.5])
kl_pq = (p * (p / q).log()).sum()   # D_KL(P||Q)
kl_qp = (q * (q / p).log()).sum()   # D_KL(Q||P)
print(kl_pq.item(), kl_qp.item())   # 0.37 vs 0.51 —— 顺序不同结果不同！
```

> 延伸阅读：Colah 的博客 [Visual Information Theory](https://colah.github.io/posts/2015-09-Visual-Information/)（交叉熵/KL 的可视化，极佳）；维基百科 [Cross entropy](https://en.wikipedia.org/wiki/Cross_entropy)、[Kullback–Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)。

## 0.5 微积分与矩阵求导

### 0.5.1 导数与链式法则

导数是"输入动一点点，输出动多少"：

$$
f'(x) = \frac{df}{dx} = \lim_{\Delta x \to 0} \frac{f(x + \Delta x) - f(x)}{\Delta x}
$$

常用导数：$(x^n)' = nx^{n-1}$，$(e^x)' = e^x$，$(\log x)' = 1/x$（0.1 节已推导）。

**链式法则**：复合函数 $f(g(x))$ 的导数是两段"变化率"相乘：

$$
\frac{d}{dx} f(g(x)) = f'(g(x)) \cdot g'(x)
$$

**直觉**：齿轮组。第一级齿轮转速乘第二级的传动比，得到末端转速。**→ 第 2 章**反向传播就是链式法则在计算图上的自动化：损失对第 100 层参数的导数 = 沿图反推的 100 次连乘。

### 0.5.2 梯度、雅可比

多元函数 $f: \mathbb{R}^d \to \mathbb{R}$（输入向量、输出一个数——损失函数正是这种形状），**梯度**（gradient）是把所有偏导排成向量：

$$
\nabla f(\mathbf{x}) = \left( \frac{\partial f}{\partial x_1}, \ldots, \frac{\partial f}{\partial x_d} \right)
$$

**直觉**：站在山坡上，梯度指向"上坡最陡的方向"，模长是那个方向的坡度。训练就是沿 $-\nabla f$ 下坡。

若输出也是向量 $f: \mathbb{R}^n \to \mathbb{R}^m$，所有偏导排成的 $m \times n$ 矩阵叫**雅可比矩阵**（Jacobian），$J_{ij} = \partial f_i / \partial x_j$。直觉：它就是向量版的导数，"每个输入分量如何影响每个输出分量"。反向传播每过一层，就把上游梯度乘上该层的雅可比。了解直觉即可，本书不需要手算大雅可比。

### 0.5.3 softmax 函数

把任意实数向量 $\mathbf{z} \in \mathbb{R}^V$ 变成合法概率分布：

$$
\text{softmax}_i(\mathbf{z}) = \frac{e^{z_i}}{\sum_{j=1}^{V} e^{z_j}}
$$

指数保证非负，除以和保证总和为 1。它还是"可微版的 argmax"：输入差距越大，输出越接近 one-hot。**→ 第 2 章**输出层、**第 3 章**注意力权重（对打分做 softmax 得到"注意力有多集中"）、第 12 章策略分布，全是它。

### 0.5.4 softmax 的雅可比：完整推导（全书反向传播的核心砖块）

目标：求 $\dfrac{\partial\, \text{softmax}_i}{\partial z_j}$。记 $s_i = \dfrac{e^{z_i}}{S}$，$S = \sum_k e^{z_k}$。分两种情况。

**情形 1：$j = i$**。用商法则 $\left(\frac{u}{v}\right)' = \frac{u'v - uv'}{v^2}$，其中 $u = e^{z_i}$，$v = S$：

$$
\frac{\partial s_i}{\partial z_i} = \frac{e^{z_i} S - e^{z_i} e^{z_i}}{S^2} = \frac{e^{z_i}}{S}\cdot\frac{S - e^{z_i}}{S} = s_i (1 - s_i)
$$

**情形 2：$j \neq i$**。此时 $e^{z_i}$ 对 $z_j$ 求导为 0，只有分母 $S$ 里有 $e^{z_j}$，$\frac{\partial S}{\partial z_j} = e^{z_j}$：

$$
\frac{\partial s_i}{\partial z_j} = \frac{0 \cdot S - e^{z_i} e^{z_j}}{S^2} = -\,s_i s_j \qquad (j \neq i)
$$

合并写成一行（全书反复出现，建议背下来）：

$$
\frac{\partial\, \text{softmax}_i}{\partial z_j} = s_i\,(\delta_{ij} - s_j), \qquad \delta_{ij} = \begin{cases}1 & j = i\\ 0 & j \neq i\end{cases}
$$

**推论（交叉熵 + softmax 的黄金组合）**：设损失 $L = -\log s_t$（$t$ 是正确类别），链式法则：

$$
\frac{\partial L}{\partial z_j} = \sum_i \frac{\partial L}{\partial s_i}\cdot\frac{\partial s_i}{\partial z_j} = -\frac{1}{s_t}\cdot\frac{\partial s_t}{\partial z_j} = -\frac{1}{s_t}\, s_t(\delta_{tj} - s_j) = s_j - \delta_{tj}
$$

即 **梯度 = 预测概率 − one-hot 标签**。预测对了梯度为 0，错得越离谱梯度越大——优雅得不像话。**→ 第 2 章**语言模型输出层的反向传播、**第 3 章**注意力权重回传、第 12 章策略梯度，都以这条结论为砖块。

### 0.5.5 数值梯度验证

不确定自己推的公式对不对？用定义 $\frac{\partial f}{\partial x} \approx \frac{f(x+h) - f(x-h)}{2h}$（中心差分，误差 $O(h^2)$）直接数值验证。这是深度学习工程师的看家本领。

```python
import torch

# 验证 softmax 雅可比公式 d s_i / d z_j = s_i (delta_ij - s_j)
z = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
s = torch.softmax(z, dim=-1)

J = torch.zeros(3, 3)
for i in range(3):                       # 对 s 的每个分量反向求梯度
    g = torch.zeros(3); g[i] = 1.0
    J[i] = torch.autograd.grad(s, z, grad_outputs=g, retain_graph=True)[0]  # J[i,j] = d s_i / d z_j

manual = torch.diag(s) - torch.outer(s, s)   # 公式: diag(s) - s s^T
print((J - manual).abs().max().item())   # ~1e-9，公式正确

# 验证“交叉熵 + softmax 的梯度 = 预测 - onehot”
logits = torch.randn(5, requires_grad=True)
target = torch.tensor(2)
loss = torch.nn.functional.cross_entropy(logits, target)
loss.backward()
print(logits.grad)                                        # 手写梯度:
print(torch.softmax(logits, -1) - torch.nn.functional.one_hot(target, 5).float())
```

> 延伸阅读：3Blue1Brown [微积分的本质](https://www.3blue1brown.com/topics/calculus)；Khan Academy [Derivatives](https://www.khanacademy.org/math/differential-calculus)；维基百科 [Softmax function](https://en.wikipedia.org/wiki/Softmax_function)、[Jacobian](https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant)。

## 0.6 优化：怎么找到下山的最短路

### 0.6.1 损失函数与梯度下降推导

损失函数（loss function）把"模型有多差"量化成一个数，训练 = 最小化它。梯度下降（gradient descent）的推导只需泰勒展开的一阶直觉：在当前点 $\mathbf{x}_t$ 附近，

$$
f(\mathbf{x}_t + \Delta\mathbf{x}) \approx f(\mathbf{x}_t) + \nabla f(\mathbf{x}_t)^\top \Delta\mathbf{x}
$$

想让 $f$ 下降最多，就要让 $\nabla f^\top \Delta\mathbf{x}$ 最负。取步子方向为负梯度、长度受学习率 $\eta$ 限制，即得更新规则：

$$
\mathbf{x}_{t+1} = \mathbf{x}_t - \eta\, \nabla f(\mathbf{x}_t)
$$

一步能降多少？$f(\mathbf{x}_{t+1}) \approx f(\mathbf{x}_t) - \eta \|\nabla f\|^2$，只要梯度不为零、$\eta$ 不太大，就一定在降。

### 0.6.2 学习率、局部最优与鞍点

- **学习率 $\eta$** 是步长。太大 → 在谷底两侧来回弹甚至发散（loss 尖叫着上升）；太小 → 爬得像蜗牛。**→ 第 6 章**会讲 warmup + cosine 调度：先热身再退火。
- **局部最优**（local minimum）：周围一个小邻域内最低的点，但不是全局最低。好消息：高维损失面里真正的"坑"反而不多，更多的是——
- **鞍点**（saddle point）：在一个方向上是谷、另一个方向上是脊，形似马鞍。梯度为 0 但不是最优点。大模型损失面上鞍点远多于局部最优，这是训练的主要障碍之一。直觉：100 亿参数的空间里，要求 100 亿个方向"恰好都朝上"才能困住你，太难了；总有一些方向是下坡。
- **凸 vs 非凸**：凸函数（碗形，任意局部最优即全局最优，如线性回归的损失）；深度网络损失是高度非凸的，没有这种保证，所以训练更像"在山脉中找足够低的谷"，理论和实践都围绕"逃出鞍点、稳定下降"展开。

### 0.6.3 动量与自适应学习率（预告）

- **动量**（momentum）：把更新想象成有惯性的小球，$\mathbf{v}_{t+1} = \mu \mathbf{v}_t - \eta \nabla f$，$\mathbf{x}_{t+1} = \mathbf{x}_t + \mathbf{v}_{t+1}$。直觉：方向一致时越滚越快（穿过鞍点附近的小平台），方向反复抵消时自动减速（震荡的峡谷里更稳）。
- **为什么需要自适应学习率**：不同参数的梯度量级可以差几个数量级（有的梯度常年 $10^{-8}$，有的 $10^{-2}$），统一学习率对前者太小、对后者太大。**自适应方法**（AdaGrad/RMSProp/Adam）为每个参数维护自己的步长——用梯度平方的滑动平均来缩放。**→ 第 6 章**完整推导 Adam/AdamW，此处只埋种子；本章你只需记住：Adam ≈ 动量 + 逐参数自适应步长。

```python
import numpy as np
import torch

# 玩具例子：f(x, y) = (x-3)^2 + 10(y+2)^2，一个椭圆谷（见下方等高线图）
def f(x, y):  return (x - 3)**2 + 10 * (y + 2)**2
def grad(x, y): return np.array([2*(x - 3), 20*(y + 2)])

x = np.array([0.0, 0.0])
for step in range(100):                 # 普通梯度下降
    x = x - 0.04 * grad(*x)
print(x)                                # 收敛到 [3, -2] 附近

# 用 torch 自动求导验证我们手写的梯度没算错
xt = torch.tensor([0.5, -0.5], requires_grad=True)
f(*xt).backward()
print(xt.grad.numpy(), grad(0.5, -0.5)) # 两行应当一致
```

下面这张图直观展示了梯度下降在椭圆谷中的"锯齿形"轨迹——这正是动量要治的病：

<svg viewBox="0 0 420 320" xmlns="http://www.w3.org/2000/svg" font-family="sans-serif">
  <rect width="420" height="320" fill="#fafafa"/>
  <text x="210" y="24" text-anchor="middle" font-size="14" fill="#333">梯度下降在椭圆等高线上的锯齿轨迹</text>
  <g stroke="#9db8d9" fill="none">
    <ellipse cx="260" cy="180" rx="140" ry="60"/>
    <ellipse cx="260" cy="180" rx="110" ry="46"/>
    <ellipse cx="260" cy="180" rx="80" ry="32"/>
    <ellipse cx="260" cy="180" rx="50" ry="19"/>
    <ellipse cx="260" cy="180" rx="22" ry="8"/>
  </g>
  <circle cx="260" cy="180" r="4" fill="#2e7d32"/>
  <text x="268" y="176" font-size="11" fill="#2e7d32">最小值 (3, -2)</text>
  <path d="M 90 60 L 172 92 L 128 118 L 196 140 L 172 158 L 216 168 L 208 182 L 238 184 L 240 192 L 254 190 L 256 182"
        fill="none" stroke="#c62828" stroke-width="2"/>
  <circle cx="90" cy="60" r="4" fill="#c62828"/>
  <text x="60" y="50" font-size="11" fill="#c62828">起点</text>
  <text x="70" y="120" font-size="11" fill="#c62828">梯度方向垂直于等高线，</text>
  <text x="70" y="135" font-size="11" fill="#c62828">在窄谷里来回震荡</text>
  <text x="70" y="290" font-size="11" fill="#666">f(x,y) = (x-3)² + 10(y+2)²，y 方向更陡 → 锯齿</text>
</svg>

> 延伸阅读：3Blue1Brown [多元微积分中"梯度"一集](https://www.3blue1brown.com/topics/multivariable-calculus)；[Distill — Why Momentum Really Works](https://distill.pub/2017/momentum/)（动量的最佳可视化讲解）；维基百科 [Stochastic gradient descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)。

## 0.7 数值表示：浮点数与溢出

### 0.7.1 IEEE 754：计算机如何存小数

计算机用有限位二进制存小数，标准是 IEEE 754。一个浮点数分三段：符号位 $s$、指数位、尾数位（mantissa/fraction），值为

$$
\text{value} = (-1)^s \times 1.m \times 2^{e}
$$

**直觉**：科学计数法的二进制版。$1.m$ 像有效数字（尾数位越多，精度越高），指数决定小数点位置（指数位越多，能表示的范围越大）。精度与范围是一对此消彼长的资源——这是 FP16/BF16 取舍的全部秘密。

<svg viewBox="0 0 460 170" xmlns="http://www.w3.org/2000/svg" font-family="monospace">
  <rect width="460" height="170" fill="#fffdf5"/>
  <text x="230" y="22" text-anchor="middle" font-size="14" font-family="sans-serif" fill="#333">三种浮点格式的位分配（共 16/16/32 位）</text>
  <g font-size="12" font-family="sans-serif">
    <text x="10" y="55">FP16</text>
    <rect x="50" y="40" width="16" height="20" fill="#e57373"/>
    <rect x="66" y="40" width="80" height="20" fill="#ffb74d"/>
    <rect x="146" y="40" width="224" height="20" fill="#81c784"/>
    <text x="52" y="76" fill="#c62828">s(1)</text><text x="88" y="76" fill="#ef6c00">exp(5)</text><text x="230" y="76" fill="#2e7d32">mantissa(10)</text>

    <text x="10" y="115">BF16</text>
    <rect x="50" y="100" width="16" height="20" fill="#e57373"/>
    <rect x="66" y="100" width="128" height="20" fill="#ffb74d"/>
    <rect x="194" y="100" width="176" height="20" fill="#81c784"/>
    <text x="52" y="136" fill="#c62828">s(1)</text><text x="112" y="136" fill="#ef6c00">exp(8)</text><text x="262" y="136" fill="#2e7d32">mantissa(7)</text>

    <text x="10" y="160">FP32</text>
    <rect x="50" y="145" width="16" height="16" fill="#e57373"/>
    <rect x="66" y="145" width="40" height="16" fill="#ffb74d"/>
    <rect x="106" y="145" width="264" height="16" fill="#81c784"/>
    <text x="52" y="140" fill="#c62828"></text>
  </g>
  <text x="380" y="55" font-size="10" fill="#666" font-family="sans-serif">同样 16 位：</text>
  <text x="380" y="70" font-size="10" fill="#666" font-family="sans-serif">BF16 牺牲精度</text>
  <text x="380" y="85" font-size="10" fill="#666" font-family="sans-serif">换取大范围</text>
</svg>

三种常见格式的对比（FP16 最小正规数按 $2^{-14}$ 计）：

| 格式 | 符号 | 指数 | 尾数 | 最大值 | 最小正规数 | 有效十进制位 |
|---|---|---|---|---|---|---|
| FP32 | 1 | 8 | 23 | $\approx 3.4\times10^{38}$ | $\approx 1.2\times10^{-38}$ | ~7 |
| FP16 | 1 | 5 | 10 | 65504 | $\approx 6.1\times10^{-5}$ | ~3 |
| BF16 | 1 | 8 | 7 | $\approx 3.4\times10^{38}$ | $\approx 1.2\times10^{-38}$ | ~2 |

**→ 第 6 章**混合精度训练：FP16 范围窄（超过 65504 就溢出成 `inf`，梯度又常常小于 $6\times10^{-5}$ 而下溢成 0），所以要 loss scaling；BF16 与 FP32 共享指数位、范围一样宽，不易溢出但精度低。**→ 第 14 章** FP8 量化、第 10 章 QLoRA 的 NF4，都是"如何用更少的信息存住权重"这一主题的延续。

### 0.7.2 上溢、下溢与 softmax 的 max 减技巧

**为什么 exp 容易溢出**：$e^{z}$ 是爆炸函数。$e^{710} \approx 10^{308}$ 就到了 FP64 的极限；FP16 下 $e^{12} \approx 162754 > 65504$，直接 `inf`。而 softmax 分子分母都是一堆 $e^z$，训练早期 logits 轻轻松松上几十，裸写必炸。

**技巧**：softmax 对平移不变——分子分母同乘 $e^{-c}$ 不变：

$$
\text{softmax}_i(\mathbf{z}) = \frac{e^{z_i - c}}{\sum_j e^{z_j - c}}, \qquad c = \max_j z_j \;\Rightarrow\; z_j - c \le 0 \;\Rightarrow\; e^{z_j - c} \le 1
$$

减去最大值后所有指数都不超过 0，$e$ 的结果都在 $(0, 1]$，永不上溢；分母至少有一项为 1，也不会因为全下溢而除零。这就是 0.4 节代码里 `F.log_softmax` 背后干的事。

```python
import torch

z = torch.tensor([500.0, 501.0, 502.0])

# 裸写 softmax：溢出
naive = torch.exp(z) / torch.exp(z).sum()
print(naive)   # tensor([nan, nan, nan]) —— exp(500) = inf，inf/inf = nan

# max 减技巧：稳定
c = z.max()
stable = torch.exp(z - c) / torch.exp(z - c).sum()
print(stable)                              # 正常的概率分布
print(torch.allclose(stable, torch.softmax(z, dim=-1)))  # True，与官方一致

# FP16 的窄范围亲测：65504 之上即 inf
a = torch.tensor([65504.0], dtype=torch.float16)
print((a + 1).dtype, (a * 2))   # float16 tensor([inf], ...)
```

交叉熵与 KL 散度的关系可以再用一张图锚定在直觉上：用 $Q$ 编码 $P$ 的"多付的代价"就是 KL：

<svg viewBox="0 0 420 200" xmlns="http://www.w3.org/2000/svg" font-family="sans-serif">
  <rect width="420" height="200" fill="#fafcff"/>
  <text x="210" y="22" text-anchor="middle" font-size="14" fill="#333">H(P,Q) = H(P) + D_KL(P‖Q)</text>
  <rect x="60"  y="50" width="150" height="110" rx="55" fill="#bbdefb" fill-opacity="0.8"/>
  <rect x="170" y="50" width="180" height="110" rx="55" fill="#c8e6c9" fill-opacity="0.8"/>
  <text x="112" y="60" text-anchor="middle" font-size="12" fill="#1565c0">真实分布 P（数据）</text>
  <text x="262" y="60" text-anchor="middle" font-size="12" fill="#2e7d32">模型分布 Q</text>
  <text x="112" y="112" text-anchor="middle" font-size="13" fill="#0d47a1">H(P)</text>
  <text x="112" y="130" text-anchor="middle" font-size="11" fill="#0d47a1">数据本身的熵</text>
  <text x="245" y="105" text-anchor="middle" font-size="13" fill="#1b5e20">D_KL(P‖Q)</text>
  <text x="245" y="123" text-anchor="middle" font-size="11" fill="#1b5e20">多付的编码代价</text>
  <text x="245" y="141" text-anchor="middle" font-size="11" fill="#1b5e20">（不对称！转正比）</text>
  <text x="210" y="185" text-anchor="middle" font-size="11" fill="#666">训练只能压缩 KL 这一块——H(P) 是数据的固有噪声，降不掉</text>
</svg>

> 延伸阅读：维基百科 [IEEE 754](https://en.wikipedia.org/wiki/IEEE_754)；[Float Exposed——浮点数交互式可视化](https://float.exposed/0x3fc00000)；[PyTorch 混合精度教程](https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html)。

## 0.8 本章小结

- **符号**：$\sum/\prod$ 是压缩写法；$\arg\max$ 返回位置不返回值；$\log$ 把连乘变连加，$\frac{d}{dx}\log x = \frac{1}{x}$。
- **线性代数**：点积 = 相似度（→ 注意力）；矩阵乘 $(m{\times}k)(k{\times}n)$ 复杂度 $O(mkn)$（→ 注意力 $O(n^2 d)$）；张量广播无处不在。
- **概率**：语言模型的核心是链式法则 $P(w_1..w_n)=\prod P(w_i \mid w_{<i})$；交叉熵最小化 = 让模型分布贴近数据分布。
- **信息论**：$H(P,Q) = H(P) + D_{\mathrm{KL}}(P\|Q)$；KL 非负且**不对称**（→ RLHF 的 KL 约束）；困惑度 $= e^{\text{交叉熵}}$。
- **微积分**：梯度指上坡最陡方向；$\frac{\partial\,\text{softmax}_i}{\partial z_j} = s_i(\delta_{ij} - s_j)$；交叉熵+softmax 的梯度 = 预测 − 标签；数值梯度可随时验证推导。
- **优化**：梯度下降来自一阶泰勒；大模型的主要障碍是鞍点而非局部最优；动量抗震荡，自适应学习率应对参数间梯度量级差异（→ 第 6 章 Adam）。
- **数值**：浮点 = $(-1)^s \times 1.m \times 2^e$，精度与范围此消彼长；FP16 窄、BF16 宽而糙；softmax 永远先减 max。

## 0.9 练习

1. （手算）设 $\mathbf{q}=(1,0,2)$，$\mathbf{k}=(2,1,0)$。计算点积、两个向量的范数，并用 $\|\mathbf{q}\|\|\mathbf{k}\|\cos\theta$ 验证。再计算 $\text{softmax}((1,2,4))$（保留 3 位小数），并验证减 max 后结果不变。
2. （概念）证明 $D_{\mathrm{KL}}(P\|Q) \ge 0$（提示：利用 $\log x \le x - 1$）。再各举一个"前向 KL 覆盖全部、反向 KL 错失模式"的直观例子，解释为什么 RLHF 中对 KL 约束方向的选择很重要（→ 第 13 章）。
3. （代码实操）手写交叉熵并与 torch 对照：随机生成 `logits = torch.randn(8, 100)` 与 `target = torch.randint(0, 100, (8,))`，先用 `log_softmax` 与 `nll_loss`（或直接索引）实现，再与 `F.cross_entropy` 比较，要求最大误差小于 $10^{-6}$。然后手动实现 `softmax` 的反向传播（利用 $s_i(\delta_{ij}-s_j)$），用 0.5.5 节的数值梯度法验证。
4. （代码实操）复现并观察混合精度：用 `torch.float16` 计算 `torch.tensor([1e3]) * torch.tensor([100])` 与 `torch.tensor([1e-5]) / torch.tensor([1e3])`，观察溢出与下溢；再写一个 `safe_softmax(z)`（先减 max），用 `z = torch.tensor([800.0, 800.0, 801.0])` 分别测试裸版与安全版，用 `torch.allclose` 与 `torch.softmax`（FP64 下）对照。
5. （思考）第 7 章 Scaling Laws 中损失形如 $\hat{L} = E + A/N^\alpha + B/D^\beta$。结合 0.4 节回答：$E$ 为什么可以解释为"数据的熵"？训练无法把损失降到 $E$ 以下这件事，与 0.4.2 节的哪个恒等式对应？

## 0.10 延伸阅读

- 3Blue1Brown 系列（视频，直觉建立最快）：[线性代数的本质](https://www.3blue1brown.com/topics/linear-algebra)、[微积分的本质](https://www.3blue1brown.com/topics/calculus)、[多元微积分](https://www.3blue1brown.com/topics/multivariable-calculus)、[神经网络](https://www.3blue1brown.com/topics/neural-networks)
- Khan Academy：[概率与统计](https://www.khanacademy.org/math/statistics-probability)、[微分学](https://www.khanacademy.org/math/differential-calculus)
- Distill（交互式图文期刊）：[Why Momentum Really Works](https://distill.pub/2017/momentum/)
- Colah's Blog：[Visual Information Theory](https://colah.github.io/posts/2015-09-Visual-Information/)、[Calculus on Computational Graphs: Backpropagation](https://colah.github.io/posts/2015-08-Backprop/)
- 参考书：Christopher Bishop, *Pattern Recognition and Machine Learning*（附录有概率与矩阵求导的紧凑复习）；Ian Goodfellow 等，[*Deep Learning*](https://www.deeplearningbook.org/)（第 2–4 章即本章对应内容的进阶版）
- 工具文档：[PyTorch 广播语义](https://pytorch.org/docs/stable/notes/broadcasting.html)、[NumPy 广播](https://numpy.org/doc/stable/user/basics.broadcasting.html)、[Float Exposed](https://float.exposed/0x3fc00000)（浮点数位级交互式查看）
- 维基百科条目：[Kullback–Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)、[Cross entropy](https://en.wikipedia.org/wiki/Cross_entropy)、[IEEE 754](https://en.wikipedia.org/wiki/IEEE_754)、[Softmax function](https://en.wikipedia.org/wiki/Softmax_function)

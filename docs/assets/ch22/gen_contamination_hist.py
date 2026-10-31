# -*- coding: utf-8 -*-
"""图②：数据污染的"训练-测试重合度"分布——污染组 vs 干净组的真实模拟。

构造 300 道测试题：其中 100 道被"爬进训练语料"（语料里存有 3% 字符被随机替换的
近拷贝），200 道保持干净（语料里只有同主题的不同文本）。对每道题计算它与语料
最相似文档的 8-gram Jaccard 重合度，画两组直方图。脚本同目录可复现：
    python gen_contamination_hist.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(7)

# ---- 1) 用真实题目模板构造 300 道测试题（槽位随机替换生成变体） ----------------
TEMPLATES = [
    "一年中有{m}个月至少有{d}天",
    "海拔每升高{h}米气温约下降{t}摄氏度",
    "光在真空中的速度约为每秒{k}公里",
    "标准马拉松的全程距离是{d2}公里",
    "人体正常体温平均约为{c}摄氏度",
    "一个八度音程包含{n}个半音",
    "地球绕太阳公转一圈约需{days}天",
    "TCP建立连接需要进行{hs}次握手",
    "IPv4地址由{b}位二进制数组成",
    "《红楼梦》全书共约{chs}回",
    "黄金的化学元素符号是{sym}，原子序数为{num}",
    "声音在15摄氏度空气中的速度约{v}米每秒",
]
FILL = {
    "m": ["三", "四", "五", "七", "十一"], "d": ["28", "30", "31"],
    "h": ["100", "500", "1000"], "t": ["0.6", "1", "3"],
    "k": ["30", "299792", "300000"], "d2": ["40", "42.195", "45"],
    "c": ["36.5", "37", "38"], "n": ["7", "12", "8"],
    "days": ["360", "365", "366"], "hs": ["2", "3", "4"],
    "b": ["16", "32", "64"], "chs": ["100", "120", "80"],
    "sym": ["Au", "Ag", "Fe"], "num": ["26", "47", "79"],
    "v": ["340", "510", "1500"],
}
def make_question(rng):
    t = rng.integers(len(TEMPLATES))
    q = TEMPLATES[t]
    for key, opts in FILL.items():
        q = q.replace("{" + key + "}", rng.choice(opts))
    # 真实基准题在题干之外还有选项/解析/材料等长文本，这里拼 4 句随机排列的
    # 材料句，让每道题的 8-gram 分布基本唯一（模拟真实题目的长度与多样性）。
    frags = rng.choice(EXPL, size=4, replace=False)
    return q + "".join(f for f in frags)

EXPL = [
    "出题组在题干后附了一段约两百字的背景材料。",
    "阅卷标准要求写出两行以上的推导过程。",
    "这道题在历年正式考试中出现过三次改写版本。",
    "解析首先排除了两个量纲不符的干扰选项。",
    "作答时需要先把文字信息转成符号表达。",
    "题图给出了一个未经标注的坐标系示意图。",
    "命审专家小组对表述做过两轮迭代打磨。",
    "该知识点属于考纲第二层级的核心要求。",
    "材料中的数据均来自公开发布的统计年鉴。",
    "评分细则对四舍五入的位数做了明确规定。",
    "干扰项的设计利用了最常见的直觉错误。",
    "题目右下角标注了满分与建议作答时长。",
    "命题人自称这道题的区分度指标达到 0.4 以上。",
    "审题时圈出限定词是拿满分的第一步。",
    "真题复盘课上老师用三种方法分别解了一遍。",
    "易错点在于混淆了题目中两个相近的常数。",
    "题干中的条件 (2) 是后续小问的铺垫。",
    "官方解析建议用排除法先砍掉明显错误项。",
    "这道题的作答区预留了六行横线。",
    "类似的表述也出现在另一省份的模拟卷里。",
    "标准答案的最后一行给出了单位换算。",
    "不少考生因看漏了括号里的补充条件而失分。",
    "解析末尾附了一个可以直接套用的经验公式。",
    "监考记录显示多数考生在十分钟内完成此题。",
]

# ---- 2) 训练语料：12 段手写的同主题知识段落 + 污染题的近拷贝 ------------------
TOPIC_PARAS = [
    "历法小知识：公历的月份长短不一，三十一天的月份有一月、三月、五月、七月、八月、十月和十二月，其余大多为三十天，二月最特殊。",
    "大气科学：气温随海拔的变化遵循大气垂直递减率，对流层内高度越高空气越稀薄，保温作用越弱，因此高山山顶往往比山脚凉爽许多。",
    "物理常数：真空光速是物理学最重要的基本常数之一，迈克耳孙等人用旋转棱镜法做过精密测量，如今它的数值已被定义为精确值。",
    "田径运动：马拉松比赛对赛道距离有严格规定，赛事组织者需要经过专业测量认证，参赛者的完赛成绩才被官方承认。",
    "生理卫生：人的体温存在昼夜波动，清晨略低、午后略高，测量部位不同读数也不同，判断发热要看腋下或口腔的参照范围。",
    "音乐理论：将一个八度按等比数例均分可以得到十二个音，钢琴的白键与黑键合起来正好对应这套体系，是现代律制的基石。",
    "天文常识：地球公转的轨道是椭圆，近日点在一月初，远日点在七月初，四季变化主要源自地轴倾斜而非日地距离差异。",
    "计算机网络：两台主机通信前需要先建立连接，客户端与服务器交互若干轮确认彼此的收发能力之后，才开始正式传输数据。",
    "网络基础：互联网协议第四版用点分十进制书写，分成网络位与主机位两部分，子网掩码用来划定两者的边界。",
    "中国古典文学：清代白话章回小说的巅峰之作以贾府的兴衰为主线，前八十回与后四十回的作者问题至今仍有争论。",
    "化学元素：这种金黄色金属化学性质极不活泼，古人已能用沙金淘洗获得，它的导电性与延展性使它在电子工业中不可替代。",
    "声学：声音的传播依赖介质，空气温度影响分子运动的快慢，冬天里声音传播得比夏天略慢一些。",
]
N_TEST = 300
N_CONTAM = 100
questions = [make_question(rng) for _ in range(N_TEST)]
contam_idx = rng.choice(N_TEST, size=N_CONTAM, replace=False)

def mutate(text, rate, rng):
    """字符级近拷贝：以 rate 概率把字符替换成随机汉字（模拟抓取+清洗的损伤）。"""
    pool = "的了是在有和人不我们中上为"
    return "".join(c if rng.random() > rate else rng.choice(list(pool))
                   for c in text)

# 语料 = 同主题知识段落（干净背景）+ 被爬进语料的 100 道题的近拷贝（污染源）
corpus = list(TOPIC_PARAS) + [mutate(questions[i], 0.03, rng) for i in contam_idx]

# ---- 3) 指标：每道测试题与语料最相似文档的 8-gram Jaccard 重合度 --------------
def ngrams(s, n=8):
    return {s[i:i + n] for i in range(len(s) - n + 1)} if len(s) >= n else {s}

corpus_grams = [ngrams(c) for c in corpus]
scores = np.zeros(N_TEST)
for i, q in enumerate(questions):
    gq = ngrams(q)
    best = 0.0
    for gc in corpus_grams:
        inter = len(gq & gc)
        if inter:
            j = inter / len(gq | gc)
            best = max(best, j)
    scores[i] = best

is_contam = np.zeros(N_TEST, dtype=bool)
is_contam[contam_idx] = True
TH = 0.5
tp = int(((scores > TH) & is_contam).sum())
fn = int(((scores <= TH) & is_contam).sum())
fp = int(((scores > TH) & ~is_contam).sum())

# ---- 4) 画图 ----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9.2, 5.0), dpi=150)
bins = np.linspace(0, 1, 41)
ax.hist(scores[is_contam], bins=bins, alpha=0.65, color="#c44e52",
        label=f"污染组（{N_CONTAM} 题，语料存近拷贝）")
ax.hist(scores[~is_contam], bins=bins, alpha=0.65, color="#4c72b0",
        label=f"干净组（{N_TEST - N_CONTAM} 题，语料仅同主题文本）")
ax.axvline(TH, color="black", ls="--", lw=1.6)
ax.text(TH + 0.015, ax.get_ylim()[1] * 0.92, f"判污染阈值 {TH}",
        fontsize=10)
ax.set_xlabel("训练-测试重合度（与语料最相似文档的 8-gram Jaccard）")
ax.set_ylabel("题数")
ax.set_title("数据污染检测：污染组重合度接近 1，干净组接近 0\n"
             "（真实模拟：300 题、字符级 3% 变异近拷贝，由本脚本真实计算）")
ax.text(0.68, ax.get_ylim()[1] * 0.62,
        f"阈值 {TH} 下：\n召回 {tp}/{tp + fn}，误报 {fp} 题",
        fontsize=10, bbox=dict(fc="#fffbe6", ec="gray"))
ax.legend(fontsize=10)
fig.tight_layout()
fig.savefig(__file__.replace("gen_contamination_hist.py", "contamination_hist.png"),
            dpi=150, bbox_inches="tight")
print(f"污染组重合度中位数 = {np.median(scores[is_contam]):.3f}")
print(f"干净组重合度中位数 = {np.median(scores[~is_contam]):.3f}")
print(f"阈值 {TH}: 检出 {tp}/{tp+fn}，误报 {fp}")
print("已保存 contamination_hist.png")

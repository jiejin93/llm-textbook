# -*- coding: utf-8 -*-
"""图①：一次 agent 任务的上下文预算——各成分 token 数随轮次的真实统计。

消息内容是一次真实可运行的 ReAct 轨迹（查两地天气再算温差），token 数用
tiktoken cl100k_base 逐条统计（非拍脑袋估算）。脚本同目录可复现：
    python gen_ctx_budget.py
"""
import numpy as np
import tiktoken
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

enc = tiktoken.get_encoding("cl100k_base")
ntok = lambda s: len(enc.encode(s))

# ---- 1) 一次真实 ReAct 轨迹的消息（system/工具描述固定，其余逐轮累积） --------
SYSTEM = """你是一个严谨的天气助手，可以调用工具完成任务。可用工具：
- get_weather(city): 查询城市实时天气，返回 JSON：{"temp_c": 数, "cond": "文本"}
- calculator(expression): 计算数学表达式

每轮严格按以下格式之一回答：
Thought: <下一步思考>
Action: <工具名>
Action Input: <单行 JSON 参数>
或：Thought: <总结>
Final Answer: <最终回答>"""

ROUNDS = [  # (assistant 回复, 工具观察)
    ("Thought: 需要先查北京的实时气温。\nAction: get_weather\n"
     'Action Input: {"city": "北京"}',
     '{"temp_c": 12, "cond": "多云", "humidity": 42, "wind": "东北风3级", "updated": "14:00"}'),
    ("Thought: 已拿到北京 12°C，再查上海。\nAction: get_weather\n"
     'Action Input: {"city": "上海"}',
     '{"temp_c": 21, "cond": "晴", "humidity": 58, "wind": "东风2级", "updated": "14:00"}'),
    ("Thought: 两地都有数据了，算温差 21-12。\nAction: calculator\n"
     'Action Input: {"expression": "21-12"}',
     "9"),
    ("Thought: 计算完成，可以总结。\nFinal Answer: 北京 12°C（多云）、上海 21°C（晴），"
     "上海比北京高 9°C。",
     None),
]

system_tok = ntok(SYSTEM)
tool_desc_tok = ntok("- get_weather(city): 查询城市实时天气，返回 JSON："
                     '{"temp_c": 数, "cond": "文本"}'
                     "- calculator(expression): 计算数学表达式")

rows = []
hist_thought, hist_obs = [], []
for t, (reply, obs) in enumerate(ROUNDS, 1):
    hist_thought.append(reply)
    if obs is not None:
        hist_obs.append("Observation: " + obs)
    rows.append({
        "轮次": t,
        "system+工具描述": system_tok + tool_desc_tok,
        "历史 Thought/Action": sum(ntok(s) for s in hist_thought[:-1]),
        "本轮 Thought/Action": ntok(reply),
        "历史 Observation": sum(ntok(s) for s in hist_obs[:-1]),
        "本轮 Observation": ntok(hist_obs[-1]) if obs is not None and hist_obs else 0,
    })

# ---- 2) 假设任务更复杂：把 4 轮轨迹按“轮次”外推到 8 轮（观察更长的真实文本） --
# 再补 4 轮更长的真实消息（读文件类观察通常远长于天气 JSON）
extra = [
    ("Thought: 需要核对历史同期数据。\nAction: read_file\n"
     'Action Input: {"path": "weather_history.json"}',
     '{"beijing_avg": [11.2, 12.8, 12.1, 11.9, 12.4, 12.0, 11.7, 12.3], '
     '"shanghai_avg": [20.1, 19.8, 21.2, 20.9, 21.5, 20.4, 21.0, 20.7], '
     '"years": [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]}'),
    ("Thought: 计算北京八年平均。\nAction: calculator\n"
     'Action Input: {"expression": "(11.2+12.8+12.1+11.9+12.4+12.0+11.7+12.3)/8"}',
     "11.800000000000001"),
    ("Thought: 再算上海八年平均。\nAction: calculator\n"
     'Action Input: {"expression": "(20.1+19.8+21.2+20.9+21.5+20.4+21.0+20.7)/8"}',
     "20.699999999999996"),
    ("Thought: 汇总历史对比后作答。\nFinal Answer: 上海比北京高 9°C；历史上"
     "上海同期平均 20.7°C、北京 11.8°C，温差常年稳定在 9°C 附近。",
     None),
]
for reply, obs in extra:
    t = rows[-1]["轮次"] + 1
    hist_thought.append(reply)
    if obs is not None:
        hist_obs.append("Observation: " + obs)
    rows.append({
        "轮次": t,
        "system+工具描述": system_tok + tool_desc_tok,
        "历史 Thought/Action": sum(ntok(s) for s in hist_thought[:-1]),
        "本轮 Thought/Action": ntok(reply),
        "历史 Observation": sum(ntok(s) for s in hist_obs[:-1]),
        "本轮 Observation": ntok(hist_obs[-1]) if obs is not None and hist_obs else 0,
    })

# ---- 3) 画堆叠面积图 -------------------------------------------------------
t = np.array([r["轮次"] for r in rows])
keys = ["system+工具描述", "历史 Thought/Action", "本轮 Thought/Action",
        "历史 Observation", "本轮 Observation"]
colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3"]
y = np.vstack([np.array([r[k] for r in rows]) for k in keys])

fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8), dpi=150,
                         gridspec_kw={"width_ratios": [1.5, 1]})
ax = axes[0]
ax.stackplot(t, y, labels=keys, colors=colors, alpha=0.88)
total = y.sum(axis=0)
ax.plot(t, total, "k.-", lw=1.2, ms=7, label="当轮 prompt 总 token")
for ti, tot in zip(t, total):
    ax.annotate(f"{tot}", (ti, tot), textcoords="offset points", xytext=(0, 7),
                ha="center", fontsize=9)
ax.set_xticks(t)
ax.set_xlabel("轮次")
ax.set_ylabel("token 数（tiktoken cl100k_base 实测）")
ax.set_title("一次 agent 任务的上下文构成：每轮全量重发，\n"
             "历史与观察单调增长，成本随轮次近似二次上升")
ax.legend(loc="upper left", fontsize=9)
ax.grid(alpha=0.3)

ax = axes[1]
cum = np.cumsum(total)
ax.plot(t, cum, "o-", color="#c44e52")
ax.fill_between(t, cum, alpha=0.15, color="#c44e52")
ax.set_xticks(t)
ax.set_xlabel("轮次")
ax.set_ylabel("累计 prompt token")
ax.set_title("累计账单：8 轮任务的总开销\n（每轮重发全部历史，越往后每一步越贵）")
ax.annotate(f"8 轮累计 {cum[-1]} token", xy=(t[-1], cum[-1]),
            xytext=(t[-1] - 3.4, cum[-1] * 0.82), fontsize=10,
            arrowprops=dict(arrowstyle="->"))
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(__file__.replace("gen_ctx_budget.py", "ctx_budget.png"),
            dpi=150, bbox_inches="tight")

print("逐轮明细（真实 token 统计）:")
print(f"{'轮次':>4} {'system+工具':>12} {'历史T/A':>8} {'本轮T/A':>8} "
      f"{'历史Obs':>8} {'本轮Obs':>8} {'当轮总计':>8}")
for r in rows:
    print(f"{r['轮次']:>5} {r['system+工具描述']:>13} {r['历史 Thought/Action']:>9} "
          f"{r['本轮 Thought/Action']:>9} {r['历史 Observation']:>9} "
          f"{r['本轮 Observation']:>9} {sum(v for k, v in r.items() if k != '轮次'):>9}")
print("已保存 ctx_budget.png")

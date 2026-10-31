# -*- coding: utf-8 -*-
"""图②：ReAct 循环状态机——主循环与两条错误恢复路径。

纯 matplotlib 画的框图（FancyBboxPatch + FancyArrowPatch），与 20.2 的
最小实现逐状态对应。脚本同目录可复现：python gen_react_fsm.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

BLUE, RED, GREEN, GRAY, ORANGE = "#4c72b0", "#c44e52", "#2e7d32", "#8c8c8c", "#b8860b"

fig, ax = plt.subplots(figsize=(11.6, 6.4), dpi=150)
ax.set_xlim(0, 120)
ax.set_ylim(0, 64)
ax.axis("off")

def box(x, y, w, h, text, fc="#eaf1fb", ec=BLUE, fs=10.5, lw=1.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6",
                                fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

def arrow(x0, y0, x1, y1, text="", color=GRAY, fs=9.5, rad=0.0,
          toff=(0, 1.6), ls="-", ha="center"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=14, color=color, lw=1.7,
                                 connectionstyle=f"arc3,rad={rad}", linestyle=ls))
    if text:
        ax.text((x0 + x1) / 2 + toff[0], (y0 + y1) / 2 + toff[1], text,
                fontsize=fs, color=color, ha=ha, va="center")

# ---- 主循环 ---------------------------------------------------------------
box(6, 47, 24, 10, "① 接收任务\nmessages = [system, user]", fc="#f5f5f5", ec=GRAY)
box(44, 47, 26, 10, "② LLM 生成回复\nThought + Action 或\nFinal Answer")
box(84, 47, 22, 10, "③ 回复含\nFinal Answer？", fc="#fff7e6", ec=ORANGE)
box(84, 26, 22, 10, "④ 按格式解析\nAction / Action Input", fc="#fff7e6", ec=ORANGE)
box(44, 26, 26, 10, "⑤ 执行工具\n得到 Observation")
box(6, 26, 26, 10, "⑥ 观察拼回上下文\nmessages += [assistant,\nuser(Observation)]",
    fc="#eef7ee", ec=GREEN, fs=9.5)
box(84, 6, 22, 9, "输出最终答案\n循环结束", fc="#e6f4e6", ec=GREEN, lw=2.2)

arrow(30, 52, 44, 52)
arrow(70, 52, 84, 52)
arrow(95, 47, 95, 36, "否", color=GREEN, toff=(3.2, 0))
arrow(84, 31, 70, 31)
# ⑤ -> ⑥ -> ② 的回环（折线，末端箭头）
ax.plot([57, 57, 24, 24], [26, 20.5, 20.5, 24.5], color=GRAY, lw=1.7)
ax.add_patch(FancyArrowPatch((24, 24.5), (24, 26), arrowstyle="-|>",
                             mutation_scale=14, color=GRAY, lw=1.7))
ax.plot([19, 19, 50], [36, 40.5, 40.5], color=GRAY, lw=1.7)
arrow(50, 40.5, 50, 47, "观察回传后进入下一轮", color=GRAY, toff=(4, 0.2), fs=9)
# ③ --是--> 终态（从右外侧绕行）
arrow(106, 51, 97, 15, "是", color=GREEN, rad=-0.42, toff=(7.5, 6))

# ---- 错误恢复路径（红色） ---------------------------------------------------
box(44, 6, 26, 9, "解析失败 → 带格式示例\n重新提示（重试预算 ≤3 次）",
    fc="#fdecec", ec=RED, fs=9.5)
arrow(89, 26, 70, 11, "解析出错", color=RED, rad=0.12, toff=(1.5, 2.5), ha="left")
ax.text(52.5, 16.8, "超过预算 →\n终止并上报失败", fontsize=8.5, color=RED, ha="left")

box(6, 6, 24, 9, "工具抛异常 → 错误文本\n作为 Observation 回传自纠",
    fc="#fdecec", ec=RED, fs=9.5)
arrow(50, 26, 24, 15, "执行异常", color=RED, rad=0.12, toff=(-2.0, 2.2), ha="right")
arrow(24, 15, 24, 26, "（错误也是观察，同样拼回上下文）", color=RED,
      toff=(14.5, 0), fs=8.5)

ax.set_title("ReAct 循环状态机：思考 → 行动 → 观察 → 回传，直到 Final Answer\n"
             "（绿箭头 = 正常推进，红箭头/红框 = 错误恢复路径；蓝框为模型决策点）",
             fontsize=12.5)
fig.tight_layout()
fig.savefig(__file__.replace("gen_react_fsm.py", "react_fsm.png"),
            dpi=150, bbox_inches="tight")
print("已保存 react_fsm.png")

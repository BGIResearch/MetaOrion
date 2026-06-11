import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 设置全局字体和PDF属性
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['figure.dpi'] = 800
plt.rcParams['savefig.dpi'] = 800
plt.rc('font', size=30)

data = {
    "Var1": ["Gut", "Oral", "Vagina", "Skin", "NA", "Body_fluids", "Nasal", "Unine"],
    "Freq": [77390, 10526, 8008, 3914, 2238, 2211, 1686, 1521]
}
df = pd.DataFrame(data)

# log10转换
df["log_Freq"] = np.log10(df["Freq"])

# 定义颜色列表
colors = ['#8AB1D2', '#F5AA61', '#9DD0C7', '#E58579', '#D0D0D0', '#D9BDDB', '#F1DFA4', '#C7988C']

# 【调整：宽度缩到和图例匹配】
fig, ax = plt.subplots(figsize=(12, 8))

# 绘制柱状图
bars = ax.bar(df["Var1"], df["log_Freq"], color=colors)

# 标注数值
for bar in bars:
    height = bar.get_height()
    idx = bars.index(bar)
    original_value = df["Freq"].iloc[idx]
    ax.text(
        bar.get_x() + bar.get_width()/2,
        height + 0.05,
        f'{original_value:,}',
        ha='center',
        va='bottom',
        fontsize=24
    )

# 坐标轴标签
ax.set_xlabel("Body Site")
ax.set_ylabel("Sample Count(log scale)")

# 去除边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Y轴刻度
yticks = np.arange(1, 6)
ax.set_yticks(yticks)
ax.set_yticklabels([f"1e+0{int(y)}" for y in yticks])

ax.set_xticks(range(len(df["Var1"])))
ax.set_xticklabels(df["Var1"], rotation=45, ha='right', va='top', rotation_mode='anchor')

plt.savefig('body_site.pdf', dpi=800, bbox_inches='tight', format='pdf')
import matplotlib.pyplot as plt
import math

# 字体设置
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['figure.dpi'] = 800
plt.rcParams['savefig.dpi'] = 800

# 数据
data = {
    "labels": ['Asia', 'Europe', 'North America', 'NA','South America', 'Oceania', 'Africa'],
    "sizes": [43099, 27660, 22332, 9262, 1220, 1822, 2099]
}
colors = ['#8AB1D2', '#F5AA61', '#9DD0C7', '#D0D0D0', '#D9BDDB', '#F1DFA4', '#C7988C']
total = sum(data["sizes"])

# 百分比显示：大于5%放内部，小于5%用外引线
def autopct_generator(pct):
    return f'{pct:.1f}%' if pct >= 5 else ''

# 创建画布
plt.figure(figsize=(7, 7))

wedges, texts, autotexts = plt.pie(
    data["sizes"],
    labels=None,
    colors=colors,
    startangle=140,
    wedgeprops={'edgecolor': 'white', 'linewidth': 1},
    autopct=autopct_generator,
    pctdistance=0.65
)

# 内部文字样式
for i, at in enumerate(autotexts):
    pct = data["sizes"][i] / total * 100
    if pct >= 5:
        # 大区块文字换行
        at.set_text(f'{data["labels"][i]}\n{pct:.1f}%')
    at.set_color('black')
    at.set_fontsize(20)
    at.set_horizontalalignment('center')

radial_len = 1.05
horizontal_len = 0.15

y_offsets = {4: 0.33, 5: 0.15, 6: 0}

# 绘制引线 + 标注
for i, size in enumerate(data["sizes"]):
    pct = size / total * 100
    if pct < 5:
        theta = (wedges[i].theta2 + wedges[i].theta1) / 2
        rad = math.radians(theta)
        x = math.cos(rad)
        y = math.sin(rad)

        turn_x = x * radial_len
        turn_y = y * radial_len

        # 应用定制的垂直偏移量，改变引线的射出方向
        if i in y_offsets:
            turn_y += y_offsets[i]
        end_x = turn_x - horizontal_len
        end_y = turn_y

        # 画两段引线
        plt.plot([x, turn_x], [y, turn_y], c='black', lw=1)
        plt.plot([turn_x, end_x], [turn_y, end_y], c='black', lw=1)

        plt.annotate(
            f'{data["labels"][i]}\n{pct:.1f}%',
            xy=(end_x, end_y),
            ha='right',
            va='center',
            fontsize=20
        )

plt.axis('equal')
plt.tight_layout()

plt.savefig('continent.pdf', dpi=800, bbox_inches='tight', format='pdf')
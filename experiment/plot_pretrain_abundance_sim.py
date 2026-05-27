import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter, MultipleLocator

lengths = np.load('/home/share/huadjyin/home/zhangkexin2/code/meta_index/output/llama/v4/pretrain/abundance_sim/crc.owen.seq.len.npy')
similarities = np.load('/home/share/huadjyin/home/zhangkexin2/code/meta_index/output/llama/v4/pretrain/abundance_sim/crc.owen.sample.pad.sim.npy')
df = pd.DataFrame({'Length': lengths, 'Similarity': similarities})
# 计算5个分位数边界（0%, 20%, 40%, 60%, 80%, 100%）
quantiles = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
bin_edges = df['Length'].quantile(quantiles).values
group_labels = [f'Q{i+1}' for i in range(5)]  # Q1-Q5
# 应用分组
df['Length_Group'] = pd.cut(df['Length'],
                           bins=bin_edges,
                           labels=group_labels,
                           include_lowest=True)
df = df.sort_values('Length_Group')
# 计算每组的样本量
group_counts = df['Length_Group'].value_counts().sort_index()
# 设置全局字体大小
plt.rcParams['font.family'] = 'Arial'
plt.rc('font', size=20)
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams.update({
    'font.size': 23,
    'axes.titlesize': 23,
    'axes.labelsize': 23,
    'xtick.labelsize': 20,
    'ytick.labelsize': 23,
    'legend.fontsize': 20
})
plt.figure(facecolor='white')
# 创建图形
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 9),
                               gridspec_kw={'height_ratios': [3, 2]})  # 调整高度比例
# sns.set_style("whitegrid")
# # 使用Set2调色板绘制小提琴图
custom_palette = sns.color_palette(["#8FB4DC", "#F5AA61", "#70CDBE", "#AC99D2", "#EB7E60", "#FFDD8E"])
ax = sns.violinplot(
    x='Length_Group',
    y='Similarity',
    data=df,
    palette=custom_palette,
    inner=None,
    linewidth=0,
    alpha=0.6,
    zorder=1,
    ax=ax1
)

# 先画散点图（放在底层）
n_groups = len(df['Length_Group'].unique())
point_colors = custom_palette[:n_groups]

sns.stripplot(
    x='Length_Group',
    y='Similarity',
    data=df,
    palette=point_colors,
    alpha=0.7,
    size=4,
    jitter=0.2,
    ax=ax1,
    zorder=2  # 散点放在中间层
)

# 最后画箱线图（放在最上层）
# boxplot = sns.boxplot(
#     x='Length_Group',
#     y='Similarity',
#     data=df,
#     width=0.2,
#     showcaps=False,
#     boxprops=dict(facecolor='white', linewidth=2.2, color='white'),
#     whiskerprops=dict(linewidth=2.2, color='white'),
#     medianprops=dict(linewidth=3, color='black'),
#     showfliers=False,
#     zorder=3,  # 箱线图放在最上层
#     ax=ax
# )
for i, color in enumerate(custom_palette[:n_groups]):
    # 获取当前组的数据
    group_data = df[df['Length_Group'] == df['Length_Group'].unique()[i]]['Similarity']

    # 在该组位置绘制箱线图
    sns.boxplot(
        x=['Q'+str(i+1)] * len(group_data),  # x坐标固定为该组位置
        y=group_data,
        width=0.2,
        showcaps=False,
        boxprops=dict(facecolor='white', linewidth=2.2, color='white'),
        whiskerprops=dict(linewidth=2.2, color='white'),
        medianprops=dict(linewidth=3, color=color),  # 使用对应的小提琴颜色
        showfliers=False,
        zorder=3,
        ax=ax1
    )



# 箱线图------------
# palette = sns.color_palette("Set2", len(quantiles))
# ax = sns.boxplot(
#     data=df,
#     x='Length_Group',
#     y='Similarity',
#     showcaps=True,
#     boxprops=dict(facecolor='none', linewidth=2),
#     whiskerprops=dict(linewidth=2),
#     capprops=dict(linewidth=2),
#     medianprops=dict(linewidth=2),
#     fliersize=0,
# )
# for patch, color in zip(ax.patches, palette):
#     patch.set_facecolor('none')       # 空心
#     patch.set_edgecolor(color)        # 边框彩色
#     patch.set_linewidth(2)
# # 添加数据点
# for i, group in enumerate(group_labels):
#     y = df[df['Length_Group'] == group]['Similarity']
#     x = np.random.normal(i, 0.08, size=len(y))  # jitter
#     plt.scatter(x, y, color=palette[i], s=50, edgecolors=palette[i], linewidths=1, zorder=10,alpha=0.4)

# 箱线图2-------------------
# palette = sns.color_palette("Set2", len(group_labels))
# box_data = [df[df['Length_Group'] == g]['Similarity'].values for g in group_labels]
# bp = plt.boxplot(
#     box_data,
#     patch_artist=True,  # 必须设置为True才能修改箱体属性
#     widths=0.6,
#     showfliers=False,  # 不显示离群点
#     labels=group_labels
# )
# # 设置所有组件颜色（统一使用palette）
# for i, color in enumerate(palette):
#     # 箱体（空心+彩色边框）
#     bp['boxes'][i].set(facecolor='none', edgecolor=color, linewidth=3)
#     # 须线（每组2条）
#     bp['whiskers'][i * 2].set(color=color, linewidth=3)  # 左侧须线
#     bp['whiskers'][i * 2 + 1].set(color=color, linewidth=3)  # 右侧须线
#     # 端帽（每组2个）
#     bp['caps'][i * 2].set(color=color, linewidth=3)  # 下端帽
#     bp['caps'][i * 2 + 1].set(color=color, linewidth=3)  # 上端帽
#     # 中位线
#     bp['medians'][i].set(color=color, linewidth=7)  # 中位线
# # 添加彩色散点（匹配箱体颜色）
# for i, group in enumerate(group_labels):
#     y = df[df['Length_Group'] == group]['Similarity']
#     x = np.random.normal(i + 1, 0.08, size=len(y))  # x坐标从1开始对齐箱体
#     plt.scatter(
#         x, y,
#         color=palette[i],
#         s=50,
#         alpha=0.5,
#         edgecolors=palette[i],
#         linewidths=1,
#         zorder=10
#     )
# # 自定义x轴标签：显示分位数范围+样本量
# # def format_xlabel(x, pos):
# #     group = group_labels[int(x)]
# #     count = group_counts[group]
# #     q_low = bin_edges[int(x)]
# #     q_high = bin_edges[int(x)+1]
# #     return f"{q_low:.0f}-{q_high:.0f}\n(N={count})"
# #
# # ax = plt.gca()
# # ax.xaxis.set_major_formatter(FuncFormatter(format_xlabel))
# ax = plt.gca()



# 生成自定义标签（分位数范围+样本量）
# new_labels = [
#     f"{int(bin_edges[i])}-{int(bin_edges[i+1])}\n(N={group_counts[group_labels[i]]})"
#     for i in range(len(group_labels))
# ]
new_labels = [
    f"{group_labels[i]}\nN={group_counts[group_labels[i]]}"
    for i in range(len(group_labels))
]

# 直接设置新标签
ax1.set_xticks(range(0, len(group_labels)))  # 箱线图的x坐标从1开始,小提琴图从0开始？
ax1.set_xticklabels(new_labels)#, rotation=45, ha='right')
# 设置y轴范围从0.6开始
ax1.set_ylim(0.8, 1)
my_y_ticks = np.arange(0.8, 1.01, 0.05)
# # # 添加统计标注（中位数）
# # medians = df.groupby('Length_Group')['Similarity'].median()
# # for i, group in enumerate(group_labels):
# #     ax.text(i, 0.58, f'Med={medians[group]:.2f}',
# #            ha='center', va='top', fontsize=20)

ax1.set_ylabel('Similarity', labelpad=10)
ax1.set_xlabel('Length Percentile Groups', labelpad=10)
ax1.set_yticks(my_y_ticks)
# # # 自定义图例
# # from matplotlib.lines import Line2D
# # legend_elements = [
# #     Line2D([0], [0], color='black', lw=2, label='Median'),
# #     Line2D([0], [0], marker='o', color='w', label='Data Points',
# #            markerfacecolor='black', markersize=10, alpha=0.3)
# # ]
# # plt.legend(handles=legend_elements, loc='upper right')
# plt.tight_layout()
# plt.savefig('/home/share/huadjyin/home/zhangkexin2/code/meta_index/experiment/results/pretrain/owen_token_pad_violin_sim.pdf',bbox_inches='tight')
# plt.show()

# import matplotlib.pyplot as plt
# import numpy as np
# # 创建图形
# plt.figure(figsize=(8, 3))
# plt.rcParams['font.family'] = 'Arial'
# plt.rc('font', size=20)
# plt.rcParams['pdf.fonttype'] = 42
# # 绘制直方图
n, bins, patches = ax2.hist(df['Length'], bins=30, color='#4e79a7', alpha=0.8,
                            edgecolor='white', linewidth=1.2)
# 计算分位数
percentiles = [0, 20, 40, 60, 80, 100]
percentile_values = np.percentile(df['Length'], percentiles)
# 添加分位数线和标注
for i, (p_val, perc) in enumerate(zip(percentile_values, percentiles)):
    ax2.axvline(x=p_val, color='red', linestyle='--', alpha=0.9, linewidth=1)
    # 在直方图内部标注，避免超出边界
    y_pos = n.max() * 0.85 if i % 2 == 0 else n.max() * 0.75
    ax2.text(p_val, y_pos, f'{perc}%\n({int(p_val)})',
             ha='center', va='center', fontsize=8, color='red',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))
# 设置标签和标题
ax2.set_ylabel('Frequency Count', labelpad=10)
ax2.set_xlabel('Sequence Length', labelpad=10)
plt.tight_layout()
plt.savefig('/home/share/huadjyin/home/zhangkexin2/code/meta_index/experiment/results/pretrain/owen_sample_pad_vilolin_sim_length.pdf',bbox_inches='tight')

plt.show()

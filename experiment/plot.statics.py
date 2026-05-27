import os
import pickle
import pandas as pd
import pycountry_convert as pc
import matplotlib.pyplot as plt

import seaborn as sns
import random
import colorsys


# def country_to_continent(country_name):
#     try:
#         country_code = pc.country_name_to_country_alpha2(country_name)
#         continent_code = pc.country_alpha2_to_continent_code(country_code)
#         continent_map = {
#             'AF': 'Africa',
#             'AS': 'Asia',
#             'EU': 'Europe',
#             'NA': 'North America',
#             'SA': 'South America',
#             'OC': 'Oceania'
#         }
#         return continent_map.get(continent_code, 'Unknown')
#     except:
#         return 'Unknown'
#
#
#
# # =========================
# # 3. 数据处理
# # =========================
#
# # 复制一份避免污染原数据
# df = pd.read_csv('/bgi-seq-model-2/datasets/zhangkexin/meta_index/metaphlan4/fine-tune/curated_LiS_CRC_20204_v0908.train_test.phe',sep='\t', index_col=0)
#
# # 只提取我用到的数据
# # samples=[]
# # with open('/bgi-seq-model-2/datasets/zhangkexin/meta_index/preprocess/metaphlan4/fine-tune/nov.specific.random.5/split1/datapath.pandisease', 'r') as f:
# #     for line in f.readlines():
# #         samples.append(line.strip().split('/')[-1].split('.json')[0])
# # df = df.loc[samples, :]
#
#
# target_disease = [
#     'IBD', 'healthy', 'CRC', 'T2D', 'OB',
#     'ACVD', 'AS', 'IGT', 'metabolic_syndrome',
#     'melanoma', 'adenoma', 'IBS', 'COVID-19',
#     'BL', 'CKD'
# ]
#
# # 国家转大洲
# df['continent'] = df['country'].apply(country_to_continent)
#
# # 去掉 others
# df.loc[~df['disease_united'].isin(target_disease), 'disease_united'] = 'others'
#
# # =========================
# # 4. 统计 (disease × continent)
# # =========================
# count_df = df.groupby(['disease_united', 'continent']).size().unstack(fill_value=0)
# count_df.index = [
#     'MS' if i == 'metabolic_syndrome'
#     else i if i.isupper()
#     else i.capitalize()
#     for i in count_df.index
# ]
#
# totals = count_df.sum(axis=1)
#
# # 筛选样本数 > 120
# mask = totals > 120
# count_df = count_df.loc[mask]
# totals = totals.loc[mask]
# order = totals.sort_values(ascending=False).index
#
# count_df = count_df.loc[order]
# totals = totals.loc[order]
# count_df = count_df.div(count_df.sum(axis=1), axis=0) * 100
#
#
# continent_order = ['Asia', 'Europe', 'North America', 'South America', 'Oceania', 'Africa']
# count_df = count_df[[c for c in continent_order if c in count_df.columns]]
#
# # =========================
# plt.style.use('default')
# plt.rcParams['font.family'] = 'Arial'
# plt.rc('font', size=18)
# plt.rcParams['pdf.fonttype'] = 42
#
# fig, ax = plt.subplots(figsize=(3, 7)) #7,7
#
# # colors = ['#FBB463', '#80B1D3', '#F47F72', '#BDBADB', '#FBF8B4', '#8DD1C6']
# # colors = ["#8FB4DC", "#F5AA61", "#70CDBE", "#EB7E60", "#AC99D2", "#FFDD8E", ]
# colors = ['#8AB1D2', '#F5AA61', '#9DD0C7', '#E58579', '#D9BDDB', '#F1DFA4']
# count_df.plot(
#     kind='barh',
#     stacked=True,
#     width=0.7,
#     ax=ax,
#     color=colors[:len(count_df.columns)]  # 自动匹配列数
# )
# # ax.set_xscale('log')
# ax.set_xlim(0, 100)
# ax.set_xticks([0, 25, 50, 75, 100])
#
# # =========================
# # 5️⃣ 标注每个疾病总数
# # =========================
# for i, (idx, total) in enumerate(totals.items()):
#     ax.text(
#         # total * 1.05,  # 稍微往右一点
#         102,
#         i,
#         f'n={total}',
#         va='center',
#         fontsize=18
#     )
#
# # =========================
# # 6️⃣ 美化
# # =========================
# # ax.set_xlabel('Sample Count (log scale)')
# ax.set_xlabel('Percent')
# ax.set_ylabel('Disease')
#
# # 去掉上右边框
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
#
# # 图例优化
# # ax.legend(
# #     title='Continent',
# #     bbox_to_anchor=(0.55, 0.17),
# #     loc='lower left',
# #     frameon=False,
# #     handlelength=1.0,  # 👉 颜色条变短（默认更长）
# #     handleheight=0.8,  # 👉 颜色条稍微变小
# #     handletextpad=0.3,  # 👉 标签离颜色条更近（关键）
# #     borderpad=0.3,  # 👉 整体 padding 更紧凑
# #     labelspacing=0.3  # 👉 行间距更紧凑
# # )
# ax.legend(
#     # title='Continent',
#     bbox_to_anchor=(1.8,-0.1),# (0.55, 0.17),
#     ncol=3,
#     # loc='lower left',
#     frameon=False,
#     handlelength=1.0,  # 👉 颜色条变短（默认更长）
#     handleheight=0.8,  # 👉 颜色条稍微变小
#     handletextpad=0.3,  # 👉 标签离颜色条更近（关键）
#     borderpad=0.3,  # 👉 整体 padding 更紧凑
#     labelspacing=0.3  # 👉 行间距更紧凑
# )
#
# # y轴反转（让最大在最上面）
# ax.invert_yaxis()
#
# plt.tight_layout()
#
# # 保存（推荐矢量图）
# plt.savefig("/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/figures/finetune/11.25/finetuine.data.statics.percent.v2.pdf", bbox_inches='tight')
#
# plt.show()



def get_clean_profile(path):
    # 转置后，行是样本，列是物种
    X = pd.read_csv(path, index_col=0, sep='\t').T
    # 使用正则/字符串方法筛选 s__ 级别，比循环快且简洁
    # 只要包含 s__ 且是路径的最后一部分
    species_cols = [c for c in X.columns if '|s__' in c and not '|t__' in c]
    X = X[species_cols]
    # 简化列名：只保留 s__ 之后的部分
    X.columns = [c.split('|s__')[-1] for c in X.columns]
    return X


def generate_random_colors(n=100, seed=111):
    random.seed(seed)
    colors = []
    for _ in range(n):
        h = random.random()
        s = random.uniform(0.2, 0.6)  # 低饱和度，保持马卡龙色系
        v = random.uniform(0.8, 1.0)  # 高明度
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        colors.append((r, g, b))
    colors_hex = ['#{:02x}{:02x}{:02x}'.format(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in colors]
    return sns.color_palette(colors_hex)
def format_species_name(name):
    if name == 'Others': return name
    # 清洗：取 s__ 后缀 -> 空格替换下划线 -> 首字母大写
    clean_name = name.split('s__')[-1].replace('_', ' ')
    return clean_name[0].upper() + clean_name[1:]
def plot_species_abundance_stackbar(profile_df, metadata_df, disease_order):
    # 1. 数据对齐
    common_samples = metadata_df.index.intersection(profile_df.index)
    merged = profile_df.loc[common_samples].copy()
    merged['disease_group'] = metadata_df.loc[common_samples, 'disease_united']
    # 2. 计算组均值
    avg_abundance = merged.groupby('disease_group').mean()
    # 3. 🌟 排序逻辑：先找出 Top 20 (从大到小)
    top_20_desc = avg_abundance.mean().sort_values(ascending=False).head(20).index.tolist()
    # 🌟 核心修改：将顺序反转 (变成从小到大)，这样绘图时会“先画 Top 20，最后画 Top 1”
    top_20_asc = top_20_desc[::-1]
    # 4. 提取数据并补齐 100%
    plot_data = avg_abundance[top_20_asc].copy()
    # Others 放在最后，也就是条形图的最右侧，通常它是最宽的
    plot_data['Others'] = 100 - plot_data.sum(axis=1)
    # 5. 排序与名称处理
    plot_data = plot_data.loc[disease_order]
    plot_data.columns = [format_species_name(c) for c in plot_data.columns]
    # 6. 绘图
    plt.style.use('default')
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 18, 'pdf.fonttype': 42})
    fig, ax = plt.subplots(figsize=(8, 7))
    # 生成 20 个颜色并反转，确保颜色与菌的对应关系也随之调整（可选）
    color_palette = list(generate_random_colors(14, seed=111)) + ['#8AB1D2', '#F5AA61', '#9DD0C7', '#E58579', '#D9BDDB', '#F1DFA4', '#D9D9D9']
    plot_data.plot(
        kind='barh', stacked=True, ax=ax, color=color_palette,
        width=0.7,
        edgecolor='white', linewidth=0.3
    )
    # 7. 美化
    ax.set_xlabel('Average Relative Abundance (%)', fontsize=18, labelpad=10)
    ax.set_ylabel('')
    ax.invert_yaxis()  # 保持疾病从上往下的顺序
    # 1. 取消 trim=True，让轴线完整显示并交汇
    sns.despine(ax=ax, top=True, right=True, trim=False)
    # 2. 强制 X 轴从 0 开始，消除柱子与 Y 轴之间的空隙
    ax.set_xlim(left=0)

    # for spine in ['left', 'bottom']:
    #     ax.spines[spine].set_linewidth(1.5)  # 调大这个值，比如 1.5 或 2.0，边框就会变粗
    #     ax.spines[spine].set_color('#000000')  # 设置为深灰色或纯黑

    # 3. 同步调节刻度线的粗细（Tick width）
    # ax.tick_params(axis='both', which='major', width=1.5, colors='#000000')

    # 🌟 图例处理：让图例的顺序重新回到从大到小（符合阅读习惯），且 Others 在最下面
    handles, labels = ax.get_legend_handles_labels()
    # 反转列表，排除最后一个(Others)，反转后再接回 Others
    new_handles = handles[:-1][::-1] + [handles[-1]]
    new_labels = labels[:-1][::-1] + [labels[-1]]
    legend = ax.legend(
        new_handles, new_labels,
        # title='Species',
        bbox_to_anchor=(1.02, 1),
        loc='upper left',
        frameon=False,
        fontsize=16,
        title_fontsize=18,
        handlelength=1.0,  # 👉 颜色条变短（默认更长）
        handleheight=0.8,  # 👉 颜色条稍微变小
        handletextpad=0.3,  # 👉 标签离颜色条更近（关键）
        borderpad=0.3,  # 👉 整体 padding 更紧凑
        labelspacing=0.3  # 👉 行间距更紧凑
    )
    for text in legend.get_texts():
        if text.get_text() != 'Others':
            text.set_style('italic')
    plt.tight_layout()
    plt.savefig("/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/figures/finetune/11.25/finetuine.data.species.statics.v3.pdf", bbox_inches='tight', dpi=800)
    return fig, ax
# --- 调用示例 ---
profile_data = get_clean_profile('/bgi-seq-model-2/datasets/zhangkexin/meta_index/metaphlan4/fine-tune/curated_LiS_CRC_20204_v1121.train_test.profile')
df = pd.read_csv('/bgi-seq-model-2/datasets/zhangkexin/meta_index/metaphlan4/fine-tune/curated_LiS_CRC_20204_v0908.train_test.phe', sep='\t', index_col=0)
# =========================
# 2. 统一疾病名称映射逻辑 (去冗余)
# =========================
target_disease = ['IBD', 'healthy', 'CRC', 'T2D', 'OB', 'ACVD', 'AS', 'IGT',
                  'metabolic_syndrome', 'melanoma', 'adenoma', 'IBS', 'COVID-19', 'BL', 'CKD']
def format_disease_name(name):
    if name == 'metabolic_syndrome': return 'MS'
    if str(name).isupper(): return name
    return str(name).capitalize()
# 统一处理 df 的疾病列
df['disease_united'] = df['disease_united'].apply(lambda x: x if x in target_disease else 'others')
df['disease_united'] = df['disease_united'].apply(format_disease_name)
# =========================
# 3. 统计与对齐
# =========================
# 确定显示顺序 (基于样本量)
order = df['disease_united'].value_counts().index

fig, ax = plot_species_abundance_stackbar(profile_data, df, order)
plt.show()



# 随机选几个样本可视化相对丰度

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import colorsys
def generate_random_colors(n=100, seed=111):
    random.seed(seed)
    colors = []
    for _ in range(n):
        h = random.random()
        s = random.uniform(0.2, 0.6)
        v = random.uniform(0.8, 1.0)
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        colors.append((r, g, b))
    colors_hex = ['#{:02x}{:02x}{:02x}'.format(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in colors]
    return colors_hex
def format_species_name(name):
    if name == 'Others' or name == '': return name
    clean_name = name.split('s__')[-1].replace('_', ' ')
    return clean_name[0].upper() + clean_name[1:]

def plot_random_samples_abundance(profile_df, n_samples=12, seed=123, max_id_len=15):
    # --- 1. 过滤长 ID 并随机采样 ---
    random.seed(seed)
    valid_sample_ids = [sid for sid in profile_df.index if len(str(sid)) <= max_id_len]

    if len(valid_sample_ids) < n_samples:
        raise ValueError(f"过滤后只剩下 {len(valid_sample_ids)} 个样本，不够抽取 {n_samples} 个！请调大 max_id_len。")

    selected_sample_ids = random.sample(valid_sample_ids, n_samples)

    # 前一半样本和后一半样本
    mid_idx = n_samples // 2
    part1_ids = selected_sample_ids[:mid_idx]
    part2_ids = selected_sample_ids[mid_idx:]

    sub_df = profile_df.loc[selected_sample_ids].copy()

    # --- 2. 筛选这批样本的 Top 20 ---
    # 算均值 -> 降序排 -> 取前20
    top_20_desc = sub_df.mean().sort_values(ascending=False).head(20).index.tolist()
    # 🌟 反转列表，确保画图时“由窄到宽”
    top_20_asc = top_20_desc[::-1]

    # --- 3. 提取数据并计算 Others ---
    plot_data = sub_df[top_20_asc].copy()
    plot_data['Others'] = 100 - plot_data.sum(axis=1)

    # --- 4. 构造带“缝隙”的数据框 ---
    part1_data = plot_data.loc[part1_ids]
    part2_data = plot_data.loc[part2_ids]
    dummy_row = pd.DataFrame(0, index=[' . . . '], columns=plot_data.columns)
    final_plot_data = pd.concat([part1_data, dummy_row, part2_data])

    # --- 5. 颜色配置 ---
    rand_14 = generate_random_colors(14, seed=111)
    fixed_6 = ['#8AB1D2', '#F5AA61', '#9DD0C7', '#E58579', '#D9BDDB', '#F1DFA4']
    others_color = ['#D9D9D9']
    color_palette = rand_14 + fixed_6 + others_color

    # --- 6. 绘图 ---
    plt.style.use('default')
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 16, 'pdf.fonttype': 42})
    fig, ax = plt.subplots(figsize=(10, 8))

    final_plot_data.columns = [format_species_name(c) for c in final_plot_data.columns]
    final_plot_data.plot(
        kind='barh', stacked=True, ax=ax, color=color_palette,
        width=0.7, edgecolor='white', linewidth=0.3
    )

    # --- 7. 视觉美化 ---
    ax.set_xlabel('Relative Abundance (%)', fontsize=18)
    ax.set_ylabel('Samples', fontsize=18)
    ax.set_xlim(0, 100)
    ax.invert_yaxis()

    sns.despine(ax=ax, top=True, right=True, trim=False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(1.5)
        ax.spines[spine].set_color('#333333')
    ax.tick_params(width=1.5, color='#333333')

    # 图例恢复从大到小排列 (Top 1 在上)
    handles, labels = ax.get_legend_handles_labels()
    new_h = handles[:-1][::-1] + [handles[-1]]
    new_l = labels[:-1][::-1] + [labels[-1]]
    legend = ax.legend(
        new_h, new_l, title='Species', bbox_to_anchor=(1.02, 1.05),
        loc='upper left', frameon=False, fontsize=16
    )
    for text in legend.get_texts():
        if text.get_text() != 'Others': text.set_style('italic')

    plt.tight_layout()
    plt.savefig("/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/figures/finetune/11.25/finetuine.some.data.species.statics.pdf", bbox_inches='tight', dpi=800)

    return fig, ax


# 要是选所有样本的top20 species
def plot_random_samples_abundance(profile_df, n_samples=12, seed=123, max_id_len=15):
    # --- 1. 🌟 核心修改：基于“所有样本”计算全局 Top 20 ---
    # 先算出所有样本的平均值来定 Top 20
    global_top_20_desc = profile_df.mean().sort_values(ascending=False).head(20).index.tolist()
    # 反转列表，确保画图时“由窄到宽”
    global_top_20_asc = global_top_20_desc[::-1]

    # --- 2. 过滤长 ID 并随机采样 ---
    random.seed(seed)
    valid_sample_ids = [sid for sid in profile_df.index if len(str(sid)) <= max_id_len]

    if len(valid_sample_ids) < n_samples:
        raise ValueError(f"过滤后只剩下 {len(valid_sample_ids)} 个样本，不够抽取 {n_samples} 个！请调大 max_id_len。")

    selected_sample_ids = random.sample(valid_sample_ids, n_samples)

    # 前一半样本和后一半样本
    mid_idx = n_samples // 2
    part1_ids = selected_sample_ids[:mid_idx]
    part2_ids = selected_sample_ids[mid_idx:]

    # 提取抽中样本的数据
    sub_df = profile_df.loc[selected_sample_ids].copy()

    # --- 3. 提取全局 Top 20 的数据并计算 Others ---
    # 这里用的是 sub_df (抽中的样本)，但列名用的是 global_top_20_asc (全局确定的前20菌)
    plot_data = sub_df[global_top_20_asc].copy()
    plot_data['Others'] = 100 - plot_data.sum(axis=1)

    # --- 4. 构造带“缝隙”的数据框 ---
    part1_data = plot_data.loc[part1_ids]
    part2_data = plot_data.loc[part2_ids]
    dummy_row = pd.DataFrame(0, index=[' . . . '], columns=plot_data.columns)
    final_plot_data = pd.concat([part1_data, dummy_row, part2_data])

    # --- 5. 颜色配置 ---
    rand_14 = generate_random_colors(14, seed=111)
    fixed_6 = ['#8AB1D2', '#F5AA61', '#9DD0C7', '#E58579', '#D9BDDB', '#F1DFA4']
    others_color = ['#D9D9D9']
    color_palette = rand_14 + fixed_6 + others_color

    # --- 6. 绘图 ---
    plt.style.use('default')
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 16, 'pdf.fonttype': 42})
    fig, ax = plt.subplots(figsize=(10, 8))

    final_plot_data.columns = [format_species_name(c) for c in final_plot_data.columns]
    final_plot_data.plot(
        kind='barh', stacked=True, ax=ax, color=color_palette,
        width=0.7, edgecolor='white', linewidth=0.3
    )

    # --- 7. 视觉美化 ---
    ax.set_xlabel('Relative Abundance (%)', fontsize=18)
    ax.set_ylabel('Samples', fontsize=18)
    ax.set_xlim(0, 100)
    ax.invert_yaxis()

    sns.despine(ax=ax, top=True, right=True, trim=False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(1.5)
        ax.spines[spine].set_color('#333333')
    ax.tick_params(width=1.5, color='#333333')

    # 图例恢复从大到小排列 (Top 1 在上)
    handles, labels = ax.get_legend_handles_labels()
    new_h = handles[:-1][::-1] + [handles[-1]]
    new_l = labels[:-1][::-1] + [labels[-1]]
    legend = ax.legend(
        new_h, new_l, title='Species', bbox_to_anchor=(1.02, 1.05),
        loc='upper left', frameon=False, fontsize=16
    )
    for text in legend.get_texts():
        if text.get_text() != 'Others': text.set_style('italic')

    plt.tight_layout()
    plt.savefig("/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/figures/finetune/11.25/finetuine.some.data.species.statics.v2.pdf", bbox_inches='tight', dpi=800)

    return fig, ax
# 调用示例：
fig, ax = plot_random_samples_abundance(profile_data, n_samples=12)
plt.show()


# 统计v22 8W+样本分身体部位采样的species组成
import os
import glob
import random
import colorsys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
# ==========================================
# 1. 核心处理与颜色生成函数
# ==========================================
def get_clean_public_profile(path, sample_id):
    """
    专门用于读取 public 单样本的原始 MetaPhlAn4 输出文件。
    自动跳过头部日志，仅提取 relative_abundance 列，并转化为 [1行 x N列] 的格式。
    """
    try:
        # 1. 动态寻找表头所在的行 (寻找以 #clade_name 开头的那一行)
        skip_lines = 0
        with open(path, 'r') as f:
            for line in f:
                if line.startswith('#clade_name'):
                    break
                skip_lines += 1
        # 2. 从表头所在行开始读取数据
        df = pd.read_csv(path, sep='\t', skiprows=skip_lines)
        # 修正列名（原文件的第一列叫 #clade_name，把前面的 # 去掉方便处理）
        df.rename(columns={'#clade_name': 'clade_name'}, inplace=True)
        # 3. 仅保留第一列 (菌名) 和 relative_abundance 列
        if 'relative_abundance' not in df.columns:
            return pd.DataFrame()
        df = df[['clade_name', 'relative_abundance']]
        # 4. 过滤为 species 级别 (包含 |s__ 且不含 |t__)
        df = df[df['clade_name'].str.contains(r'\|s__') & ~df['clade_name'].str.contains(r'\|t__')].copy()
        # 5. 清理菌名
        df['clade_name'] = df['clade_name'].apply(lambda x: x.split('|s__')[-1])
        # 6. 转置为以 sample_id 为索引的单行 DataFrame
        df.set_index('clade_name', inplace=True)
        df_transposed = df.T
        df_transposed.index = [sample_id]
        return df_transposed
    except Exception as e:
        print(f"⚠️ Public文件读取异常 {path} | 错误: {e}")
        return pd.DataFrame()
def get_clean_inhouse_profile(path):
    """
    专门用于读取 inhouse 的多样本合并宽表 (如 SZ-4D_3.3k_gut.profile)。
    返回的 DataFrame 行是样本，列是物种。
    """
    try:
        # 同样为了安全，跳过可能存在的顶部合并日志
        skip_lines = 0
        with open(path, 'r') as f:
            for line in f:
                # 合并表的表头可能是 #clade_name 或者直接是 clade_name 或者 UNKNOWN
                if 'clade_name' in line or 'UNKNOWN' in line or line.startswith('#'):
                    # 碰到 #clade_name 就是头，如果是其他 # 开头的说明是日志
                    if line.startswith('#clade_name') or line.startswith('clade_name'):
                        break
                skip_lines += 1
        # 读取数据并将第一列(物种名)设为 index
        df = pd.read_csv(path, sep='\t', skiprows=skip_lines, index_col=0)
        # 提取 Species 行并清理名称
        species_idx = [idx for idx in df.index if isinstance(idx, str) and '|s__' in idx and '|t__' not in idx]
        df = df.loc[species_idx]
        df.index = [idx.split('|s__')[-1] for idx in df.index]
        # 转置 (变成 行=样本，列=物种)
        return df.T
    except Exception as e:
        print(f"⚠️ Inhouse合并文件读取异常 {path} | 错误: {e}")
        return pd.DataFrame()
# def get_clean_inhouse_profile(path):
#     """
#     通用读取函数：支持单样本或多样本的大表。
#     返回的 DataFrame 行是样本，列是物种
#     """
#     try:
#         # 转置后，行是样本，列是物种
#         X = pd.read_csv(path, index_col=0, sep='\t').T
#         species_cols = [c for c in X.columns if '|s__' in c and not '|t__' in c]
#         X = X[species_cols]
#         X.columns = [c.split('|s__')[-1] for c in X.columns]
#         return X
#     except Exception as e:
#         print(f"文件读取异常 {path} | 错误: {e}")
#         return pd.DataFrame()
def generate_random_colors(n=100, seed=123):
    random.seed(seed)
    colors = []
    for _ in range(n):
        h = random.random()
        s = random.uniform(0.2, 0.6)
        v = random.uniform(0.8, 1.0)
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        colors.append((r, g, b))
    return ['#{:02x}{:02x}{:02x}'.format(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in colors]
def format_species_name(name):
    if name == 'Others': return name
    clean_name = name.split('s__')[-1].replace('_', ' ')
    return clean_name[0].upper() + clean_name[1:]
# ==========================================
# 2. 路径与全局配置
# ==========================================
meta_path = '/bgi-seq-model-2/datasets/zhangkexin/meta_index/metaphlan4/pre-train/v22.profile.bodysite.path.csv'
public_dir = '/bgi-seq-model-2/datasets/zhangkexin/meta_index/metaphlan4/pre-train/mp4_profile'
inhouse_base = '/bgi-seq-model-2/datasets/zhangkexin/meta_index/metaphlan4/pre-train'
output_dir = './bodysite_figures'

os.makedirs(output_dir, exist_ok=True)

# ==========================================
# 🌟 3. 预加载 inhouse 数据池 (提速 100 倍的核心)
# ==========================================
print("正在预加载所有 inhouse profile 大表 (可能需要几分钟，请稍候)...")
inhouse_files = glob.glob(os.path.join(inhouse_base, 'SZ-4D*')) + \
                glob.glob(os.path.join(inhouse_base, 'VIGANO*'))

inhouse_dfs = []
for f in inhouse_files:
    if os.path.isfile(f): # 确保是文件而不是文件夹
        print(f"  -> 载入: {os.path.basename(f)}")
        df_bulk = get_clean_inhouse_profile(f)
        inhouse_dfs.append(df_bulk)

if inhouse_dfs:
    # 将所有的 inhouse 表合并为一个巨大的数据库，缺失的物种自动填 0
    inhouse_db = pd.concat(inhouse_dfs, axis=0).fillna(0)
    # 如果有重复的样本名，保留第一个
    inhouse_db = inhouse_db[~inhouse_db.index.duplicated(keep='first')]  # 会有重复吗？
    print(f"✅ Inhouse 数据库加载完成，共包含 {len(inhouse_db)} 个样本。\n")
else:
    inhouse_db = pd.DataFrame()
    print("⚠️ 警告：没有找到任何 inhouse 文件。\n")
# ==========================================
# 4. 按身体部位汇总数据与局部 Top 20 计算
# ==========================================

v22_df = pd.read_csv(meta_path, index_col=1)
grouped = v22_df.groupby('body_site')

plot_rows = []
global_top5_species = set()

for body_site, group_df in grouped:
    print(f"处理部位: {body_site} (n={len(group_df)})")

    public_samples = group_df[group_df['source'] == 'public'].index.tolist()
    inhouse_samples = group_df[group_df['source'] == 'inhouse'].index.tolist()

    # 🌟 核心优化：不再使用列表收集 DataFrame，而是直接用 Series 累加丰度
    site_sum = pd.Series(dtype=float)
    site_sample_count = 0

    # 🌟 4.1 处理 inhouse 样本 (批量截取并累加)
    if not inhouse_db.empty and inhouse_samples:
        valid_inhouse = [s for s in inhouse_samples if s in inhouse_db.index]
        if valid_inhouse:
            inhouse_chunk = inhouse_db.loc[valid_inhouse]
            site_sample_count += len(inhouse_chunk)
            # 按列求和并累加到总和中，fill_value=0 保证不同样本独有的菌能完美对齐
            site_sum = site_sum.add(inhouse_chunk.sum(), fill_value=0)

    # 🌟 4.2 处理 public 样本 (逐个读取并累加，读完立刻释放内存)
    if public_samples:
        for sample_id in tqdm(public_samples, desc="  Loading Public", leave=False):
            file_path = os.path.join(public_dir, f"{sample_id}.mp4.profile")
            if os.path.exists(file_path):
                sample_profile = get_clean_public_profile(file_path, sample_id)
                if not sample_profile.empty:
                    site_sample_count += 1
                    # 由于 public 函数返回的是 1行 的 DataFrame，iloc[0] 取出变为 Series 进行累加
                    site_sum = site_sum.add(sample_profile.iloc[0], fill_value=0)

    # 如果该部位没有任何有效数据，跳过
    if site_sample_count == 0:
        continue

    # 🌟 4.3 直接计算均值：彻底绕过 pd.concat 大表拼接！
    avg_abundance = (site_sum / site_sample_count).sort_values(ascending=False)

    # 提取部位 Top 20，剩下的归为 Others
    top20 = avg_abundance.head(20)
    others_sum = avg_abundance.iloc[20:].sum() if len(avg_abundance) > 20 else 0

    row_data = top20.to_dict()
    row_data['Others'] = others_sum
    row_data['body_site'] = body_site
    plot_rows.append(row_data)

    # 收集部位 Top 5 加入全局白名单
    global_top5_species.update(avg_abundance.head(5).index.tolist())
# v22_df = pd.read_csv(meta_path, index_col=1)
# grouped = v22_df.groupby('body_site')
# plot_rows = []
# global_top5_species = set()
# for body_site, group_df in grouped:
#     print(f"处理部位: {body_site} (n={len(group_df)})")
#     # 将该部位的样本分为 public 和 inhouse 两拨
#     public_samples = group_df[group_df['source'] == 'public'].index.tolist()
#     inhouse_samples = group_df[group_df['source'] == 'inhouse'].index.tolist()
#     site_dfs = []
#     # 🌟 4.1 极速提取 inhouse 样本
#     if not inhouse_db.empty and inhouse_samples:
#         # 找出存在于总库里的样本
#         valid_inhouse = [s for s in inhouse_samples if s in inhouse_db.index]
#         if valid_inhouse:
#             site_dfs.append(inhouse_db.loc[valid_inhouse])
#     # 🌟 4.2 常规逐个读取 public 样本
#     if public_samples:
#         for sample_id in tqdm(public_samples, desc="  Loading Public", leave=False):
#             file_path = os.path.join(public_dir, f"{sample_id}.mp4.profile")
#             if os.path.exists(file_path):
#                 sample_profile = get_clean_public_profile(file_path, sample_id)
#                 if not sample_profile.empty:
#                     # 强制重置 index，防止单样本读取时表头不是样本名
#                     sample_profile.index = [sample_id]
#                     site_dfs.append(sample_profile)
#     # 如果该部位没有任何有效数据，跳过
#     if not site_dfs:
#         print(f'{body_site}部位没有任何有效数据')
#         continue
#     # 合并该部位所有样本，计算平均丰度
#     merged_site_df = pd.concat(site_dfs, axis=0).fillna(0)
#     avg_abundance = merged_site_df.mean().sort_values(ascending=False)
#     # 提取部位 Top 20，剩下的归为 Others
#     top20 = avg_abundance.head(20)
#     others_sum = avg_abundance.iloc[20:].sum() if len(avg_abundance) > 20 else 0
#     row_data = top20.to_dict()
#     row_data['Others'] = others_sum
#     row_data['body_site'] = body_site
#     plot_rows.append(row_data)
#     # 收集部位 Top 5 加入全局白名单
#     global_top5_species.update(avg_abundance.head(5).index.tolist())
# ==========================================
# 5. 组装全局矩阵与排序
# ==========================================
print("\n正在生成全局图表...")
plot_df = pd.DataFrame(plot_rows).set_index('body_site').fillna(0)
cols_species = [c for c in plot_df.columns if c != 'Others']
cols_species_sorted = plot_df[cols_species].mean().sort_values(ascending=False).index.tolist()
final_cols = cols_species_sorted + ['Others']
plot_df = plot_df[final_cols]
plot_df.columns = [format_species_name(c) for c in plot_df.columns]
global_top5_formatted = [format_species_name(c) for c in global_top5_species]
# ==========================================
# 6. 全局绘图与图例过滤
# ==========================================
plt.style.use('default')
plt.rcParams.update({'font.family': 'Arial', 'font.size': 18, 'pdf.fonttype': 42})
# 动态调整图片高度，身体部位越多图越长
fig, ax = plt.subplots(figsize=(14, max(8, len(plot_df) * 0.8)))
color_palette = generate_random_colors(len(cols_species_sorted), seed=101) + ['#E0E0E0']
plot_df.plot(
    kind='barh', stacked=True, ax=ax, color=color_palette,
    width=0.75, edgecolor='white', linewidth=0.5
)
# 坐标轴美化
ax.set_xlabel('Average Relative Abundance (%)', fontsize=20, labelpad=15)
ax.set_ylabel('')
ax.invert_yaxis()
ax.set_xlim(0, 100)
sns.despine(ax=ax, top=True, right=True, trim=False)
for spine in ['left', 'bottom']:
    ax.spines[spine].set_linewidth(1.5)
ax.tick_params(width=1.5, colors='#333333')
# 🌟 图例拦截器：只显示白名单中的 Top 5 及 Others
handles, labels = ax.get_legend_handles_labels()
filtered_handles, filtered_labels = [], []
for h, l in zip(handles, labels):
    if l in global_top5_formatted or l == 'Others':
        filtered_handles.append(h)
        filtered_labels.append(l)
legend = ax.legend(
    filtered_handles, filtered_labels,
    title='Species (Top 5 per site)',
    bbox_to_anchor=(1.02, 1), loc='upper left',
    frameon=False, fontsize=14, title_fontsize=16
)
for text in legend.get_texts():
    if text.get_text() != 'Others':
        text.set_style('italic')
plt.tight_layout()
# save_path = os.path.join(output_dir, "All_BodySites_Species_Composition.pdf")
# plt.savefig(save_path, bbox_inches='tight', dpi=800)
plt.show()
# print(f"🎉 绘图圆满完成！合并图已保存至: {save_path}")


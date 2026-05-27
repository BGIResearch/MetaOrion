import pickle
import random
import colorsys
import warnings

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu
import matplotlib.transforms as transforms
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator

from sklearn.metrics import (
    roc_auc_score, roc_curve, auc,
    accuracy_score, precision_score, recall_score, confusion_matrix, balanced_accuracy_score
)


def calculate_mndi_v2(individual_network, sample_list, taxa2id,
                      case_samples, ctrl_samples,
                      bad_taxa, good_taxa):
    """
    individual_network: (n_sample, n_taxa, n_taxa)
    """
    # 1. 索引准备
    bad_idx = [taxa2id[t] for t in bad_taxa if t in taxa2id]
    good_idx = [taxa2id[t] for t in good_taxa if t in taxa2id]
    bad_list = [t for t in bad_taxa if t in taxa2id]
    good_list = [t for t in good_taxa if t in taxa2id]

    mndi_features = []

    for i in range(individual_network.shape[0]):
        sample_feature_dict = {'sample_id': sample_list[i]}
        # 获取当前样本的邻接矩阵 (|cosine| > 0 或设定阈值)
        # 假设原始数据是 correlation，转为二值或权重图
        adj = np.abs(individual_network[i])
        # adj = individual_network[i]
        adj[np.isnan(adj)] = 0  # 处理空值

        # ---------------要是看边数的话
        # threshold = 0
        # adj = (adj > threshold).astype(float)

        np.fill_diagonal(adj, 0)


        # 特征 A: 坏菌内部相互作用总和 (Bad-Bad)
        bad_internal = adj[np.ix_(bad_idx, bad_idx)].sum() / 2

        # 特征 B: 坏菌与全网的相互作用 (Bad-All)
        bad_all = adj[bad_idx, :].sum()

        # 特征 C: 好菌与全网的相互作用 (Good-All)
        good_internal = adj[np.ix_(good_idx, good_idx)].sum() / 2
        good_all = adj[good_idx, :].sum()
        bad_good_internal = adj[np.ix_(bad_idx, good_idx)].sum()

        # 特征 D: 全网总连接
        total_edges = adj.sum() / 2

        sample_feature_dict.update({
            'f_bad_bad': bad_internal,
            'f_bad_all': bad_all,
            'f_bad_dominance': bad_all / (total_edges + 1e-5),
            'f_good_good': good_internal,
            'f_good_all': good_all,
            'f_good_dominance': good_all / (total_edges + 1e-5),
            'f_total_edges': total_edges,
            'f_tmp': (bad_internal + 1e-5) / (good_internal + 1e-5),
            'f_bad_good_internal': bad_good_internal,
            # 'f_bad_good_internal': (bad_internal - good_internal) / (good_internal + bad_internal + 1e-5),
            'f_good_loss': 1 / (good_all + 1e-5)
        })

        mndi_features.append(sample_feature_dict)

    df_features = pd.DataFrame(mndi_features)
    df_features['group'] = df_features['sample_id'].apply(lambda x: 'case' if x in case_samples else 'ctrl')

    df_features['MNDI'] = np.log10(df_features['f_tmp'] + 1e-5)

    return df_features




# -------------------------分cohort
def generate_random_colors(n=100, seed=111):
    random.seed(seed)
    colors = []
    for _ in range(n):
        h = random.random()
        s = random.uniform(0.2, 0.6)  # 低饱和度
        v = random.uniform(0.8, 1.0)  # 高明度
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        colors.append((r, g, b))

    colors_hex = [
        '#{:02x}{:02x}{:02x}'.format(int(r * 255), int(g * 255), int(b * 255))
        for r, g, b in colors
    ]
    sns_palette = sns.color_palette(colors_hex)
    return sns_palette


def plot_cohort_case_ctrl_mndi(df_features, label_df, disease):
    # --- 1. 数据整合与排序 ---
    label_df['sample_id'] = list(label_df.index)
    label_df['disease_united'] = ['Healthy' if i == 'healthy' else i for i in label_df['disease_united']]
    # 注意：确保 label_df 中包含 'disease_united' 列
    plot_df = pd.merge(df_features, label_df[['sample_id', 'project', 'disease_united']], on='sample_id', how='left')
    # 强制转换 group 为分类变量并指定顺序，这决定了同一项目内的上下顺序
    plot_df['group'] = pd.Categorical(plot_df['group'], categories=['case', 'ctrl'], ordered=True)
    # 计算样本量 (按 project 和 disease_united 分组)
    counts = plot_df.groupby(['project', 'disease_united']).size().reset_index(name='n')
    plot_df = pd.merge(plot_df, counts, on=['project', 'disease_united'])
    # --- 核心修改：构造 Y 轴标签 ---
    # 格式：Project | Disease (n=xx)
    plot_df['y_label'] = plot_df.apply(
        lambda x: f"{x['project']} | {x['disease_united']} (n={x['n']})", axis=1
    )
    # 排序：先按项目名排，再按 group 排 (确保 Case 在上面)
    plot_df = plot_df.sort_values(['project', 'group'])
    y_order = list(plot_df['y_label'].unique())
    projects = plot_df['project'].unique()
    # --- 2. 绘图初始化 ---
    fig, ax = plt.subplots(figsize=(12, len(y_order) * 0.7))
    plt.rc('font', size=22)
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['pdf.fonttype'] = 42
    sns.set_style("ticks")
    # --- 核心修改：图例颜色映射 ---
    # 自动获取所有疾病名称并分配颜色（Healthy 用蓝色，其他疾病用不同色调的红色/橘色）
    unique_diseases = plot_df['disease_united'].unique()
    # color_map = {}
    # for d in unique_diseases:
    #     if 'healthy' in d.lower() or 'ctrl' in d.lower() or 'control' in d.lower():
    #         color_map[d] = '#3498db'  # 蓝色
    #     else:
    #         color_map[d] = '#e74c3c'  # 红色
    custom_palette = generate_random_colors(len(unique_diseases), seed=101)
    # --- 3. 绘制背景色块 ---
    trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)
    for i, proj in enumerate(projects):
        if i % 2 == 0:
            proj_labels = [label for label in y_order if label.startswith(proj)]
            indices = [y_order.index(l) for l in proj_labels]
            if indices:
                start_y, end_y = min(indices) - 0.5, max(indices) + 0.5
                rect = plt.Rectangle((-1, start_y), 2.2, end_y - start_y,
                                     transform=trans, color='lightgray', alpha=0.5,
                                     zorder=0, clip_on=False)
                ax.add_patch(rect)
    # --- 4. 绘制图形 (hue 使用 disease_united) ---
    # sns.boxplot(
    #     data=plot_df, x='MNDI', y='y_label', hue='disease_united',
    #     order=y_order, palette=color_map, dodge=False, width=0.6, fliersize=0, ax=ax, zorder=2
    # )
    # sns.stripplot(
    #     data=plot_df, x='MNDI', y='y_label', color='black',
    #     alpha=0.2, size=3, jitter=True, ax=ax, zorder=3
    # )
    # def filter_extreme(group):
    #     q_low = group['MNDI'].quantile(0.00001)  # 砍掉下限最极端的 1%
    #     q_high = group['MNDI'].quantile(0.99999) # 砍掉上限最极端的 1%
    #     return group[(group['MNDI'] >= q_low) & (group['MNDI'] <= q_high)]
    #
    # violin_df = plot_df.groupby('y_label', group_keys=False).apply(filter_extreme)
    v = sns.violinplot(
        data=plot_df, x='MNDI', y='y_label', hue='y_label',
        order=y_order, palette=custom_palette, dodge=False, color='#333333',
        inner=None, linewidth=1.5, cut=0, ax=ax, zorder=2, legend=False, bw_adjust=0.8
    )
    for pc in v.collections:
        pc.set_edgecolor('#333333')  # 强制边框颜色
        pc.set_alpha(1)  # 确保边框是不透明的

    # 2. 叠加内部的迷你箱线图
    sns.boxplot(
        data=plot_df, x='MNDI', y='y_label',
        order=y_order, dodge=False,
        width=0.3,  # 🌟 把箱子调得很窄，刚好放在小提琴内部
        color='white',  # 箱体填充白色
        linewidth=1.5,  # 箱体外边框粗细
        fliersize=4,  # 隐藏离群点（避免和后面的散点图重复）
        flierprops={'marker': 'o', 'markerfacecolor': 'black', 'markeredgecolor': 'black', 'markersize': 4,
                    'alpha': 0.7},
        showcaps=False,  # 隐藏胡须末端的横线，使其更清爽
        ax=ax, zorder=3,
        boxprops={'edgecolor': '#333333'},  # 箱体边框颜色
        whiskerprops={'color': '#333333', 'linewidth': 1.5},  # 上下须的颜色和粗细
        medianprops={'color': '#e74c3c', 'linewidth': 3}  # 🌟 核心：中位线加粗！
    )
    # --- 5. P值计算与标注 ---
    x_max = plot_df['MNDI'].max()
    annotation_x = x_max * 1.05
    for i, label_str in enumerate(y_order):
        proj_name = label_str.split(' | ')[0]
        proj_data = plot_df[plot_df['project'] == proj_name]
        case_vals = proj_data[proj_data['group'] == 'case']['MNDI']
        ctrl_vals = proj_data[proj_data['group'] == 'ctrl']['MNDI']
        stars = ""
        # 逻辑：在 Case 所在的行显示该项目与 Ctrl 的对比结果
        # 我们通过 group 来判断当前行是否为 Case
        current_group = plot_df[plot_df['y_label'] == label_str]['group'].iloc[0]
        if current_group == 'case':
            if len(ctrl_vals) > 0:
                stat, p_val = mannwhitneyu(case_vals, ctrl_vals)
                if p_val < 0.001:
                    stars = '***'
                elif p_val < 0.01:
                    stars = '**'
                elif p_val < 0.05:
                    stars = '*'
                else:
                    stars = ''
            else:
                stars = '—'
        elif current_group == 'ctrl':
            if len(case_vals) == 0:
                stars = '—'
        if stars:
            ax.text(annotation_x, i, stars, va='center', ha='center',
                    fontsize=20, fontweight='bold', color='#333333')
    # --- 6. 美化与保存 ---
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    # plt.title('MNDI Distribution by Disease Type', fontsize=14, pad=20)
    ax.tick_params(axis='y', labelsize=22)  # 调节 Y 轴疾病标签的字体大小
    ax.tick_params(axis='x', labelsize=22)  # 调节 X 轴数字的字体大小
    plt.xlabel('MNDI Score', fontsize=24)
    plt.ylabel('')
    # --- 核心修改：自定义图例 ---
    # 手动创建两个代表 Disease 和 Healthy 的色块，颜色与你的前面设定的保持一致
    # disease_patch = mpatches.Patch(color='#e74c3c', label='Disease')
    # healthy_patch = mpatches.Patch(color='#3498db', label='Healthy')
    # # 覆盖默认图例
    # plt.legend(handles=[disease_patch, healthy_patch], title='Condition',
    #            bbox_to_anchor=(1.05, 1), loc='upper left')
    sns.despine()
    ax.xaxis.set_major_locator(MultipleLocator(1))
    plt.xlim(right=annotation_x * 1.1)
    fig.subplots_adjust(left=0.35)  # 预留左侧空间
    save_path = f'/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/figures/finetune/11.25/networks/4.1/{disease}.project.individual.case.ctrl.mndi.pdf'
    plt.savefig(save_path, dpi=800, bbox_inches='tight')
    plt.show()


def plot_grouped_boxplot(df, disease, filename, margin=0.5, custom_ylabel='Edge Weight', roc_auc=0):
    # --- 1. 数据重构：宽表转长表 (Melt) ---
    # 把 f_bad_bad 和 f_good_good 融合成一列变量，方便 Seaborn 使用 x 和 hue
    df_melt = pd.melt(df, id_vars=['group'],
                      value_vars=['f_bad_bad', 'f_good_good'],
                      var_name='feature', value_name='score')
    # 强制排序
    df_melt['group'] = pd.Categorical(df_melt['group'], categories=['case', 'ctrl'], ordered=True)
    # --- 2. 绘图初始化 ---
    plt.rc('font', size=20)
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['axes.linewidth'] = 2.0
    # 画布稍微加宽一点以适应两组并排的箱子
    fig, ax = plt.subplots(figsize=(7, 6), facecolor='white')
    custom_palette = ['#E44A33', '#4DBAD6']
    # --- 3. 绘制分组箱线图 ---
    sns.boxplot(
        x='feature',  # X 轴现在是特征名 (f_bad_bad, f_good_good)
        y='score',  # Y 轴是数值
        hue='group',  # 颜色还是区分 case 和 ctrl
        data=df_melt,
        palette=custom_palette,
        dodge=True,  # 🌟 核心：必须为 True 才能让 case 和 ctrl 并排显示
        width=0.6,  # 箱体总宽度
        linewidth=2.0,
        showfliers=True,
        fliersize=4,
        flierprops={'marker': 'o', 'markerfacecolor': 'black', 'markeredgecolor': 'black', 'markersize': 2.5,
                    'alpha': 1},
        legend=True,  # 这里开启图例，后面会重新定位美化
        boxprops={'edgecolor': 'black', 'linewidth': 2.0},
        whiskerprops={'color': 'black', 'linewidth': 2.0},
        medianprops={'color': 'black', 'linewidth': 2.0},
        capprops={'color': 'black', 'linewidth': 2.0},
        ax=ax
    )
    ax.margins(x=0.1)
    # --- 4. 动态计算 Y 轴范围和刻度 ---
    # y_data_min = df_melt['score'].min()
    # y_data_max = df_melt['score'].max()
    # # 预留比单组稍微多一点的空间画两条 P 值线
    # y_pos = y_data_max + abs(y_data_max - y_data_min) * 0.12
    # tick_min = np.floor(y_data_min * 2) / 2
    # tick_max = np.ceil((y_pos + 0.3) * 2) / 2
    # y_ticks = np.arange(tick_min, tick_max, margin)
    # ax.set_ylim(tick_min - 0.2, tick_max)
    # ax.set_yticks(y_ticks)
    # --- 4. 动态计算 Y 轴范围 (保留默认刻度并自动顶部留白) ---
    # 此时 seaborn 已经画好了箱线图，并自动生成了舒适的 Y 轴范围
    ymin, ymax = ax.get_ylim()
    y_span = ymax - ymin  # 获取当前 Y 轴的视觉跨度

    # 获取数据的真实最大值，作为画 P 值横线的起跳点
    y_data_max = df_melt['score'].max()

    # P 值线条的高度：在最顶端的数据点上方，加上整体跨度的 5%
    y_pos = y_data_max + y_span * 0.05

    # 重新设置 Y 轴范围：下限保持默认不动，上限在 P 值线条基础上再留出 10% 空间给文字
    ax.set_ylim(ymin, y_pos + y_span * 0.1)
    # --- 5. 计算统计显著性并添加双 P 值标注 ---
    features = ['f_bad_bad', 'f_good_good']
    # 🌟 核心坐标计算：当 total width=0.6 时，双类别的偏移量恰好为 0.15
    offset = 0.15
    print("=" * 50)
    print("统计检验结果：")
    for i, feat in enumerate(features):
        case_vals = df[df['group'] == 'case'][feat].dropna()
        ctrl_vals = df[df['group'] == 'ctrl'][feat].dropna()
        u_stat, p_value = stats.mannwhitneyu(case_vals, ctrl_vals)
        ks_stat, ks_p = stats.ks_2samp(case_vals, ctrl_vals)
        if p_value == 0.0:
            p_display = "P < 2.22e-308"
        else:
            p_display = f"P = {p_value:.2e}"
        # 绘制该特征上方的显著性线
        x1 = i - offset  # Case 盒子的中心 x 坐标
        x2 = i + offset  # Ctrl 盒子的中心 x 坐标
        plt.plot([x1, x1, x2, x2], [y_pos * 0.96, y_pos, y_pos, y_pos * 0.96], color='black', lw=1.5)
        plt.text(i, y_pos * 1.02, p_display, ha='center', va='bottom', fontsize=18)
        # 打印信息
        print(f"\n[{feat}]")
        print(f"Mann-Whitney U: U = {u_stat:,.0f}, {p_display}")
        print(f"Case组: 均值 = {np.mean(case_vals):.2f} ± {np.std(case_vals):.2f}")
        print(f"Ctrl组: 均值 = {np.mean(ctrl_vals):.2f} ± {np.std(ctrl_vals):.2f}")
    print("=" * 50)
    # --- 6. 标签与图例美化 ---
    # 重新命名 X 轴的两个大类，你可以根据需要修改文字
    ax.set_xticklabels(['PR-increasing taxa', 'PR-decreasing taxa'], fontsize=20)#, rotation=30,  ha='right')
    # 获取并美化图例 (替换默认的 'case' / 'ctrl' 为实际名字加样本量)
    handles, labels = ax.get_legend_handles_labels()
    n_case = len(df[df['group'] == 'case'])
    n_ctrl = len(df[df['group'] == 'ctrl'])
    new_labels = [f'{disease} (n={n_case})', f'Healthy (n={n_ctrl})']
    new_labels = [f'{disease}', f'Healthy']
    ax.legend(handles, new_labels, loc='upper right', frameon=False, fontsize=20,bbox_to_anchor=(0.92, 0.88),
              handlelength=1.0,  # 👉 颜色条变短（默认更长）
              handleheight=0.8,  # 👉 颜色条稍微变小
              handletextpad=0.3,  # 👉 标签离颜色条更近（关键）
              borderpad=0.3,  # 👉 整体 padding 更紧凑
              labelspacing=0.3  # 👉 行间距更紧凑
              )
    plt.ylabel(custom_ylabel)
    plt.xlabel('')
    plt.grid(True, alpha=0.3, linestyle='--', axis='y')
    plt.tight_layout()
    # 保存图片
    plt.savefig(
        f'/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/figures/finetune/11.25/networks/4.1/{filename}',
        dpi=800, bbox_inches='tight')
    plt.show()


def plot_boxplot(df, col_name, disease, filename, margin=0.5, custom_ylabel='Top30 biomarker edges number', roc_auc=0):
    # 1. 准备数据
    df = df.sort_values('group')
    # 2. 绘图初始化 (清理了原代码中重复的 figure 创建)
    plt.rc('font', size=20)
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['axes.linewidth'] = 2.0  # 全局设置边框粗细为 2.0
    fig, ax = plt.subplots(figsize=(5, 6), facecolor='white')
    # 颜色设置
    # custom_palette = ['#ed91bb', '#b6dafd']
    # custom_palette = ['#C9352B', '#339DB5']
    custom_palette = ['#E44A33', '#4DBAD6']
    # 3. 绘制箱线图
    sns.boxplot(
        x='group',
        y=col_name,
        data=df,
        hue='group',
        palette=custom_palette,
        order=['case', 'ctrl'],
        dodge=False,
        width=0.55,
        linewidth=2.0,
        showfliers=True,
        fliersize=4,
        flierprops={'marker': 'o', 'markerfacecolor': 'black', 'markeredgecolor': 'black', 'markersize': 2.5,
                    'alpha': 1},
        legend=False,
        boxprops={'edgecolor': 'black', 'linewidth': 2.0},  # 箱体边框纯黑
        whiskerprops={'color': 'black', 'linewidth': 2.0},  # 上下须纯黑
        medianprops={'color': 'black', 'linewidth': 2.0},  # 中位线纯黑
        capprops={'color': 'black', 'linewidth': 2.0},
        ax=ax
    )
    ax.margins(x=0.25)

    # 4. 提取数据用于统计
    case_biomarker_len = df[df['group'] == 'case'][col_name]
    ctrl_biomarker_len = df[df['group'] == 'ctrl'][col_name]

    ymin, ymax = ax.get_ylim()
    y_span = ymax - ymin  # 获取当前 Y 轴的视觉跨度

    # 获取数据的真实最大值，作为画 P 值横线的起跳点
    y_data_max = df[col_name].max()

    # P 值线条的高度：在最顶端的数据点上方，加上整体跨度的 5%
    y_pos = y_data_max + y_span * 0.05

    # 重新设置 Y 轴范围：下限保持默认不动，上限在 P 值线条基础上再留出 10% 空间给文字
    ax.set_ylim(ymin, y_pos + y_span * 0.1)

    # # --- 核心修改：动态计算 Y 轴范围和刻度 ---
    # # 获取数据的真实最小和最大值
    # y_data_min = df[col_name].min()
    # y_data_max = df[col_name].max()
    #
    # # 计算显著性标注线的位置 (兼容最大值为负数的情况)
    # y_pos = y_data_max + abs(y_data_max - y_data_min) * 0.08
    #
    # # 巧妙利用乘除 2 和向下/上取整，把范围吸附到 0.5 的倍数上
    # # 比如: -0.2 -> -0.5,  1.8 -> 2.0
    # tick_min = np.floor(y_data_min * 2) / 2
    # # 上限要考虑到 y_pos 的高度，再稍微多给 0.2 的留白以免贴边
    # tick_max = np.ceil((y_pos + 0.3) * 2) / 2
    #
    # # 生成 0.5 间隔的刻度序列
    # y_ticks = np.arange(tick_min, tick_max, margin)
    #
    # ax.set_ylim(tick_min - 0.2, tick_max)
    # ax.set_yticks(y_ticks)
    # ----------------------------------------

    # 5. 计算统计显著性
    u_stat, p_value = stats.mannwhitneyu(case_biomarker_len, ctrl_biomarker_len)
    ks_stat, ks_p = stats.ks_2samp(case_biomarker_len, ctrl_biomarker_len)
    delta, effect_size = calculate_cliffs_delta(case_biomarker_len, ctrl_biomarker_len)

    # --- 核心修改：极其微小 P 值的格式化防御 ---
    # 如果 P 值小到触碰浮点数底线变成 0.0，我们就给它一个非常拉风的极小值标识
    if p_value == 0.0:
        p_display = "P < 2.22e-308"
        p_print = "< 2.22e-308"
    else:
        p_display = f"P = {p_value:.2e}"
        p_print = f"{p_value:.2e}"  # 打印时也建议用科学计数法，原来的 .4f 遇到 0.00001 会变成 0.0000
    # ----------------------------------------

    # 6. 添加显著性标注
    # 绘制显著性线 (高度稍作调整适应新的 y_pos 逻辑)
    plt.plot([0, 0, 1, 1], [y_pos * 0.98, y_pos, y_pos, y_pos * 0.98], color='black', lw=1.5)
    # 标注 P 值文本 (应用新的字符串格式)
    plt.text(0.5, y_pos, p_display, ha='center', va='bottom', fontsize=18)
    # 7. 添加样本量信息到x轴标签
    new_labels = [f'{disease}\n(n={len(case_biomarker_len)})',
                  f'Healthy\n(n={len(ctrl_biomarker_len)})']
    ax.set_xticklabels(new_labels, fontsize=20)
    # 8. 设置标题和标签
    # plt.title(f'AUC = {roc_auc:.3f}', fontsize=22, pad=15)
    # plt.text(0.5, -1, f'AUC = {roc_auc:.3f}', fontsize=18, ha='center', va='bottom')
    plt.ylabel(custom_ylabel)
    plt.xlabel('')
    plt.grid(True, alpha=0.3, linestyle='--', axis='y')
    plt.tight_layout()
    # 保存图片 (取消注释以使用)
    plt.savefig(f'/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/figures/finetune/11.25/networks/4.1/{filename}', dpi=800, bbox_inches='tight')
    plt.show()
    # 9. 打印详细统计结果
    print("=" * 50)
    print("统计检验结果：")
    print(f"Mann-Whitney U检验: U = {u_stat:,.0f}, p = {p_value:.4f}")
    print(f"KS检验: D = {ks_stat:.4f}, p = {ks_p:.4f}")
    print(f"Cliff's Delta: {delta:.4f}, Effect Size: {effect_size}")
    print(
        f"Case组: 中位数 = {np.median(case_biomarker_len):.1f}, 均值 = {np.mean(case_biomarker_len):.1f} ± {np.std(case_biomarker_len):.1f}")
    print(
        f"Ctrl组: 中位数 = {np.median(ctrl_biomarker_len):.1f}, 均值 = {np.mean(ctrl_biomarker_len):.1f} ± {np.std(ctrl_biomarker_len):.1f}")
    print("=" * 50)


def calculate_cliffs_delta(case, ctrl):
    """
    手动计算 Cliff's Delta (无需外部库)
    公式: (count(x > y) - count(x < y)) / (n1 * n2)
    """
    n1, n2 = len(case), len(ctrl)

    # 将 list 转为 numpy 数组提高速度
    case = np.array(case)
    ctrl = np.array(ctrl)

    # 矩阵广播计算 (适用于样本量在几千以内的量级)
    # diff 会生成一个 (n1, n2) 的矩阵
    diff = case[:, None] - ctrl

    greater = np.sum(diff > 0)
    less = np.sum(diff < 0)

    delta = (greater - less) / (n1 * n2)

    # 定性判断
    if abs(delta) < 0.147:
        res = "Negligible"
    elif abs(delta) < 0.33:
        res = "Small"
    elif abs(delta) < 0.474:
        res = "Medium"
    else:
        res = "Large"

    return delta, res


def plot_mndi_results(df, col_name, roc_auc, threshold=None):
    plt.figure(figsize=(8, 10))  # 调高比例，更适合展示纵向差异
    plt.style.use('default')
    # plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['pdf.fonttype'] = 42
    plt.rc('font', size=22)
    # --- 改进 1: 确定 Y 轴显示范围 ---
    # 计算 99 分位数，防止极个别离群点拉飞坐标轴
    y_limit = df[col_name].quantile(0.99)
    y_min = df[col_name].min()
    # --- 改进 2: 绘制箱线图并去掉默认离群点符号 ---
    # showfliers=False 会隐藏那些黑色的空心小圆点，因为后面有 stripplot 展示原始点
    ax = sns.boxplot(
        x='group', y=col_name, data=df,
        palette=['#C9352B', '#339DB5'],
        width=0.5,
        showfliers=False,  # 关键：不显示超长离群点
        linewidth=2
    )
    # --- 改进 3: 限制散点图的显示范围 ---
    # 这样极端点不会消失，只是不会把图撑扁
    sns.stripplot(
        x='group', y=col_name, data=df,
        color='black', alpha=0.2, s=5, jitter=0.2
    )
    # 3. 统计显著性
    case_val = df[df['group'] == 'case'][col_name]
    ctrl_val = df[df['group'] == 'ctrl'][col_name]
    u_stat, p_val = stats.mannwhitneyu(case_val, ctrl_val)

    line_y = y_limit * 1.05
    h = y_limit * 0.03  # 竖杠的高度
    plt.plot([0, 0, 1, 1], [line_y, line_y + h, line_y + h, line_y],
             color='black', lw=1.5)

    # 根据 p 值确定星号
    if p_val < 0.001:
        stars = '***'
    elif p_val < 0.01:
        stars = '**'
    elif p_val < 0.05:
        stars = '*'
    else:
        stars = 'ns'

    # 在横线上方标注星号和具体的 P 值
    plt.text(0.5, line_y + h, f"{stars}\n(P = {p_val:.2e})",
             ha='center', va='bottom', fontsize=20, fontweight='bold')

    # --- 改进 4: 优化标题和布局 ---
    plt.title(f'Network Dysbiosis Index\nAUC = {roc_auc:.3f}', fontsize=22, pad=15)
    plt.ylabel(f'{col_name} Score')
    plt.xlabel('')
    # 限制 Y 轴，留出 10% 的空白放显著性标注
    # plt.ylim(y_min - 1, y_limit * 1.2)
    plt.ylim(y_min - 0.1, y_limit * 1.2)
    # 绘制阈值线
    if threshold:
        plt.axhline(y=threshold, color='red', linestyle='--', lw=2, label=f'Threshold: {threshold:.2f}')
        plt.legend(frameon=False, loc='upper left', fontsize='19')
    # # 标注 P 值 (位置随 Y 轴动态调整)
    # plt.text(0.5, y_limit * 1.1, f'P = {p_val:.2e}',
    #          ha='center', va='bottom', fontsize=18, fontweight='bold')
    # 细节美化
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    sns.despine(left=False, bottom=False, right=False, top=False)
    plt.tight_layout()
    # plt.savefig(f'/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/figures/finetune/11.25/networks/crc.case.ctrl.{col_name}.edge.number.pdf')
    plt.show()
    return u_stat, p_val


def plot_violinplot(df, col_name, filename, custom_ylabel='Top30 biomarker edges number', roc_auc=0):
    df = df.sort_values('group')
    # y_name = 'Edge number'
    # 1. 准备数据为长格式（Seaborn小提琴图需要）
    # df = pd.DataFrame({
    #     y_name: np.concatenate([case_biomarker_len, ctrl_biomarker_len]),
    #     'group': ['case'] * len(case_biomarker_len) + ['ctrl'] * len(ctrl_biomarker_len)
    # })
    # df = pd.DataFrame({
    #     'group': ['case'] * len(case_degree_list) + ['ctrl'] * len(ctrl_degree_list),
    #     'Degree': case_degree_list + ctrl_degree_list
    # })

    # 2. 绘制小提琴图
    plt.figure(figsize=(8, 8))
    # plt.rcParams['font.family'] = 'Arial'
    plt.rc('font', size=24)
    plt.rcParams['pdf.fonttype'] = 42
    plt.figure(facecolor='white')

    fig, ax = plt.subplots(figsize=(8, 8))
    custom_palette = sns.color_palette(['#C9352B', '#339DB5'])
    ax = sns.violinplot(
        x='group',
        y=col_name,
        data=df,
        palette=custom_palette,
        inner=None,
        linewidth=0,
        alpha=1,
        zorder=1,
        cut=0,
        ax=ax,
        order=['case', 'ctrl']
    )
    n_groups = len(df['group'].unique())
    for i, color in enumerate(custom_palette[:n_groups]):
        # 获取当前组的数据
        group_data = df[df['group'] == df['group'].unique()[i]][col_name]
        # 在该组位置绘制箱线图
        sns.boxplot(
            x=[df['group'].unique()[i]] * len(group_data),  # x坐标固定为该组位置
            y=group_data,
            width=0.2,
            showcaps=False,
            boxprops=dict(facecolor='white', linewidth=2.2, color='white'),
            whiskerprops=dict(linewidth=2.2, color='white'),
            showfliers=False,
            zorder=3,
            ax=ax,
            order=['case', 'ctrl'],
            medianprops=dict(linewidth=3, color=color),  # 使用对应的小提琴颜色
        )
    # max_degree = df[y_name].max()
    # max_y = np.ceil(max_degree / 50) * 50
    # ax.set_ylim(0, max_y)
    # my_y_ticks = np.arange(0, max_y + 1, 50)
    # ax.set_ylabel(y_name, labelpad=10)

    case_biomarker_len = df[df['group'] == 'case'][col_name]
    ctrl_biomarker_len = df[df['group'] == 'ctrl'][col_name]
    # 4. 计算统计显著性
    # 执行曼-惠特尼U检验（非参数，适合比较分布）
    u_stat, p_value = stats.mannwhitneyu(case_biomarker_len, ctrl_biomarker_len)
    # 也可执行KS检验比较整体分布
    ks_stat, ks_p = stats.ks_2samp(case_biomarker_len, ctrl_biomarker_len)
    delta, effect_size = calculate_cliffs_delta(case_biomarker_len, ctrl_biomarker_len)
    # 5. 添加显著性标注
    # 确定标注位置
    y_max = max(max(case_biomarker_len), max(ctrl_biomarker_len))
    y_pos = y_max * 1.08  # 在最高点上方8%处
    # 绘制显著性线和星号
    plt.plot([0, 0, 1, 1], [y_pos * 0.97, y_pos, y_pos, y_pos * 0.97],
             color='black', lw=1.5)
    # 根据p值确定星号数量
    if p_value < 0.001:
        stars = '***'
    elif p_value < 0.01:
        stars = '**'
    elif p_value < 0.05:
        stars = '*'
    else:
        stars = 'ns'
    # plt.text(0.5, y_pos * 0.99, stars,
    #          ha='center', va='bottom', fontsize=18, fontweight='bold')
    plt.text(0.5, y_pos * 0.995, f"P = {p_value:.2e}",
             ha='center', va='bottom', fontsize=18)
    # 6. 添加统计信息文本框
    stats_text = f'Mann-Whitney U test:\nU = {u_stat:,.0f}, p = {p_value:.2e}'
    # if p_value < 0.001:
    #     stats_text += ' ***'
    # elif p_value < 0.01:
    #     stats_text += ' **'
    # elif p_value < 0.05:
    #     stats_text += ' *'
    # plt.text(0.02, 0.98, stats_text, transform=ax.transAxes,
    #          verticalalignment='top', fontsize=12)
    #          # bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    # 7. 添加样本量信息到x轴标签
    new_labels = [f'case\n(n={len(case_biomarker_len)})',
                  f'ctrl\n(n={len(ctrl_biomarker_len)})']
    ax.set_xticklabels(new_labels, fontsize=24)
    # 8. 设置标题和标签
    plt.title(f'Network Dysbiosis Index\nAUC = {roc_auc:.3f}', fontsize=22, pad=15)
    plt.ylabel(custom_ylabel)
    plt.xlabel('')
    plt.grid(True, alpha=0.3, linestyle='--', axis='y')
    plt.tight_layout()
    # plt.savefig(f'/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/figures/finetune/11.25/networks/3.9/{filename}',
    #             dpi=800)
    plt.show()
    # 9. 打印详细统计结果
    print("=" * 50)
    print("统计检验结果：")
    print(f"Mann-Whitney U检验: U = {u_stat:,.0f}, p = {p_value:.4f}")
    print(f"KS检验: D = {ks_stat:.4f}, p = {ks_p:.4f}")
    print(f"Cliff's Delta: {delta:.4f}, Effect Size: {effect_size}")
    print(
        f"Case组: 中位数 = {np.median(case_biomarker_len):.1f}, 均值 = {np.mean(case_biomarker_len):.1f} ± {np.std(case_biomarker_len):.1f}")
    print(
        f"Ctrl组: 中位数 = {np.median(ctrl_biomarker_len):.1f}, 均值 = {np.mean(ctrl_biomarker_len):.1f} ± {np.std(ctrl_biomarker_len):.1f}")
    print("=" * 50)



#    CRC
dir='/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results/network/1.8/'
crc_profile = pd.read_csv(dir+'crc.all.profile.clean.csv', index_col=0)
sample_list = list(crc_profile.index)

label_df = pd.read_csv(
    '/bgi-seq-model-2/datasets/zhangkexin/meta_index/metaphlan4/fine-tune/curated_LiS_CRC_20204_v0908.train_test.phe',
    sep='\t', index_col=0)
case_samples = []
ctrl_samples = []
for i in sample_list:
    if label_df.loc[i, 'disease_united'] == 'CRC':
        case_samples.append(i)
    elif label_df.loc[i, 'disease_united'] == 'healthy':
        ctrl_samples.append(i)

dir='/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results/network/1.8/'
individual_network = np.load(dir+'crc.pretrain.emb.cosine.individual.npy')
taxa2id = pickle.load(open(dir+'crc.pretrain.emb.cosine.individual.index.pkl','rb'))

crc_biomarker = pd.read_csv('/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results/features/11.25/sort.multidisease.mean/CRC.biomarker.txt', sep='\t', header=None)
crc_biomarker['weight.abs'] = crc_biomarker[1].abs()
crc_biomarker = crc_biomarker.sort_values('weight.abs', ascending=False)
top_biomarker = list(crc_biomarker[:30][0])
# crc_good_biomarker = list(crc_biomarker.iloc[:30][crc_biomarker.iloc[:30][1] < 0][0])
crc_good_biomarker = list(crc_biomarker[crc_biomarker[1] < 0].iloc[:, 0])
crc_bad_biomarker = list(crc_biomarker[crc_biomarker[1] > 0].iloc[:, 0])

# --- 执行计算 ---
CRC_df_features = calculate_mndi_v2(individual_network, sample_list, taxa2id,
                         case_samples, ctrl_samples,
                         crc_bad_biomarker, crc_good_biomarker)


# # 3. 确定阈值 (Youden's Index)
# # 将 case 标为 1，ctrl 标为 0
CRC_y_true = (CRC_df_features['group'] == 'case').astype(int)
print("CRC MNDI 中 NaN 的数量:", CRC_df_features['MNDI'].isna().sum())
CRC_fpr, CRC_tpr, thresholds = roc_curve(CRC_y_true, CRC_df_features['MNDI'])
CRC_roc_auc = auc(CRC_fpr, CRC_tpr)
print(f"AUC: {CRC_roc_auc:.4f}")

# # 寻找最佳阈值：J = Sensitivity + Specificity - 1 最大值
optimal_idx = np.argmax(CRC_tpr - CRC_fpr)
optimal_threshold = thresholds[optimal_idx]

print(f"Optimal MNDI Threshold: {optimal_threshold:.4f}")

y_pred = (CRC_df_features['MNDI'] >= optimal_threshold).astype(int)
# 2. 计算 Balanced Accuracy
# 公式等价于：(Sensitivity + Specificity) / 2
bacc = balanced_accuracy_score(CRC_y_true, y_pred)
# 3. 为了让你看清具体的分类情况，建议同时打印混淆矩阵
tn, fp, fn, tp = confusion_matrix(CRC_y_true, y_pred).ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
acc = (tp+tn)/(tn+fp+fn+tp)
print(f"--- 分类性能评估 ---")
print(f"Accuracy: {acc:.4f}")
print(f"Balanced Accuracy: {bacc:.4f}")
print(f"Sensitivity (Sens): {sensitivity:.4f}")
print(f"Specificity (Spec): {specificity:.4f}")
print(f"Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")

plot_boxplot(CRC_df_features, 'MNDI', 'CRC', f'abs/CRC.index.pdf', 0.5, custom_ylabel='MNDI Score', roc_auc=0)

top30_crc_good_biomarker = list(crc_biomarker[crc_biomarker[1] < 0].iloc[:30, 0])
top30_crc_bad_biomarker = list(crc_biomarker[crc_biomarker[1] > 0].iloc[:30, 0])

CRC_top30_df_features = calculate_mndi_v2(individual_network, sample_list, taxa2id,
                         case_samples, ctrl_samples,
                         top30_crc_bad_biomarker, top30_crc_good_biomarker)
plot_grouped_boxplot(CRC_top30_df_features, 'CRC', 'abs/CRC.top30.biomarker.edge.weight.pdf', margin=100, custom_ylabel='Edge Weight', roc_auc=0)



# crc P验证队列
print('CRC   PRJNA758208')
PRJNA758208_label = pd.read_csv('/bgi-seq-model-2/datasets/zhangkexin/meta_index/metaphlan4/validate/7.25/CRC/PRJNA758208/PRJNA758208.info', sep='\t', index_col=0)
case_samples = list(PRJNA758208_label.loc[PRJNA758208_label['Group'] == 'Disease'].index)
ctrl_samples = list(PRJNA758208_label.loc[PRJNA758208_label['Group'] == 'Health'].index)

dir = '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results/network/1.8/crc.PRJNA758208/'
sample_list = []
with open(dir + 'samples.list.txt', 'r') as f:
    for line in f.readlines():
        sample_list.append(line.strip())
individual_network = np.load(dir+'pretrain.emb.individual.npy')
taxa2id = pickle.load(open(dir+'pretrain.emb.individual.index.pkl','rb'))

df_features = calculate_mndi_v2(individual_network, sample_list, taxa2id,
                         case_samples, ctrl_samples,
                         crc_bad_biomarker, crc_good_biomarker)


# # 3. 确定阈值 (Youden's Index)
# # 将 case 标为 1，ctrl 标为 0
y_true = (df_features['group'] == 'case').astype(int)
print("CRC VAL MNDI 中 NaN 的数量:", df_features['MNDI'].isna().sum())
fpr, tpr, thresholds = roc_curve(y_true, df_features['MNDI'])
roc_auc = auc(fpr, tpr)
print(f"AUC: {roc_auc:.4f}")

y_pred = (df_features['MNDI'] >= optimal_threshold).astype(int)
# 2. 计算 Balanced Accuracy
# 公式等价于：(Sensitivity + Specificity) / 2
bacc = balanced_accuracy_score(y_true, y_pred)
# 3. 为了让你看清具体的分类情况，建议同时打印混淆矩阵
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
acc = (tp+tn)/(tn+fp+fn+tp)
print(f"--- 分类性能评估 ---")
print(f"Balanced Accuracy: {bacc:.4f}")
print(f"Accuracy: {acc:.4f}")
print(f"Sensitivity (Sens): {sensitivity:.4f}")
print(f"Specificity (Spec): {specificity:.4f}")
print(f"Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
plot_boxplot(df_features, 'MNDI', 'CRC', f'abs/crc.PRJNA758208.index.pdf', 0.5, custom_ylabel='MNDI Score', roc_auc=roc_auc)



# # crc owen验证队列
# print('CRC   owen')
#
# dir = '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results/network/1.8/'
# sample_list = []
# with open(dir + 'crc.owen.samples.txt', 'r') as f:
#     for line in f.readlines():
#         sample_list.append(line.strip())
# individual_network = np.load(dir+'crc.owen.emb.individual.npy')
# taxa2id = pickle.load(open(dir+'crc.owen.emb.individual.index.pkl','rb'))
#
# PRJNA758208_label = pd.read_csv('/bgi-seq-model-2/datasets/zhangkexin/meta_index/metaphlan4/validate/7.25/CRC/owen/CRC_colon_phenotype_Qin.txt', sep='\t', index_col=0)
# PRJNA758208_label = PRJNA758208_label.loc[sample_list, :]
# case_samples = sample_list
#
# # df_features = calculate_mndi_with_ecdf(individual_network, sample_list, taxa2id,
# #                          case_samples, ctrl_samples,
# #                          crc_bad_biomarker, crc_good_biomarker, trained_ecdfs=discovery_ecdfs)
# df_features = calculate_mndi_v2(individual_network, sample_list, taxa2id,
#                          case_samples, sample_list,
#                          crc_bad_biomarker, crc_good_biomarker)
# # df_features['MNDI'], _, _ = calculate_advanced_mndi(df_features,ref_med, ref_mad)
# # df_features['MNDI'],_ = calculate_simple_mndi(df_features,my_ref)
# # df_features['MNDI'] = calculate_ndp_production(df_features, my_ref)
# # robust_auc, mndi_values, labels, optimal_threshold, df_mndi = evaluate_mndi_robust(df_features)
# # print(f"Realistic Cross-Validated AUC: {robust_auc:.4f}")
#
#
# # # 3. 确定阈值 (Youden's Index)
# # # 将 case 标为 1，ctrl 标为 0
# y_true = (df_features['group'] == 'case').astype(int)
# fpr, tpr, thresholds = roc_curve(y_true, df_features['MNDI'])
# roc_auc = auc(fpr, tpr)
# print(f"AUC: {roc_auc:.4f}")
#
# y_pred = (df_features['MNDI'] >= optimal_threshold).astype(int)
# # 2. 计算 Balanced Accuracy
# # 公式等价于：(Sensitivity + Specificity) / 2
# bacc = balanced_accuracy_score(y_true, y_pred)
# # 3. 为了让你看清具体的分类情况，建议同时打印混淆矩阵
# tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
# sensitivity = tp / (tp + fn)
# specificity = tn / (tn + fp)
# acc = (tp+tn)/(tn+fp+fn+tp)
# print(f"--- 分类性能评估 ---")
# print(f"Balanced Accuracy: {bacc:.4f}")
# print(f"Accuracy: {acc:.4f}")
# print(f"Sensitivity (Sens): {sensitivity:.4f}")
# print(f"Specificity (Spec): {specificity:.4f}")
# print(f"Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
#
#
#
# label_df=PRJNA758208_label.copy()
# label_df['sample_id']=list(PRJNA758208_label.index)
# plot_df = pd.merge(df_features, label_df[['sample_id', 'cStage']], on='sample_id', how='left')
# # --- 2. 预处理：排序与清洗 ---
# # 临床分期通常是类别型，建议手动指定顺序（如 0, I, II, III, IV），否则绘图会乱序
# # 请根据你数据里的实际分期名称修改这个列表
# stage_order = [1, 2, 3, 4]
# plot_df['cstage'] = pd.Categorical(plot_df['cStage'], categories=stage_order, ordered=True)
# # --- 3. 绘制箱线图 ---
# plt.figure(figsize=(8, 6))
# plt.rc('font', size=26)
# plt.rcParams['font.family'] = 'Arial'
# plt.rcParams['pdf.fonttype'] = 42
# # sns.set_style("whitegrid") # 设置清爽背景
# # 使用 boxenplot 或 boxplot，同时叠加 stripplot (打点) 可以看到样本分布情况
# ax = sns.boxplot(data=plot_df, x='cstage', y='MNDI', palette='Set3', order=stage_order, width=0.6, showfliers=False)
# # sns.stripplot(data=plot_df, x='cstage', y='MNDI', color='black', alpha=0.3, jitter=True)
# # --- 4. 统计装饰 ---
# # plt.title('MNDI Distribution across Clinical Stages')
# plt.xlabel('Clinical Stage (cstage)')
# plt.ylabel('MNDI Score')
# # 如果你想看每个阶段的人数，可以在 X 轴刻度上标注
# n_obs = plot_df['cstage'].value_counts().reindex(stage_order)
# ax.set_xticklabels([f'{l}\n(n={n})' for l, n in zip(stage_order, n_obs)])
# plt.tight_layout()
# # plt.savefig('/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/figures/finetune/11.25/networks/3.9/crc.owen.stage.mndi.1.pdf', dpi=800, bbox_inches='tight')
# plt.show()




# IBD
print('--------IBD')
dir = '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results/network/1.8/ibd/'
individual_network = np.load(dir + 'ibd.pretrain.emb.individual.npy')
taxa2id = pickle.load(open(dir + 'ibd.pretrain.emb.individual.index.pkl', 'rb'))

ibd_biomarker = pd.read_csv(
    '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results/features/11.25/sort.multidisease.mean/IBD.biomarker.txt',
    sep='\t', header=None)
ibd_biomarker['weight.abs'] = ibd_biomarker[1].abs()
ibd_biomarker = ibd_biomarker.sort_values('weight.abs', ascending=False)
top_biomarker = list(ibd_biomarker[:30][0])  # 30
ibd_good_biomarker = list(ibd_biomarker[ibd_biomarker[1] < 0].iloc[:, 0])
ibd_bad_biomarker = list(ibd_biomarker[ibd_biomarker[1] > 0].iloc[:, 0])

sample_list = []
with open(dir + 'ibd.pretrain.emb.samples.list.txt', 'r') as f:
    for line in f.readlines():
        sample_list.append(line.strip())
label_df = pd.read_csv(
    '/bgi-seq-model-2/datasets/zhangkexin/meta_index/metaphlan4/fine-tune/curated_LiS_CRC_20204_v0908.train_test.phe',
    sep='\t', index_col=0)
case_samples = []
ctrl_samples = []
for i in sample_list:
    if label_df.loc[i, 'disease_united'] == 'IBD':
        case_samples.append(i)
    elif label_df.loc[i, 'disease_united'] == 'healthy':
        ctrl_samples.append(i)


IBD_df_features = calculate_mndi_v2(individual_network, sample_list, taxa2id,
                         case_samples, ctrl_samples,
                         ibd_bad_biomarker, ibd_good_biomarker)


# # 3. 确定阈值 (Youden's Index)
# # 将 case 标为 1，ctrl 标为 0
IBD_y_true = (IBD_df_features['group'] == 'case').astype(int)

# 去除有nan的样本
print("IBD MNDI 中 NaN 的数量:", IBD_df_features['MNDI'].isna().sum())
valid_mask = ~IBD_df_features['MNDI'].isna()
IBD_df_features = IBD_df_features[valid_mask]
IBD_y_true = IBD_y_true[valid_mask]

IBD_fpr, IBD_tpr, thresholds = roc_curve(IBD_y_true, IBD_df_features['MNDI'])
IBD_roc_auc = auc(IBD_fpr, IBD_tpr)
print(f"AUC: {IBD_roc_auc:.4f}")

# # 寻找最佳阈值：J = Sensitivity + Specificity - 1 最大值
optimal_idx = np.argmax(IBD_tpr - IBD_fpr)
optimal_threshold = thresholds[optimal_idx]

print(f"Optimal MNDI Threshold: {optimal_threshold:.4f}")

y_pred = (IBD_df_features['MNDI'] >= optimal_threshold).astype(int)
# 2. 计算 Balanced Accuracy
# 公式等价于：(Sensitivity + Specificity) / 2
bacc = balanced_accuracy_score(IBD_y_true, y_pred)
# 3. 为了让你看清具体的分类情况，建议同时打印混淆矩阵
tn, fp, fn, tp = confusion_matrix(IBD_y_true, y_pred).ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
acc = (tp+tn)/(tn+fp+fn+tp)
print(f"--- 分类性能评估 ---")
print(f"Accuracy: {acc:.4f}")
print(f"Balanced Accuracy: {bacc:.4f}")
print(f"Sensitivity (Sens): {sensitivity:.4f}")
print(f"Specificity (Spec): {specificity:.4f}")
print(f"Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")

plot_boxplot(IBD_df_features, 'MNDI', 'IBD', f'abs/IBD.index.pdf', 2, custom_ylabel='MNDI Score', roc_auc=0)

top30_ibd_good_biomarker = list(ibd_biomarker[ibd_biomarker[1] < 0].iloc[:30, 0])
top30_ibd_bad_biomarker = list(ibd_biomarker[ibd_biomarker[1] > 0].iloc[:30, 0])
IBD_top30_df_features = calculate_mndi_v2(individual_network, sample_list, taxa2id,
                         case_samples, ctrl_samples,
                         top30_ibd_bad_biomarker, top30_ibd_good_biomarker)
plot_grouped_boxplot(IBD_top30_df_features, 'IBD', 'abs/IBD.top30.biomarker.edge.weight.pdf', margin=100, custom_ylabel='Edge Weight', roc_auc=0)





## IBD ning验证队列
print('IBD    Ning')
individual_network = np.load(dir + 'Ning.pretrain.emb.individual.npy')
taxa2id = pickle.load(open(dir + 'Ning.pretrain.emb.individual.index.pkl', 'rb'))

sample_list = []
with open(dir + 'Ning.pretrain.emb.samples.list.txt', 'r') as f:
    for line in f.readlines():
        sample_list.append(line.strip())
ning_label = pd.read_csv(
    '/bgi-seq-model-2/datasets/zhangkexin/meta_index/metaphlan4/validate/7.25/IBD/13_Ning_2023/13_Ning_2023.info',
    sep='\t', index_col=0)
case_samples = list(ning_label.loc[ning_label['Group'] == 'Disease'].index)
ctrl_samples = list(ning_label.loc[ning_label['Group'] == 'Health'].index)

df_features = calculate_mndi_v2(individual_network, sample_list, taxa2id,
                         case_samples, ctrl_samples,
                         ibd_bad_biomarker, ibd_good_biomarker)

y_true = (df_features['group'] == 'case').astype(int)
print("IBD val MNDI 中 NaN 的数量:", df_features['MNDI'].isna().sum())
valid_mask = ~df_features['MNDI'].isna()
df_features = df_features[valid_mask]
y_true = y_true[valid_mask]

fpr, tpr, thresholds = roc_curve(y_true, df_features['MNDI'])
roc_auc = auc(fpr, tpr)
print(f"AUC: {roc_auc:.4f}")

y_pred = (df_features['MNDI'] >= optimal_threshold).astype(int)
# 2. 计算 Balanced Accuracy
# 公式等价于：(Sensitivity + Specificity) / 2
bacc = balanced_accuracy_score(y_true, y_pred)
# 3. 为了让你看清具体的分类情况，建议同时打印混淆矩阵
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
acc = (tp+tn)/(tn+fp+fn+tp)
print(f"--- 分类性能评估 ---")
print(f"Balanced Accuracy: {bacc:.4f}")
print(f"Accuracy: {acc:.4f}")
print(f"Sensitivity (Sens): {sensitivity:.4f}")
print(f"Specificity (Spec): {specificity:.4f}")
print(f"Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
plot_boxplot(df_features, 'MNDI', 'IBD', f'abs/IBD.ning.index.pdf', margin=2, custom_ylabel='MNDI Score', roc_auc=roc_auc)



# pandisease
print('--------PANDISEASE')
dir = '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results/network/1.8/pandisease/'
sample_list = list(pd.read_csv(dir + 'pandisease.split1.valtest.samples.csv')['samples'])
label_df = pd.read_csv(
    '/bgi-seq-model-2/datasets/zhangkexin/meta_index/metaphlan4/fine-tune/curated_LiS_CRC_20204_v0908.train_test.phe',
    sep='\t', index_col=0)
case_samples = []
ctrl_samples = []
for i in sample_list:
    if label_df.loc[i, 'disease_united'] != 'healthy':
        case_samples.append(i)
    elif label_df.loc[i, 'disease_united'] == 'healthy':
        ctrl_samples.append(i)

individual_network = np.load(dir + 'pandisease.split1.valtest.pretrain.emb.cosine.individual.npy')
taxa2id = pickle.load(open(dir + 'pandisease.split1.valtest.pretrain.emb.cosine.individual.index.pkl', 'rb'))
# top20_good_biomarker = ['s-' + i.lower().replace(" ", "-") for i in
#                         list(pd.read_csv(dir + 'top20.negative.o2.csv', index_col=0).index)]
# top20_bad_biomarker = ['s-' + i.lower().replace(" ", "-") for i in
#                        list(pd.read_csv(dir + 'top20.positive.o2.csv', index_col=0).index)]
top30_core_good_biomarker = ['s-' + i.lower().replace(" ", "-") for i in
                        list(pd.read_csv(dir + 'top30.negative.o3.csv', index_col=0).index)]
top30_core_bad_biomarker = ['s-' + i.lower().replace(" ", "-") for i in
                       list(pd.read_csv(dir + 'top30.positive.o3.csv', index_col=0).index)]

pan_biomarker = pd.read_csv(
    '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results/features/11.25/sort.multidisease.mean/pandisease.biomarker.txt',
    sep='\t', header=None)
pan_biomarker['weight.abs'] = pan_biomarker[1].abs()
pan_biomarker = pan_biomarker.sort_values('weight.abs', ascending=False)
good_biomarker = list(pan_biomarker[pan_biomarker[1] < 0].iloc[:, 0])
bad_biomarker = list(pan_biomarker[pan_biomarker[1] > 0].iloc[:, 0])

PAN_df_features = calculate_mndi_v2(individual_network, sample_list, taxa2id,
                                case_samples, ctrl_samples,
                                top30_core_bad_biomarker, top30_core_good_biomarker)
PAN_full_df_features = calculate_mndi_v2(individual_network, sample_list, taxa2id,
                                case_samples, ctrl_samples,
                                bad_biomarker, good_biomarker)


# # 3. 确定阈值 (Youden's Index)
# # 将 case 标为 1，ctrl 标为 0
PAN_y_true = (PAN_df_features['group'] == 'case').astype(int)

# 去除有nan的样本
print("pandisease core MNDI 中 NaN 的数量:", PAN_df_features['MNDI'].isna().sum())
valid_mask = ~PAN_df_features['MNDI'].isna()
PAN_df_features = PAN_df_features[valid_mask]
PAN_y_true = PAN_y_true[valid_mask]

PAN_fpr, PAN_tpr, thresholds = roc_curve(PAN_y_true, PAN_df_features['MNDI'])
PAN_roc_auc = auc(PAN_fpr, PAN_tpr)
print(f"AUC: {PAN_roc_auc:.4f}")

PAN_full_y_true = (PAN_full_df_features['group'] == 'case').astype(int)

# 去除有nan的样本
print("pandisease full MNDI 中 NaN 的数量:", PAN_full_df_features['MNDI'].isna().sum())
valid_mask = ~PAN_full_df_features['MNDI'].isna()
PAN_full_df_features = PAN_full_df_features[valid_mask]
PAN_y_true = PAN_y_true[valid_mask]

PAN_full_fpr, PAN_full_tpr, thresholds = roc_curve(PAN_full_y_true, PAN_full_df_features['MNDI'])
PAN_full_roc_auc = auc(PAN_full_fpr, PAN_full_tpr)
print(f"AUC: {PAN_full_roc_auc:.4f}")

# # 寻找最佳阈值：J = Sensitivity + Specificity - 1 最大值
optimal_idx = np.argmax(PAN_tpr - PAN_fpr)
optimal_threshold = thresholds[optimal_idx]

print(f"Optimal MNDI Threshold: {optimal_threshold:.4f}")

y_pred = (PAN_df_features['MNDI'] >= optimal_threshold).astype(int)
# 2. 计算 Balanced Accuracy
# 公式等价于：(Sensitivity + Specificity) / 2
bacc = balanced_accuracy_score(PAN_y_true, y_pred)
# 3. 为了让你看清具体的分类情况，建议同时打印混淆矩阵
tn, fp, fn, tp = confusion_matrix(PAN_y_true, y_pred).ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
acc = (tp + tn) / (tn + fp + fn + tp)
print(f"--- 分类性能评估 ---")
print(f"Accuracy: {acc:.4f}")
print(f"Balanced Accuracy: {bacc:.4f}")
print(f"Sensitivity (Sens): {sensitivity:.4f}")
print(f"Specificity (Spec): {specificity:.4f}")
print(f"Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
plot_boxplot(PAN_df_features, 'MNDI', 'Disease', f'abs/pandisease.core.index.v1.pdf', 2, custom_ylabel='MNDI Score', roc_auc=0)
plot_boxplot(PAN_full_df_features, 'MNDI', 'Disease', f'abs/pandisease.full.index.v1.pdf', 2, custom_ylabel='MNDI Score', roc_auc=0)


top30_pan_good_biomarker = list(pan_biomarker[pan_biomarker[1] < 0].iloc[:30, 0])
top30_pan_bad_biomarker = list(pan_biomarker[pan_biomarker[1] > 0].iloc[:30, 0])
PAN_full_top30_df_features = calculate_mndi_v2(individual_network, sample_list, taxa2id,
                         case_samples, ctrl_samples,
                         top30_pan_bad_biomarker, top30_pan_good_biomarker)
plot_grouped_boxplot(PAN_full_top30_df_features, 'Disease', 'abs/pandisease.top30.biomarker.edge.weight.pdf', margin=100, custom_ylabel='Edge Weight', roc_auc=0)
plot_grouped_boxplot(PAN_df_features, 'Disease', 'abs/pandisease.top30.core.biomarker.edge.weight.pdf', margin=100, custom_ylabel='Edge Weight', roc_auc=0)

#
# ## 泛病的验证队列，IMSMS
print('Pandisease IMSMS')
individual_network = np.load(
    '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results/network/1.8/iMSMS/pretrain.emb.individual.npy')
taxa2id = pickle.load(open(
    '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results/network/1.8/iMSMS/pretrain.emb.individual.index.pkl',
    'rb'))

sample_list = []
with open('/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results/network/1.8/iMSMS/samples.list.txt',
          'r') as f:
    for line in f.readlines():
        sample_list.append(line.strip())
ning_label = pd.read_csv(
    '/bgi-seq-model-2/datasets/zhangkexin/meta_index/metaphlan4/validate/iMSMS_2022.MS/iMSMS_2022.metadata.tsv',
    sep='\t', index_col=0)
case_samples = list(ning_label.loc[ning_label['study_condition'] == 'MS'].index)
ctrl_samples = list(ning_label.loc[ning_label['study_condition'] == 'control'].index)

df_features = calculate_mndi_v2(individual_network, sample_list, taxa2id,
                                case_samples, ctrl_samples,
                                bad_biomarker, good_biomarker)
# df_features['MNDI'] = calculate_ndp_production(df_features, my_ref)
# # 3. 确定阈值 (Youden's Index)
# # 将 case 标为 1，ctrl 标为 0
y_true = (df_features['group'] == 'case').astype(int)
fpr, tpr, thresholds = roc_curve(y_true, df_features['MNDI'])
roc_auc = auc(fpr, tpr)
print(f"AUC: {roc_auc:.4f}")

y_pred = (df_features['MNDI'] >= optimal_threshold).astype(int)
# 2. 计算 Balanced Accuracy
# 公式等价于：(Sensitivity + Specificity) / 2
bacc = balanced_accuracy_score(y_true, y_pred)
# 3. 为了让你看清具体的分类情况，建议同时打印混淆矩阵
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
acc = (tp + tn) / (tn + fp + fn + tp)
print(f"--- 分类性能评估 ---")
print(f"Balanced Accuracy: {bacc:.4f}")
print(f"Accuracy: {acc:.4f}")
print(f"Sensitivity (Sens): {sensitivity:.4f}")
print(f"Specificity (Spec): {specificity:.4f}")
print(f"Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")


#
# def extract_raw_features(individual_network, taxa2id, bad_taxa, good_taxa):
#     """只负责提取原始物理特征"""
#     bad_idx = [taxa2id[t] for t in bad_taxa if t in taxa2id]
#     good_idx = [taxa2id[t] for t in good_taxa if t in taxa2id]
#     features_list = []
#     for i in range(individual_network.shape[0]):
#         adj = np.abs(individual_network[i])
#         adj[np.isnan(adj)] = 0
#         np.fill_diagonal(adj, 0)
#         # 1. 核心物理量计算
#         total_weight = adj.sum() / 2
#         bad_internal = adj[np.ix_(bad_idx, bad_idx)].sum() / 2
#         good_internal = adj[np.ix_(good_idx, good_idx)].sum() / 2
#         bad_all = adj[bad_idx, :].sum()
#         good_all = adj[good_idx, :].sum()
#         features_list.append({
#             'f_bad_bad': bad_internal,
#             'f_bad_dominance': bad_all / (total_weight + 1e-5),
#             'f_good_good': good_internal,
#             'f_good_dominance': good_all / (total_weight + 1e-5),
#             'f_good_loss_raw': 1 / (good_all + 1e-5)
#         })
#     return pd.DataFrame(features_list)
# # --- 第一步：分别提取原始特征 ---
# case_raw = extract_raw_features(case_individual_network, taxa2id, ibd_bad_biomarker, ibd_good_biomarker)
# ctrl_raw = extract_raw_features(ctrl_individual_network, taxa2id, ibd_bad_biomarker, ibd_good_biomarker)
# case_raw['group'] = 'case'
# ctrl_raw['group'] = 'ctrl'
# # --- 第二步：合并后统一计算 Rank (关键点！) ---
# all_features = pd.concat([case_raw, ctrl_raw], ignore_index=True)
# # 定义参与 MNDI 计算的列
# calc_cols = ['f_bad_bad', 'f_bad_dominance', 'f_good_loss_raw']
# # 在全人群范围内计算百分比排名
# df_ranks = all_features[calc_cols].rank(pct=True)
# # --- 第三步：合成指标 (建议尝试对抗性逻辑，效应值更高) ---
# # 逻辑：(坏菌的强) + (好菌优势的缺失)
# # 如果想尝试你之前的加法：
# all_features['MNDI'] = df_ranks.sum(axis=1)
# # 如果想尝试效应值更高的“对抗逻辑”：
# # r_good_dom_inv = 1 - all_features['f_good_dominance'].rank(pct=True)
# # all_features['MNDI_v2'] = df_ranks['f_bad_bad'] + df_ranks['f_bad_dominance'] + r_good_dom_inv
# # --- 第四步：绘图 ---
# plot_violinplot(all_features, 'MNDI', 'ibd.popu.networl.index.unified_rank.pdf', 'MNDI')


# 画AUC

from scipy.stats import norm
# --- 你的 AUC 置信区间计算函数 ---
def AUC_CI(auc_val, label, alpha=0.05):
    label = np.array(label)
    n1, n2 = np.sum(label == 1), np.sum(label == 0)
    q1 = auc_val / (2 - auc_val)
    q2 = (2 * auc_val ** 2) / (1 + auc_val)
    se = np.sqrt(
        (auc_val * (1 - auc_val) + (n1 - 1) * (q1 - auc_val ** 2) + (n2 - 1) * (q2 - auc_val ** 2)) / (n1 * n2))
    confidence_level = 1 - alpha
    z_lower, z_upper = norm.interval(confidence_level)
    lowerb, upperb = auc_val + z_lower * se, auc_val + z_upper * se
    return (lowerb, upperb)

from sklearn.utils import resample

def plot_multiple_roc_with_ci(data_dict, filename, n_bootstraps=1000):
    """
    绘制多条带 Bootstrap 置信区间的 ROC 曲线

    参数:
    data_dict: dict, 格式为 {'疾病名称': 对应的 df_features 数据框}
    filename: str, 保存的 pdf 文件名
    n_bootstraps: int, Bootstrap 重采样的次数（1000次是科研标配）
    """
    # 1. 绘图初始化 (保持与箱线图一致的科研风格)
    plt.rc('font', size=20)
    plt.rcParams['pdf.fonttype'] = 42
    fig, ax = plt.subplots(figsize=(6, 6), facecolor='white')

    # 定义三种疾病的颜色 (红、蓝、黄/橘)
    # 你可以根据需要自己修改这里的颜色
    colors = ['#C9352B', '#339DB5', '#E69F00', '#009E73']
    # colors = ['#D9BDDB', '#9DD0C7', '#8AB1D2', '#E58579']

    # 生成一个用于插值的标准假阳性率序列 (0到1，100个点)
    mean_fpr = np.linspace(0, 1, 100)

    # 2. 遍历每个疾病进行计算和绘制
    for i, (disease, df) in enumerate(data_dict.items()):
        color = colors[i % len(colors)]

        # 提取真实标签和预测得分
        y_true = (df['group'] == 'case').astype(int).values
        y_score = df['MNDI'].values

        # 计算原始 AUC
        fpr_orig, tpr_orig, _ = roc_curve(y_true, y_score)
        roc_auc_orig = auc(fpr_orig, tpr_orig)

        # --- Bootstrap 计算置信区间 ---
        tprs_boot = []
        aucs_boot = []

        for _ in range(n_bootstraps):
            # 有放回地重采样
            indices = resample(np.arange(len(y_true)))

            # 确保重采样的样本中同时包含 case 和 ctrl，否则无法算 ROC
            if len(np.unique(y_true[indices])) < 2:
                continue

            fpr_b, tpr_b, _ = roc_curve(y_true[indices], y_score[indices])
            roc_auc_b = auc(fpr_b, tpr_b)

            # 将每次采样得到的 TPR 插值对齐到我们统一的 mean_fpr 上
            tpr_interp = np.interp(mean_fpr, fpr_b, tpr_b)
            tpr_interp[0] = 0.0  # 起点强制为0

            tprs_boot.append(tpr_interp)
            aucs_boot.append(roc_auc_b)

        # 计算 TPR 的 95% 置信区间边界 (2.5% 和 97.5% 分位数)
        tpr_lower = np.percentile(tprs_boot, 2.5, axis=0)
        tpr_upper = np.percentile(tprs_boot, 97.5, axis=0)
        # 强制终点为1
        tpr_upper[-1] = 1.0

        # 计算 AUC 的 95% 置信区间 (用于图例)
        auc_lower = np.percentile(aucs_boot, 2.5)
        auc_upper = np.percentile(aucs_boot, 97.5)

        # --- 绘制线条和阴影 ---
        # 绘制平均 ROC 曲线 (这里我们用平滑后的 mean_tpr 替代原线条以对齐阴影)
        mean_tpr = np.mean(tprs_boot, axis=0)
        mean_tpr[-1] = 1.0

        # 图例文字：疾病名 + AUC值及其置信区间
        label_text = f"{disease}\nAUC={roc_auc_orig:.3f} ({auc_lower:.3f}-{auc_upper:.3f})"

        ax.plot(mean_fpr, mean_tpr, color=color, linewidth=2.5, label=label_text, zorder=3)
        # 绘制置信区间阴影
        ax.fill_between(mean_fpr, tpr_lower, tpr_upper, color=color, alpha=0.15, zorder=2)

    # 3. 绘制对角虚线 (随机猜测线)
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray', alpha=0.8, zorder=1)

    # 4. 坐标轴和样式美化
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    # 设置刻度
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis='both', labelsize=20)

    # 边框粗细与箱线图保持一致
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)

    ax.set_xlabel('False Positive Rate', fontsize=20)
    ax.set_ylabel('True Positive Rate', fontsize=20)

    # 图例设置 (去掉边框，放右下角)
    ax.legend(loc="lower right", fontsize=14, frameon=False)

    plt.tight_layout()

    # 保存图片
    plt.savefig(f'/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/figures/finetune/11.25/networks/4.1/{filename}', dpi=800, bbox_inches='tight')
    plt.show()

# --- 绘图函数 ---
def plot_multiple_roc_analytic_ci(data_dict, filename):
    """
    绘制多条 ROC 曲线，并在图例中使用解析法标注 95% CI
    参数:
    data_dict: dict, 格式为 {'疾病名称': 对应的 df_features 数据框}
    filename: str, 保存的 pdf 文件名
    """
    # 1. 绘图初始化 (保持科研风格)
    plt.rc('font', size=20)
    plt.rcParams['pdf.fonttype'] = 42
    fig, ax = plt.subplots(figsize=(6, 6), facecolor='white')
    # 颜色设置：红、蓝、黄 (对应你之前的配色习惯)
    colors = ['#C9352B', '#339DB5', '#E69F00', '#009E73']
    # colors = ['#F1DFA4','#8AB1D2', '#E58579']
    # 2. 遍历每个疾病进行计算和绘制
    for i, (disease, df) in enumerate(data_dict.items()):
        color = colors[i % len(colors)]
        # 提取真实标签 (0和1) 和预测得分
        y_true = (df['group'] == 'case').astype(int).values
        y_score = df['MNDI'].values
        # 计算 FPR, TPR 和 AUC
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        # 🌟 调用你的函数计算 95% CI
        ci_lower, ci_upper = AUC_CI(roc_auc, y_true)
        # 格式化图例文字
        label_text = f"{disease}\nAUC={roc_auc:.3f} ({ci_lower:.3f}-{ci_upper:.3f})"
        # 绘制 ROC 曲线线条
        ax.plot(fpr, tpr, color=color, linewidth=2.5, label=label_text, zorder=3)
    # 3. 绘制对角虚线 (随机猜测线)
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray', alpha=0.8, zorder=1)
    # 4. 坐标轴和样式美化
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis='both', labelsize=20)
    # 边框粗细与箱线图保持一致 (2.0)
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)
    ax.set_xlabel('1-Specificity', fontsize=20)
    ax.set_ylabel('Sensitivity', fontsize=20)
    # 图例设置：去掉边框，放右下角
    ax.legend(loc="lower right", fontsize=14, frameon=False)
    plt.tight_layout()
    # 保存图片
    plt.savefig(f'/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/figures/finetune/11.25/networks/4.1/{filename}', dpi=800, bbox_inches='tight')
    plt.show()
# ----------------- 使用方法 -----------------
data_dict = {
    'CRC': CRC_df_features,
    'IBD': IBD_df_features,
    'Pandisease (Full)': PAN_full_df_features,
    'Pandisease (Core)': PAN_df_features
}
plot_multiple_roc_analytic_ci(data_dict, 'abs/combined_roc.pdf')


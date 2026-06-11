# @Date    : 2026/6/10 16:23
# @Email   : zhangkexin2@genomics.cn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def set_global_font():
    """设置全局通用的字体和输出格式"""
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['pdf.fonttype'] = 42


def plot_pan_contrast_bar(input_csv, output_pdf):
    """
    绘制泛病预测性能对比柱状图 (Bar Chart)
    """
    # 1. 读取数据
    df = pd.read_csv(input_csv)
    labels = ['Precision', 'Recall', 'F1', 'ACC', 'AUC', 'AUPR', 'MCC']
    data = df.values[:, 1:]
    name = list(df.values[:, 0])

    # 2. 样式设置
    plt.style.use('default')
    set_global_font()
    plt.rcParams.update({
        'font.size': 22,
        'axes.titlesize': 22,
        'axes.labelsize': 22,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 20,
    })

    fig = plt.figure(figsize=(11, 5))
    x = np.arange(len(labels))
    plt.grid(linestyle='-.', zorder=0, axis="y")
    width = 0.16

    # 3. 绘制柱状图
    colors = ['#F1DFA4', '#D9BDDB', '#9DD0C7', '#8AB1D2', '#E58579']
    offsets = [-2 * width, -1 * width, 0, 1 * width, 2 * width]

    for i in range(5):
        plt.bar(
            x + offsets[i], data[i, :], lw=0.5, fc=colors[i],
            width=width, label=name[i], zorder=100, edgecolor='k'
        )

    # 4. 图例与坐标轴美化
    plt.legend(
        bbox_to_anchor=(0.95, -0.08), ncol=5, loc=1, frameon=False,
        labelspacing=0.2, columnspacing=0.2, handletextpad=0.2,
        handlelength=1.6, fontsize=22
    )
    plt.ylim((0, 1))
    plt.yticks(np.arange(0, 1.02, 0.2), fontsize=20)
    plt.xticks(x, labels=labels, fontsize=20)
    plt.ylabel("Performance", fontsize=22)

    plt.gcf().subplots_adjust(bottom=0.15)
    # plt.savefig(output_pdf, dpi=800, bbox_inches='tight', format='pdf')
    plt.show()


def plot_multiclass_radar(input_csv, output_pdf):
    """
    绘制多分类性能对比雷达图 (Radar Chart)
    """
    # 1. 读取并排序数据
    df = pd.read_csv(input_csv, index_col=0)
    selected_cols = ['Healthy', 'IBD', 'CRC', 'T2D', 'MS', 'AS', 'OB', 'IBS',
                     'IGT', 'BL', 'ACVD', 'CKD', 'COVID-19', 'Adenoma', 'Melanoma']
    df = df.loc[:, selected_cols]

    # 按 UMetaGPT 表现从高到低排序
    sorted_columns = df.T.sort_values('UmetaGPT', ascending=False).index
    df = df.reindex(columns=sorted_columns)

    labels = list(df.columns)
    num_vars = len(labels)

    # 闭合角度
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    # 2. 样式设置
    set_global_font()
    plt.rc('font', size=20)
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    colors = ['#F1DFA4', '#D9BDDB', '#9DD0C7', '#8AB1D2', '#E58579']
    models = ['MLP', 'LR', 'RF', 'XGBoost', 'UmetaGPT']

    # 3. 绘制每个模型
    for i, model in enumerate(models):
        is_main = (model == 'UmetaGPT')
        values = df.loc[model].values.flatten().tolist()
        values += values[:1]

        ax.plot(
            angles, values, linewidth=3 if is_main else 1.8, linestyle='solid',
            label=model, color=colors[i], marker='o', markersize=9 if is_main else 7,
            markerfacecolor=colors[i], markeredgecolor='white', markeredgewidth=0.5
        )
        ax.fill(angles, values, color=colors[i], alpha=0.2)

    # 4. 坐标轴与网格美化
    grid_color, grid_width, grid_style = '#DDDDDD', 1.2, '--'
    ax.spines['polar'].set_color(grid_color)
    ax.spines['polar'].set_linewidth(grid_width)
    ax.spines['polar'].set_linestyle(grid_style)
    ax.grid(True, color=grid_color, linestyle=grid_style, linewidth=grid_width, alpha=1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=18)

    # 对齐 Adenoma
    adenoma_idx = labels.index('Adenoma')
    ax.set_rlabel_position(np.rad2deg(angles[adenoma_idx]))

    # 图例设置
    plt.legend(
        loc='center left', bbox_to_anchor=(-0.16, 1), frameon=False,
        handlelength=1.0, handleheight=0.8, handletextpad=0.3,
        borderpad=0.1, labelspacing=0.1
    )

    plt.subplots_adjust(left=0.1, right=0.75, top=0.9, bottom=0.1)
    # plt.savefig(output_pdf, dpi=800, bbox_inches='tight')
    plt.show()


def plot_independent_test_heatmap(output_pdf):
    """
    绘制独立测试集性能对比热力图 (Pcolormesh Heatmap)
    """
    # 1. 数据准备
    data_values_orig = np.array([
        [0.582088, 0.8479, 0.5199],
        [0.655286, 0.7823, 0.5223],
        [0.614848, 0.9239, 0.5325],
        [0.778990, 0.8543, 0.5469],
        [0.8452, 0.9020, 0.5782]
    ])

    models = ['MLP', 'LR', 'RF', 'XGBoost', 'UMetaGPT']
    datasets = ['PRJNA75820', 'Ning_2013', 'CNP0000175']
    diseases = ['CRC', 'IBD', 'T2D']

    # 行归一化处理 (用于决定颜色深度)
    data_values = data_values_orig.T
    data_normalized = data_values.astype('float') / data_values.sum(axis=1)[:, np.newaxis]
    data_normalized = np.nan_to_num(data_normalized)

    # 转置回 5x3 用于画图
    data_values = data_values.T
    data_normalized = data_normalized.T

    # 2. 样式设置
    set_global_font()
    plt.rc('font', size=22)
    fig = plt.figure(figsize=(5.5, 7), facecolor='white')
    ax = fig.add_subplot(111)

    # 3. 绘制热图
    cmap = sns.blend_palette(["#FFFFFF", "#9DD0C7"], as_cmap=True)
    im = ax.pcolormesh(data_normalized, cmap=cmap, edgecolors='white', linewidths=2)
    ax.invert_yaxis()  # 翻转匹配顺序
    ax.set_aspect('equal')

    # 4. 设置坐标轴
    ax.tick_params(left=True, labelleft=True, right=False, labelright=False, top=False, bottom=False)

    # 左侧 Y 轴 (模型名称)
    ax.set_yticks(np.arange(len(models)) + 0.5)
    ax.set_yticklabels(models)

    # 底部 X 轴 (疾病名称)
    ax.set_xticks(np.arange(len(diseases)) + 0.5)
    ax.set_xticklabels(diseases, rotation=90, ha="center")

    # 顶部 X 轴 (项目编号)
    ax_top = ax.secondary_xaxis('top')
    ax_top.set_xticks(np.arange(len(datasets)) + 0.5)
    ax_top.set_xticklabels(datasets, rotation=90, ha="center")
    ax_top.spines['top'].set_visible(False)

    # 5. 写入数值
    threshold = (data_normalized.max() + data_normalized.min()) / 2
    for i in range(len(models)):
        for j in range(len(diseases)):
            val_orig = data_values[i, j]
            text_color = "white" if data_normalized[i, j] > threshold else "black"
            ax.text(
                j + 0.5, i + 0.5, f"{val_orig:.3f}",
                ha="center", va="center", color="black", fontsize=18
            )

    # 6. 细节美化
    ax.spines[:].set_visible(False)
    ax.set_xticks(np.arange(len(diseases) + 1), minor=True)
    ax.set_yticks(np.arange(len(models) + 1), minor=True)
    ax.grid(which="minor", color="white", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", top=False, bottom=False, left=False, right=False)

    plt.tight_layout()
    # plt.savefig(output_pdf, bbox_inches='tight')
    plt.show()


# ==========================================
# 主执行入口
# ==========================================
if __name__ == '__main__':
    # 定义基础路径
    BASE_RESULTS_DIR = '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results/contrast/11.25'
    BASE_FIGURES_DIR = '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/figures/finetune/11.25/contrast'

    # 1. 绘制柱状图
    csv_bar = f'{BASE_RESULTS_DIR}/pan.contrast.csv'
    pdf_bar = f'{BASE_FIGURES_DIR}/pan.contrast.2.9.pdf'
    plot_pan_contrast_bar(csv_bar, pdf_bar)

    # 2. 绘制多分类雷达图
    csv_radar = f'{BASE_RESULTS_DIR}/multi.class.contrast.f1.csv'
    pdf_radar = f'{BASE_FIGURES_DIR}/multi.class.f1.3.24.pdf'
    plot_multiclass_radar(csv_radar, pdf_radar)

    # 3. 绘制热力图 (使用硬编码数据，只需输出路径)
    pdf_heatmap = f'{BASE_FIGURES_DIR}/indep.test.comparison.4.22.2.vertical.pdf'
    plot_independent_test_heatmap(pdf_heatmap)

# @Date    : 2026/6/11 10:37
# @Email   : zhangkexin2@genomics.cn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
# --- 1. 计算函数 (保持原样) ---
def calculate_recall(group_df, threshold=0.5):
    if len(group_df) == 0: return np.nan
    true_positives = pd.Series([True] * len(group_df), index=group_df.index)
    pred_positives = group_df['pred'] == 1
    tp = ((true_positives) & (pred_positives)).sum()
    fn = ((true_positives) & (~pred_positives)).sum()
    if (tp + fn) == 0: return np.nan
    return tp / (tp + fn)
def get_group_recall(subpheno_df, dir, filename, order, object, index_col):
    split_5_recall = {}
    for i in range(1, 6):
        split = 'split' + str(i)
        prob_df = pd.read_csv(os.path.join(dir, split, filename), index_col=index_col)
        prob_459_df = prob_df.loc[list(subpheno_df.index), :]
        prob_459_df[object] = subpheno_df[object]
        location_results = prob_459_df.groupby(object).apply(calculate_recall).reset_index()
        location_results.columns = ['Group', 'Recall']
        location_results['Group'] = pd.Categorical(location_results['Group'], categories=order, ordered=True)
        location_results_sorted = location_results.sort_values('Group')
        split_5_recall[split] = location_results_sorted['Recall'].values.tolist()
    split_5_recall_df = pd.DataFrame(dict(split_5_recall), index=order)
    split_5_recall_df['mean'] = split_5_recall_df.mean(1)
    group_counts = subpheno_df[object].value_counts()
    return split_5_recall_df, group_counts
def create_and_plot_heatmap_vertical(df, group_counts, object_name):
    # ------------------ 数据处理部分（保持不变） ------------------
    # 依然按原始行（groups）进行归一化，保证颜色深浅的统计意义不变
    data_values_orig = df.values
    data_normalized = data_values_orig.astype('float') / data_values_orig.sum(axis=1)[:, np.newaxis]
    data_normalized = np.nan_to_num(data_normalized)
    models = df.columns
    groups = df.index
    # 【修改 1】：转置用于画图的矩阵
    # 使模型 (models) 在行 (Y轴)，分组 (groups) 在列 (X轴)
    data_values_orig_T = data_values_orig.T
    data_normalized_T = data_normalized.T
    # 配置样式
    plt.rcParams['font.family'] = 'Arial'
    plt.rc('font', size=22)
    plt.rcParams['pdf.fonttype'] = 42
    # 【修改 2】：动态调整画布比例，宽高逻辑互换
    # 宽度由列数(groups)决定，高度由行数(models)决定
    fig = plt.figure(figsize=(len(groups) * 1, 5.5), facecolor='white')
    ax = fig.add_subplot(111)
    # 【修改 3】：绘制热图传入转置后的数据 data_normalized_T
    cmap = sns.blend_palette(["#FFFFFF", "#9DD0C7"], as_cmap=True)
    im = ax.imshow(data_normalized_T, cmap=cmap, aspect='equal')
    # ------------------ 设置坐标轴 ------------------
    # 【修改 4】：互换 X 轴和 Y 轴的标签逻辑
    # 底部显示 X 轴（Groups），顶部关闭
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    # 左侧显示 Y 轴（Models），右侧关闭
    ax.tick_params(left=False, labelleft=False, right=False, labelright=False)
    # 4.1 设置 X 轴（原本的 Y 轴逻辑移到这里）
    new_xticklabels = []
    for g in groups:
        if g == 'Overall':
            new_xticklabels.append('Overall')
        else:
            prefix = ""
            n_count = group_counts.get(g, 0)
            label = str(int(g)) if isinstance(g, float) else str(g)
            # 建议加一个换行符 \n 防止标签在底部挤在一起
            new_xticklabels.append(f"{prefix}{label} (n={n_count})")
    ax.set_xticks(np.arange(len(groups)))
    # X 轴标签如果是多行，建议旋转一下以防重叠
    ax.set_xticklabels(new_xticklabels, rotation=90, ha='center')
    # 4.2 设置 Y 轴
    # ax.set_yticks(np.arange(len(models)))
    # ax.set_yticklabels(models)
    # ------------------ 写入数值 ------------------
    # 【修改 5】：循环边界互换，行列索引互换
    threshold = (data_normalized_T.max() + data_normalized_T.min()) / 2
    for i in range(len(models)):  # 行：models (对应转置后的第一维)
        for j in range(len(groups)):  # 列：groups (对应转置后的第二维)
            val_orig = data_values_orig_T[i, j]
            # 文字颜色根据背景深浅自动切换 (需用到转置后的 normalized 数据)
            text_color = "white" if data_normalized_T[i, j] > threshold else "black"
            # 注意 ax.text(x, y, ...)，x 是列索引 j，y 是行索引 i
            ax.text(j, i, f"{val_orig:.3f}", ha="center", va="center", color="black", fontsize=18)
    # ------------------ 细节美化 ------------------
    ax.spines[:].set_visible(False)
    # 网格线边界同样需要互换 len(groups) 和 len(models)
    ax.set_xticks(np.arange(len(groups) + 1) - .5, minor=True)
    ax.set_yticks(np.arange(len(models) + 1) - .5, minor=True)
    ax.grid(which="minor", color="white", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", top=False, bottom=False, left=False, right=False)
    # 避免截断边缘
    plt.tight_layout()
    # plt.savefig(
    #     f'/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/figures/finetune/11.25/contrast/owen_Recall_{object_name}.4.22.vertical.pdf',
    #     bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    subpheno_df = pd.read_csv(
        '/bgi-seq-model-2/datasets/zhangkexin/meta_index/metaphlan4/validate/7.25/CRC/owen/CRC_colon_phenotype_Qin.txt',
        sep='\t', index_col=0)
    subpheno_df = subpheno_df.drop('T2104103043', errors='ignore')

    contrast_group_recall = {}
    subpheno_df_filter = subpheno_df.dropna(subset=['cStage']).copy()
    map_stage = {1.0: 'I', 2.0: 'II', 3.0: 'III', 4.0: 'IV'}
    subpheno_df_filter['cStage'] = subpheno_df_filter['cStage'].map(map_stage)
    contrast_group_recall_stage = {}
    order_stage = ['I', 'II', 'III', 'IV']
    MLP_res_s, _ = get_group_recall(subpheno_df_filter,
                                    '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results/MLP-model/indep.test/12.1/pandisease/',
                                    'owen_profile_mlp_probs.csv', order_stage, 'cStage', 1)
    contrast_group_recall_stage['MLP'] = MLP_res_s['mean'].tolist()
    for method in ['lr', 'rf', 'xgb']:
        ml_dir = '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results/ML-model/indep.test/11.25/pandisease/'
        res, _ = get_group_recall(subpheno_df_filter, ml_dir, f'owen_profile_{method}_probs.csv', order_stage, 'cStage', 1)
        contrast_group_recall_stage[method.upper()] = res['mean'].tolist()
    metaGPT_res_s, group_counts_stage = get_group_recall(subpheno_df_filter,
                                                         '/bgi-seq-model-2/codes/zhangkexin/meta_index/output/llama/v4/PanDisease/Split5.specific.aug.Full.sortabu.ema.12.4/',
                                                         'best_ckpt/result/probs/metaGPT.crc.owen.test.prob.csv',
                                                         order_stage, 'cStage', 0)
    contrast_group_recall_stage['UMetaGPT'] = metaGPT_res_s['mean'].tolist()
    contrast_df_stage = pd.DataFrame(contrast_group_recall_stage, index=order_stage)
    contrast_df_stage.loc['Overall'] = [0.7804, 0.8087, 0.7521, 0.8366, 0.8444]

    create_and_plot_heatmap_vertical(contrast_df_stage, group_counts_stage, 'cStage')

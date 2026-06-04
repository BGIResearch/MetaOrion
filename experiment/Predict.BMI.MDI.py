# @Date    : 2026/6/3 16:56 
# @Email   : zhangkexin2@genomics.cn
import os
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

from scipy.stats import pearsonr, spearmanr
from network_dysbiosis_index import calculate_mdi


def categorize_bmi(bmi):
    if pd.isna(bmi): return 'Unknown'
    try:
        bmi = float(bmi)
    except:
        return 'Unknown'

    if 18.5 <= bmi < 25:
        return 'Healthy weight'
    elif 25 <= bmi < 30:
        return 'Overweight'
    elif bmi >= 30:
        return 'Obese'
    else:
        return 'Underweight'


def get_star(p):
    if p <= 1e-3:
        return '***'
    elif p <= 1e-2:
        return '**'
    elif p <= 0.05:
        return '*'
    else:
        return 'ns'


# 新增：计算数据剔除离群点后的上边缘（即箱线图的实际可见最高点）
def get_upper_whisker(v):
    if len(v) == 0: return 0
    q1, q3 = np.percentile(v, [25, 75])
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    # 返回小于等于上边界的最大值
    return np.max(v[v <= upper_bound])


def add_stat_annotation(ax, df, x_col, y_col, hue_col, cohort_order, hue_order):
    results = []
    # 此时 seaborn 已经去掉了离群点，获取现在的 y 轴视觉范围作为比例尺
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min

    offsets = {'Healthy weight': -0.2, 'Overweight': 0, 'Obese': 0.2}
    global_max_y = y_max  # 用于记录画完所有线后需要的最高 y 轴刻度

    for x_idx, cohort in enumerate(cohort_order):
        cohort_data = df[df[x_col] == cohort]

        data_h = cohort_data[cohort_data[hue_col] == 'Healthy weight'][y_col].dropna().values
        data_ow = cohort_data[cohort_data[hue_col] == 'Overweight'][y_col].dropna().values
        data_ob = cohort_data[cohort_data[hue_col] == 'Obese'][y_col].dropna().values

        pairs = []
        if len(data_h) >= 3 and len(data_ow) >= 3:
            pairs.append(('Healthy weight', 'Overweight', data_h, data_ow))
        if len(data_ow) >= 3 and len(data_ob) >= 3:
            pairs.append(('Overweight', 'Obese', data_ow, data_ob))
        if len(data_h) >= 3 and len(data_ob) >= 3:
            pairs.append(('Healthy weight', 'Obese', data_h, data_ob))

        # 核心修改：动态获取当前 Cohort 里所有组的实际可见箱子最高点
        whiskers = [get_upper_whisker(d) for d in [data_h, data_ow, data_ob] if len(d) > 0]
        local_max = max(whiskers) if whiskers else y_max

        # 让连线紧贴在该 cohort 的最高箱子上方 (加上3%的留白)
        current_y = local_max + y_range * 0.03
        step = y_range * 0.08  # 连线之间的阶梯高度

        for g1, g2, v1, v2 in pairs:
            try:
                stat, p = stats.mannwhitneyu(v1, v2, alternative='two-sided')
                star = get_star(p)
                results.append({
                    'Cohort': cohort,
                    'Group 1': g1,
                    'Group 2': g2,
                    'n_Group1': len(v1),
                    'n_Group2': len(v2),
                    'U_Statistic': stat,
                    'P_value': p,
                    'Significance': star
                })

                if star != 'ns':
                    x1 = x_idx + offsets[g1]
                    x2 = x_idx + offsets[g2]

                    ax.plot([x1, x1, x2, x2],
                            [current_y, current_y + step * 0.2, current_y + step * 0.2, current_y], lw=1.2, c='k')
                    ax.text((x1 + x2) / 2, current_y + step * 0.2, star, ha='center', va='bottom', color='k',
                            fontsize=14)
                    current_y += step
            except Exception as e:
                pass

        if current_y > global_max_y:
            global_max_y = current_y

    # 画完所有的标注后，精准设置整个图的 Y 轴高度（最高连线再留出 5% 的留白防止星号被切出边界）
    ax.set_ylim(y_min, global_max_y + y_range * 0.05)

    results_df = pd.DataFrame(results)

    # 格式化 P 值以便于阅读 (科学计数法或保留小数)
    results_df['P_value_formatted'] = results_df['P_value'].apply(lambda x: f"{x:.4e}" if x < 0.001 else f"{x:.4f}")
    output_path = '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/figures/finetune/11.25/networks/4.1/Predict.mdi.bmi.state.csv'
    results_df.to_csv(output_path, index=False)


dir = '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results/network/1.8/Predict/'
biomarker_dir = '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results/network/1.8/pandisease/'
top30_core_good_biomarker = ['s-' + i.lower().replace(" ", "-") for i in
                             list(pd.read_csv(biomarker_dir + 'top30.negative.o3.csv', index_col=0).index)]
top30_core_bad_biomarker = ['s-' + i.lower().replace(" ", "-") for i in
                            list(pd.read_csv(biomarker_dir + 'top30.positive.o3.csv', index_col=0).index)]

bmi_all_cohort_df = []
label_dir = '/bgi-seq-model-2/datasets/zhangkexin/meta_index/metaphlan4/validate/processed_predict_cohorts/'
for file in os.listdir(dir):
    label_df = pd.read_csv(label_dir + file + '_metadata.tsv', sep='\t', index_col=0)
    sample_list = pd.read_csv(dir + file + '/samples.list.txt', header=None, index_col=0).index.tolist()
    individual_network = np.load(dir + file + '/pretrain.emb.individual.npy', mmap_mode='r')
    taxa2id = pickle.load(open(dir + file + '/pretrain.emb.individual.index.pkl', 'rb'))
    PAN_shared_df_features = calculate_mdi(individual_network, sample_list, taxa2id,
                                           sample_list, sample_list,
                                           top30_core_bad_biomarker, top30_core_good_biomarker)
    label_df = label_df.dropna(subset=['bmi'])
    merge_mdi_age = pd.merge(PAN_shared_df_features, label_df, left_on='sample_id', right_on='sampleID')
    merge_mdi_age['Project'] = file  # <--- 增加这一行，标记所属的队列

    corr_pearson, p_pearson = pearsonr(merge_mdi_age["MDI"], merge_mdi_age["bmi"])
    print(f'\ncohort: {file}, sample nums: {len(sample_list)}, bmi & MDI')
    print("Pearson r:", corr_pearson)
    print("Pearson p-value:", p_pearson)

    corr_spearman, p_spearman = spearmanr(merge_mdi_age["MDI"], merge_mdi_age["bmi"])
    print("Spearman rho:", corr_spearman)
    print("Spearman p-value:", p_spearman)

    bmi_all_cohort_df.append(merge_mdi_age)

bmi_all_cohort_df = pd.concat(
    bmi_all_cohort_df,
    axis=0
).reset_index(drop=True)

overall_pearson_r, overall_pearson_p = pearsonr(
    bmi_all_cohort_df['MDI'],
    bmi_all_cohort_df['bmi']
)

overall_spearman_r, overall_spearman_p = spearmanr(
    bmi_all_cohort_df['MDI'],
    bmi_all_cohort_df['bmi']
)

print('\n======== bmi & MDI OVERALL ========')

print('Overall Pearson r:', overall_pearson_r)
print('Overall Pearson p:', overall_pearson_p)

print('Overall Spearman rho:', overall_spearman_r)
print('Overall Spearman p:', overall_spearman_p)

bmi_all_cohort_df.to_csv(
    '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/figures/finetune/11.25/networks/4.1/predict.bmi.mdi.csv')

bmi_all_cohort_df['BMI_Group'] = bmi_all_cohort_df['bmi'].apply(categorize_bmi)
df_filtered = bmi_all_cohort_df[bmi_all_cohort_df['BMI_Group'].isin(['Healthy weight', 'Overweight', 'Obese'])].copy()

cohort_order = sorted(df_filtered['Project'].unique())
hue_order = ['Healthy weight', 'Overweight', 'Obese']
colors = {'Healthy weight': '#8AB1D2', 'Overweight': '#F5AA61', 'Obese': '#E58579'}

plt.rcParams['font.family'] = 'Arial'
plt.rc('font', size=16)
plt.rcParams['pdf.fonttype'] = 42

fig, ax = plt.subplots(figsize=(12, 7))

# 绘制箱线图
sns.boxplot(data=df_filtered, x='Project', y='MNDI', hue='BMI_Group',
            order=cohort_order, hue_order=hue_order, palette=colors,
            width=0.6, boxprops=dict(alpha=0.7), showfliers=False, ax=ax)

# 动态调用标注函数，并在内部自适应调节 Y 轴
add_stat_annotation(ax, df_filtered, 'Project', 'MNDI', 'BMI_Group', cohort_order, hue_order)

ax.set_ylabel('MDI')
ax.set_xlabel('Cohorts')
ax.tick_params(axis='x', rotation=30)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:3], labels[:3], title='BMI Group', frameon=False, loc='upper left', bbox_to_anchor=(1, 1))

plt.tight_layout()
plt.savefig(
    '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/figures/finetune/11.25/networks/4.1/Predict.mdi.bmi.pdf',
    dpi=800)
plt.show()

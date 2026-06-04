# @Date    : 2026/6/3 16:53 
# @Email   : zhangkexin2@genomics.cn
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statannotations.Annotator import Annotator
import warnings
warnings.filterwarnings('ignore')
# ==========================================
# 1. 全局配置与函数定义
# ==========================================
base_dir = '/bgi-seq-model-2/datasets/zhangkexin/meta_index/metaphlan4/validate/processed_predict_cohorts/'
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
def extract_clean_species_name(clade):
    parts = str(clade).split('|')
    for part in parts:
        if part.startswith('s__'):
            return part[3:].lower().replace(" ", "_").replace("-", "_")
    return ""
# 加载 Target Taxa，分成 Pos 和 Neg 两个集合
try:
    top30_pos = pd.read_csv(
        '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results/network/1.8/pandisease/top30.positive.o3.csv',
        index_col=0).index.tolist()
    top30_neg = pd.read_csv(
        '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results/network/1.8/pandisease/top30.negative.o3.csv',
        index_col=0).index.tolist()
except FileNotFoundError:
    print("警告: 找不到 top30 表格，确保运行路径下有这两个文件。")
    top30_pos, top30_neg = [], []
pos_taxa_cleaned = {t.lower().replace(" ", "_").replace("-", "_") for t in top30_pos}
neg_taxa_cleaned = {t.lower().replace(" ", "_").replace("-", "_") for t in top30_neg}
def calculate_shannon(col):
    col = col[col > 0]
    if len(col) == 0: return 0
    p = col / col.sum()
    return -np.sum(p * np.log(p))
# ==========================================
# 2. 批处理所有 Cohort 提取数据
# ==========================================
all_cum_pos_data = []
all_cum_neg_data = []
all_shannon_data = []
meta_files = glob.glob(os.path.join(base_dir, '*metadata.tsv'))
print(f"找到 {len(meta_files)} 个 metadata 文件，开始批处理...")
for meta_path in meta_files:
    filename = os.path.basename(meta_path)
    cohort_name = filename.replace('_metadata.tsv', '').replace('processed.', '')
    profile_path_1 = os.path.join(base_dir, f'processed.{cohort_name}_metaphlan_profile.tsv')
    profile_path_2 = os.path.join(base_dir, f'{cohort_name}_profile.tsv')
    if os.path.exists(profile_path_1):
        profile_path = profile_path_1
    elif os.path.exists(profile_path_2):
        profile_path = profile_path_2
    else:
        potential_profiles = glob.glob(os.path.join(base_dir, f'*{cohort_name}*profile.tsv'))
        if potential_profiles:
            profile_path = potential_profiles[0]
        else:
            continue
    # ---- 2.1 处理 Metadata ----
    meta = pd.read_csv(meta_path, sep='\t')
    meta.columns = [c.replace('"', '') for c in meta.columns]
    for col in meta.select_dtypes(include=['object']):
        meta[col] = meta[col].astype(str).str.replace('"', '')
    if 'bmi' not in meta.columns: continue
    meta['BMI_Group'] = meta['bmi'].apply(categorize_bmi)
    meta_filtered = meta[meta['BMI_Group'].isin(['Healthy weight', 'Overweight', 'Obese'])].copy()
    group_map = dict(zip(meta_filtered['sampleID'], meta_filtered['BMI_Group']))
    if len(group_map) == 0: continue
    # ---- 2.2 处理 Profile ----
    profile = pd.read_csv(profile_path, sep='\t', index_col=0)
    if not any('s__' in str(idx) for idx in profile.index):
        profile = pd.read_csv(profile_path, sep='\t', index_col=0, header=None)
    profile.index = profile.index.astype(str).str.replace('"', '')
    if isinstance(profile.columns, pd.core.indexes.base.Index) and type(profile.columns[0]) == str:
        profile.columns = profile.columns.str.replace('"', '')
    species_rows = [idx for idx in profile.index if 's__' in str(idx) and 't__' not in str(idx)]
    profile_species = profile.loc[species_rows].copy()
    common_samples = [col for col in profile_species.columns if col in group_map]
    if len(common_samples) == 0: continue
    profile_species = profile_species[common_samples]
    profile_species['clean_name'] = [extract_clean_species_name(idx) for idx in profile_species.index]
    # ---- 2.3 计算 Cumulative Abundance (Positive) ----
    matched_pos = profile_species[profile_species['clean_name'].isin(pos_taxa_cleaned)]
    cum_pos = matched_pos[common_samples].sum(axis=0)
    df_pos = pd.DataFrame({'Sample_ID': cum_pos.index, 'Cumulative_Abundance': cum_pos.values})
    df_pos['BMI_Group'] = df_pos['Sample_ID'].map(group_map)
    df_pos['Cohort'] = cohort_name
    all_cum_pos_data.append(df_pos)
    # ---- 2.4 计算 Cumulative Abundance (Negative) ----
    matched_neg = profile_species[profile_species['clean_name'].isin(neg_taxa_cleaned)]
    cum_neg = matched_neg[common_samples].sum(axis=0)
    df_neg = pd.DataFrame({'Sample_ID': cum_neg.index, 'Cumulative_Abundance': cum_neg.values})
    df_neg['BMI_Group'] = df_neg['Sample_ID'].map(group_map)
    df_neg['Cohort'] = cohort_name
    all_cum_neg_data.append(df_neg)
    # ---- 2.5 计算 Shannon ----
    shannon_vals = profile_species[common_samples].apply(calculate_shannon, axis=0)
    df_shannon = pd.DataFrame({'Sample_ID': shannon_vals.index, 'Shannon': shannon_vals.values})
    df_shannon['BMI_Group'] = df_shannon['Sample_ID'].map(group_map)
    df_shannon['Cohort'] = cohort_name
    all_shannon_data.append(df_shannon)
# ==========================================
# 3. 合并数据准备画图与统计
# ==========================================
if not all_cum_pos_data: raise ValueError("没有成功提取到任何数据。")
master_pos = pd.concat(all_cum_pos_data, ignore_index=True)
master_neg = pd.concat(all_cum_neg_data, ignore_index=True)
master_shannon = pd.concat(all_shannon_data, ignore_index=True)
cohort_order = sorted(master_pos['Cohort'].unique())
hue_order = ['Healthy weight', 'Overweight', 'Obese']
colors = {'Healthy weight': '#8AB1D2', 'Overweight': '#F5AA61', 'Obese': '#E58579'}
def generate_dynamic_pairs(df, group_col='Cohort', hue_col='BMI_Group'):
    pairs = []
    for c in cohort_order:
        existing_groups = df[df[group_col] == c][hue_col].unique()
        if 'Healthy weight' in existing_groups and 'Overweight' in existing_groups:
            pairs.append(((c, 'Healthy weight'), (c, 'Overweight')))
        if 'Healthy weight' in existing_groups and 'Obese' in existing_groups:
            pairs.append(((c, 'Healthy weight'), (c, 'Obese')))
        if 'Overweight' in existing_groups and 'Obese' in existing_groups:
            pairs.append(((c, 'Overweight'), (c, 'Obese')))
    return pairs
pairs_pos = generate_dynamic_pairs(master_pos)
pairs_neg = generate_dynamic_pairs(master_neg)
pairs_shannon = generate_dynamic_pairs(master_shannon)
# ==========================================
# 4. 生成组间检验统计表格
# ==========================================
def get_stat_table(df, metric_name, value_col):
    res = []
    for c in cohort_order:
        c_df = df[df['Cohort'] == c]
        test_pairs = [('Healthy weight', 'Overweight'), ('Healthy weight', 'Obese'), ('Overweight', 'Obese')]
        for g1, g2 in test_pairs:
            v1 = c_df[c_df['BMI_Group'] == g1][value_col].dropna().values
            v2 = c_df[c_df['BMI_Group'] == g2][value_col].dropna().values
            if len(v1) >= 3 and len(v2) >= 3:
                stat, p = stats.mannwhitneyu(v1, v2, alternative='two-sided')
                star = "***" if p <= 1e-3 else "**" if p <= 1e-2 else "*" if p <= 0.05 else "ns"
                res.append({
                    'Metric': metric_name,
                    'Cohort': c,
                    'Group 1': g1,
                    'Group 2': g2,
                    'n_Group1': len(v1),
                    'n_Group2': len(v2),
                    'P_value': p,
                    'Significance': star
                })
    return pd.DataFrame(res)
table_pos = get_stat_table(master_pos, 'Cum_Abundance_Pos', 'Cumulative_Abundance')
table_neg = get_stat_table(master_neg, 'Cum_Abundance_Neg', 'Cumulative_Abundance')
table_shannon = get_stat_table(master_shannon, 'Shannon_Diversity', 'Shannon')
master_stats = pd.concat([table_pos, table_neg, table_shannon], ignore_index=True)
master_stats['P_value'] = master_stats['P_value'].apply(lambda x: f"{x:.4e}" if x < 0.001 else f"{x:.4f}")
stats_path = '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/figures/finetune/11.25/networks/4.1/Predict.cumulative_shannon.stats.csv'
master_stats.to_csv(stats_path, index=False)
print(f"\n✅ 统计表格已保存至: {stats_path}")
# ==========================================
# 5. 绘图 (Grouped Boxplot - 3 Panels)
# ==========================================
plt.rcParams['font.family'] = 'Arial'
plt.rc('font', size=16)
plt.rcParams['pdf.fonttype'] = 42
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 18))
custom_thresholds = [[1e-3, "***"], [1e-2, "**"], [0.05, "*"], [1, "ns"]]
# -------- 图 1: Cumulative Abundance (Positive) --------
sns.boxplot(data=master_pos, x='Cohort', y='Cumulative_Abundance', hue='BMI_Group',
            order=cohort_order, hue_order=hue_order, palette=colors,
            width=0.6, boxprops=dict(alpha=0.7), showfliers=False, ax=axes[0])
if pairs_pos:
    annotator1 = Annotator(axes[0], pairs_pos, data=master_pos, x='Cohort', y='Cumulative_Abundance', hue='BMI_Group',
                           order=cohort_order, hue_order=hue_order)
    annotator1.configure(test='Mann-Whitney', text_format='star', loc='inside', hide_non_significant=True,
                         pvalue_thresholds=custom_thresholds)
    annotator1.apply_and_annotate()
axes[0].set_ylabel('Cum. Abundance (%)\n[PR-increasing]')
axes[0].set_xlabel('')
axes[0].tick_params(axis='x', rotation=30)
handles, labels = axes[0].get_legend_handles_labels()
axes[0].legend(handles[:3], labels[:3], title='BMI Group', frameon=False, loc='upper left', bbox_to_anchor=(1, 1))
# -------- 图 2: Cumulative Abundance (Negative) --------
sns.boxplot(data=master_neg, x='Cohort', y='Cumulative_Abundance', hue='BMI_Group',
            order=cohort_order, hue_order=hue_order, palette=colors,
            width=0.6, boxprops=dict(alpha=0.7), showfliers=False, ax=axes[1])
if pairs_neg:
    annotator2 = Annotator(axes[1], pairs_neg, data=master_neg, x='Cohort', y='Cumulative_Abundance', hue='BMI_Group',
                           order=cohort_order, hue_order=hue_order)
    annotator2.configure(test='Mann-Whitney', text_format='star', loc='inside', hide_non_significant=True,
                         pvalue_thresholds=custom_thresholds)
    annotator2.apply_and_annotate()
axes[1].set_ylabel('Cum. Abundance (%)\n[PR-decreasing]')
axes[1].set_xlabel('')
axes[1].tick_params(axis='x', rotation=30)
axes[1].get_legend().remove()
# -------- 图 3: Shannon Diversity --------
sns.boxplot(data=master_shannon, x='Cohort', y='Shannon', hue='BMI_Group',
            order=cohort_order, hue_order=hue_order, palette=colors,
            width=0.6, boxprops=dict(alpha=0.7), showfliers=False, ax=axes[2])
if pairs_shannon:
    annotator3 = Annotator(axes[2], pairs_shannon, data=master_shannon, x='Cohort', y='Shannon', hue='BMI_Group',
                           order=cohort_order, hue_order=hue_order)
    annotator3.configure(test='Mann-Whitney', text_format='star', loc='inside', hide_non_significant=True,
                         pvalue_thresholds=custom_thresholds)
    annotator3.apply_and_annotate()
axes[2].set_ylabel('Shannon Index')
axes[2].set_xlabel('Cohorts')
axes[2].tick_params(axis='x', rotation=30)
axes[2].get_legend().remove()
plt.tight_layout()
fig_path = '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/figures/finetune/11.25/networks/4.1/Predict.cumulative_shannon.bmi.pdf'
plt.savefig(fig_path, dpi=800)
plt.show()
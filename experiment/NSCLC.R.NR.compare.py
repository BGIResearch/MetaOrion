# @Date    : 2026/6/3 16:52 
# @Email   : zhangkexin2@genomics.cn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


# ==========================================
# 1. 辅助计算与通用绘图函数
# ==========================================
def cliffs_delta(lst1, lst2):
    x = np.array(lst1)[:, None]
    y = np.array(lst2)
    dom = np.sum(x > y) - np.sum(x < y)
    return dom / (len(lst1) * len(lst2))


def get_upper_whisker(v):
    if len(v) == 0: return 0
    q1, q3 = np.percentile(v, [25, 75])
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    return np.max(v[v <= upper_bound])


# 通用绘图核心：自动适应不同数据表、不同指标的绘制
def analyze_and_plot(ax, df, title, x_col='Group', y_col='Cumulative_Abundance', ylabel=''):
    R_vals = df[df[x_col] == 'R'][y_col].dropna().values
    NR_vals = df[df[x_col] == 'NR'][y_col].dropna().values

    # 统计检验
    stat, p_val = stats.mannwhitneyu(R_vals, NR_vals, alternative='two-sided')
    effect_size = cliffs_delta(R_vals, NR_vals)

    print(f"\n===== {title} =====")
    print(f"R 组样本数: {len(R_vals)}, 中位数: {np.median(R_vals):.4f}")
    print(f"NR 组样本数: {len(NR_vals)}, 中位数: {np.median(NR_vals):.4f}")
    print(f"Mann-Whitney U P-value: {p_val:.4e} ({p_val:.5f})")
    print(f"Cliff's Delta: {effect_size:.4f}")

    # 画箱线图和散点
    sns.boxplot(data=df, x=x_col, y=y_col,
                order=['R', 'NR'], palette={'R': '#E44A33', 'NR': '#4DBAD6'},
                width=0.5, boxprops=dict(alpha=0.7), showfliers=False, ax=ax)

    sns.stripplot(data=df, x=x_col, y=y_col,
                  order=['R', 'NR'], color='black', alpha=0.6, jitter=True, ax=ax)

    ax.set_title(f'{title}\n(R vs NR)', pad=15, fontweight='bold')
    ax.set_ylabel(ylabel)
    ax.set_xlabel('OS12 Group')

    # 星号标注逻辑 (ns也画连线)
    sig_text = "ns"
    if p_val <= 0.001:
        sig_text = "***"
    elif p_val <= 0.01:
        sig_text = "**"
    elif p_val <= 0.05:
        sig_text = "*"

    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min

    # 动态计算箱子最高点以贴合画线
    max_whisker = max(get_upper_whisker(R_vals), get_upper_whisker(NR_vals))
    current_y = max_whisker + y_range * 0.05
    step = y_range * 0.05

    ax.plot([0, 0, 1, 1], [current_y, current_y + step, current_y + step, current_y], lw=1.5, c='k')
    ax.text(0.5, current_y + step, sig_text, ha='center', va='bottom', color='k', fontsize=18)

    ax.set_ylim(y_min, max(y_max, current_y + step * 2.5))


# ==========================================
# 2. 数据加载与预处理 (Biomarkers & Shannon)
# ==========================================
# Load Metadata
meta = pd.read_csv(
    '/bgi-seq-model-2/datasets/zhangkexin/meta_index/metaphlan4/fine-tune/cancer.valid.dataset/SraRunTable-PRJNA1023797.csv')
meta = meta.dropna(subset=['OS12'])
meta = meta[meta['OS12'].isin(['R', 'NR'])]
meta['Sample_ID'] = meta['Run'] + '.mp4'
group_map = dict(zip(meta['Sample_ID'], meta['OS12']))

# Load Top 30 Taxa
top30_pos = pd.read_csv(
    '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results/network/1.8/pandisease/top30.positive.o3.csv',
    index_col=0).index.tolist()
top30_neg = pd.read_csv(
    '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results/network/1.8/pandisease/top30.negative.o3.csv',
    index_col=0).index.tolist()


def clean_target_name(name):
    return name.lower().replace(" ", "_").replace("-", "_")


pos_taxa_cleaned = {clean_target_name(t) for t in top30_pos}
neg_taxa_cleaned = {clean_target_name(t) for t in top30_neg}

# Load Profile Data
profile = pd.read_csv(
    '/bgi-seq-model-2/datasets/zhangkexin/meta_index/metaphlan4/fine-tune/cancer.valid.dataset/PRJNA1023797.profile',
    sep='\t', index_col=0)
species_rows = [idx for idx in profile.index if 's__' in str(idx) and 't__' not in str(idx)]
profile_species = profile.loc[species_rows].copy()


def extract_clean_species_name(clade):
    parts = str(clade).split('|')
    for part in parts:
        if part.startswith('s__'):
            return part[3:].lower().replace(" ", "_").replace("-", "_")
    return ""


profile_species['clean_name'] = [extract_clean_species_name(idx) for idx in profile_species.index]

# Match Samples and Taxa
valid_samples = [col for col in profile_species.columns if col in group_map]
matched_pos = profile_species[profile_species['clean_name'].isin(pos_taxa_cleaned)][valid_samples]
matched_neg = profile_species[profile_species['clean_name'].isin(neg_taxa_cleaned)][valid_samples]

# Calculations (Cum. Abundance & Shannon)
cum_pos = matched_pos.sum(axis=0).reset_index()
cum_pos.columns = ['Sample_ID', 'Cumulative_Abundance']
cum_pos['Group'] = cum_pos['Sample_ID'].map(group_map)

cum_neg = matched_neg.sum(axis=0).reset_index()
cum_neg.columns = ['Sample_ID', 'Cumulative_Abundance']
cum_neg['Group'] = cum_neg['Sample_ID'].map(group_map)


def calculate_shannon(col):
    col = col[col > 0]
    if len(col) == 0: return 0
    p = col / col.sum()
    return -np.sum(p * np.log(p))


shannon_vals = profile_species[valid_samples].apply(calculate_shannon, axis=0)
df_shannon = pd.DataFrame({'Sample_ID': shannon_vals.index, 'Shannon': shannon_vals.values})
df_shannon['Group'] = df_shannon['Sample_ID'].map(group_map)

# ==========================================
# 3. 数据加载与预处理 (MDI)
# ==========================================
df_mdi = pd.read_csv(
    '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results/network/1.8/cancer.valid/mdi.merge.label.csv')
df_os12_mdi = df_mdi.dropna(subset=['OS12']).copy()

# ==========================================
# 4. 图形绘制 (2x2 布局)
# ==========================================
plt.rcParams['font.family'] = 'Arial'
plt.rc('font', size=16)
plt.rcParams['pdf.fonttype'] = 42

# 创建 2x2 画布
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# (0, 0) 左上: Positive Biomarkers
analyze_and_plot(axes[1, 0], cum_pos, 'Positive Biomarkers',
                 x_col='Group', y_col='Cumulative_Abundance', ylabel='Cumulative Abundance (%)')

# (0, 1) 右上: Negative Biomarkers
analyze_and_plot(axes[1, 1], cum_neg, 'Negative Biomarkers',
                 x_col='Group', y_col='Cumulative_Abundance', ylabel='Cumulative Abundance (%)')

# (1, 0) 左下: MDI (默认使用 MDI_shared, 若需全景可改为 MDI_full)
analyze_and_plot(axes[0, 0], df_os12_mdi, 'Microbial Dysbiosis Index (Shared)',
                 x_col='OS12', y_col='MDI_shared', ylabel='MDI Shared')

# (1, 1) 右下: Shannon Diversity
analyze_and_plot(axes[0, 1], df_shannon, 'Shannon Diversity',
                 x_col='Group', y_col='Shannon', ylabel='Shannon Index')

plt.tight_layout(pad=2.0)
save_path = '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/figures/finetune/11.25/networks/4.1/cancer.R_vs_NR_shannon_mdi_abundance_4panels.pdf'
plt.savefig(save_path, dpi=800)
plt.show()
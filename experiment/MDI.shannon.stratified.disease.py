# @Date    : 2026/6/3
# @Email   : zhangkexin2@genomics.cn

import pickle
import warnings
import random
import colorsys
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.transforms as transforms
from matplotlib.ticker import MultipleLocator
from scipy.stats import mannwhitneyu, entropy
from sklearn.metrics import roc_curve, auc

# Import the core MDI calculation function from the first file
from network_dysbiosis_index import calculate_mdi

warnings.filterwarnings('ignore')

# ==========================================
# Global Visualization Configuration
# ==========================================
plt.rc('font', size=22)
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['axes.linewidth'] = 2.0
plt.rcParams['text.color'] = 'black'
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'
sns.set_style("ticks")

# ==========================================
# Output Directory Configuration
# ==========================================
SAVE_DIR = '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/figures/finetune/11.25/networks/4.1/tmp/'

# ==========================================
# Global Data Loading: Microbiome Profiles
# ==========================================
X = pd.read_csv(
    '/bgi-seq-model-2/datasets/zhangkexin/meta_index/metaphlan4/fine-tune/curated_LiS_CRC_20204_v1121.train_test.profile',
    sep='\t', index_col=0).transpose()

# Filter for species-level taxa (s__) and reformat column names
species_indices = [i for i in range(len(X.columns)) if X.columns[i].split('|')[-1].startswith('s__')]
X = X.iloc[:, species_indices]
X.columns = [col.split('|')[-1].replace('__', '-').replace(' ', '-').replace('_', '-').lower() for col in X.columns]


# ==========================================
# Utility & Plotting Functions
# ==========================================
def generate_random_colors(n=100, seed=111):
    """
    Generate a seaborn color palette with random hues but controlled,
    low-saturation and high-value parameters for a soft aesthetic.
    """
    random.seed(seed)
    colors = []
    for _ in range(n):
        h = random.random()
        s = random.uniform(0.2, 0.6)  # Low saturation
        v = random.uniform(0.8, 1.0)  # High brightness
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        colors.append((r, g, b))

    colors_hex = [f'#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}' for r, g, b in colors]
    return sns.color_palette(colors_hex)


def plot_disease_case_ctrl_mdi_split_rows(df_features, label_df):
    """
    Plot horizontal violins for MDI across multiple diseases, matching case vs ctrl
    within the same projects. Annotates significance and exports stats.
    Baseline (vertical line) is dynamically set to the mean MDI of the control group.
    """
    target_diseases = ['IBD', 'IBS', 'adenoma', 'CRC', 'melanoma', 'T2D', 'metabolic_syndrome', 'OB', 'IGT', 'BL',
                       'others', 'AS', 'ACVD', 'CKD', 'COVID-19']
    display_map = {'adenoma': 'Adenoma', 'melanoma': 'Melanoma', 'metabolic_syndrome': 'MS', 'others': 'Others'}

    label_df['sample_id'] = list(label_df.index)
    label_df['disease_united'] = label_df['disease_united'].apply(
        lambda d: 'Healthy' if str(d).lower() == 'healthy' else (str(d) if str(d) in target_diseases else 'others')
    )

    merged_df = pd.merge(df_features, label_df[['sample_id', 'disease_united', 'project']], on='sample_id', how='left')
    all_diseases = [d for d in merged_df['disease_united'].dropna().unique() if d != 'Healthy']
    diseases = [d for d in target_diseases if d in all_diseases]

    plot_rows = []
    for d in diseases:
        projs_with_disease = merged_df[merged_df['disease_united'] == d]['project'].unique()
        matched_subset = merged_df[merged_df['project'].isin(projs_with_disease)].copy()
        matched_subset = matched_subset[matched_subset['disease_united'].isin([d, 'Healthy'])].copy()
        matched_subset['plot_disease'] = d
        matched_subset['plot_group'] = matched_subset['disease_united'].apply(
            lambda x: 'ctrl' if x == 'Healthy' else 'case')
        if matched_subset['plot_group'].nunique() == 2:
            y_true = (matched_subset['group'] == 'case').astype(int)
            fpr, tpr, _ = roc_curve(y_true, matched_subset['MDI'])
            print(f"{d} AUC: {auc(fpr, tpr):.4f}")
        plot_rows.append(matched_subset)

    plot_df = pd.concat(plot_rows, ignore_index=True)
    plot_df['plot_disease'] = plot_df['plot_disease'].apply(lambda x: display_map.get(x, x))
    display_diseases = [display_map.get(d, d) for d in diseases]

    counts = plot_df.groupby(['plot_disease', 'plot_group']).size().reset_index(name='n')
    plot_df = pd.merge(plot_df, counts, on=['plot_disease', 'plot_group'])
    plot_df['y_label'] = plot_df.apply(
        lambda row: f"{row['plot_disease']} | {'case' if row['plot_group'] == 'case' else 'control'} (n={row['n']})",
        axis=1
    )

    plot_df['plot_group'] = pd.Categorical(plot_df['plot_group'], categories=['case', 'ctrl'], ordered=True)
    plot_df['plot_disease'] = pd.Categorical(plot_df['plot_disease'], categories=display_diseases, ordered=True)
    plot_df = plot_df.sort_values(['plot_disease', 'plot_group'])

    y_order = list(plot_df['y_label'].unique())
    unique_diseases = list(plot_df['plot_disease'].unique())

    fig, ax = plt.subplots(figsize=(12, len(y_order) * 0.7), facecolor='white')
    ax.set_facecolor('white')
    custom_palette = generate_random_colors(len(y_order), seed=101)

    trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)
    for i, d in enumerate(unique_diseases):
        indices = [idx for idx, label in enumerate(y_order) if label.startswith(f"{d} |")]
        if i % 2 == 0 and indices:
            start_y = min(indices) - 0.5
            height = max(indices) - min(indices) + 1
            rect = plt.Rectangle((-0.65, start_y), 1.65, height, transform=trans, color='lightgray', alpha=0.5,
                                 zorder=0, clip_on=False)
            ax.add_patch(rect)

    def filter_extreme(group):
        q_low, q_high = group['MDI'].quantile(0.00001), group['MDI'].quantile(0.99999)
        return group[(group['MDI'] >= q_low) & (group['MDI'] <= q_high)]

    violin_df = plot_df.groupby('y_label', group_keys=False).apply(filter_extreme)
    v = sns.violinplot(
        data=violin_df, x='MDI', y='y_label', hue='y_label',
        order=y_order, palette=custom_palette, dodge=False, color='#333333',
        inner=None, linewidth=1.5, cut=0, ax=ax, zorder=2, legend=False, bw_adjust=0.8
    )
    for pc in v.collections:
        pc.set_edgecolor('#333333')
        pc.set_alpha(1)

    sns.boxplot(
        data=violin_df, x='MDI', y='y_label', order=y_order, dodge=False,
        width=0.3, color='white', linewidth=1.5, fliersize=4,
        flierprops={'marker': 'o', 'markerfacecolor': 'black', 'markeredgecolor': 'black', 'markersize': 4,
                    'alpha': 0.7},
        showcaps=False, ax=ax, zorder=3,
        boxprops={'edgecolor': '#333333'}, whiskerprops={'color': '#333333', 'linewidth': 1.5},
        medianprops={'color': '#a82315', 'linewidth': 3}
    )

    annotation_x = 2.8
    stats_results = []

    for d in unique_diseases:
        indices = [idx for idx, label in enumerate(y_order) if label.startswith(f"{d} |")]
        y_pos = sum(indices) / len(indices)
        disease_data = plot_df[plot_df['plot_disease'] == d]

        case_vals = disease_data[disease_data['plot_group'] == 'case']['MDI']
        ctrl_vals = disease_data[disease_data['plot_group'] == 'ctrl']['MDI']
        case_shannon = disease_data[disease_data['plot_group'] == 'case']['shannon'].dropna()
        ctrl_shannon = disease_data[disease_data['plot_group'] == 'ctrl']['shannon'].dropna()

        mdi_p_val, shannon_p_val, mdi_auc_val, shannon_auc_val = None, None, None, None
        stars = "—"
        star_color = '#333333'

        if len(case_vals) > 0 and len(ctrl_vals) > 0:
            _, mdi_p_val = mannwhitneyu(case_vals, ctrl_vals)
            y_true_mdi = [1] * len(case_vals) + [0] * len(ctrl_vals)
            y_score_mdi = list(case_vals) + list(ctrl_vals)
            fpr_m, tpr_m, _ = roc_curve(y_true_mdi, y_score_mdi)
            mdi_auc_val = auc(fpr_m, tpr_m)

            if len(case_shannon) > 0 and len(ctrl_shannon) > 0:
                _, shannon_p_val = mannwhitneyu(case_shannon, ctrl_shannon)
                y_true_shan = [1] * len(case_shannon) + [0] * len(ctrl_shannon)
                y_score_shan = list(case_shannon) + list(ctrl_shannon)
                fpr_s, tpr_s, _ = roc_curve(y_true_shan, y_score_shan)
                shannon_auc_val = max(auc(fpr_s, tpr_s), 1 - auc(fpr_s, tpr_s))

                if mdi_p_val < 0.05 and shannon_p_val >= 0.05:
                    star_color = '#a82315'
                elif mdi_p_val < shannon_p_val:
                    star_color = '#eb6100'

            if mdi_p_val < 0.001:
                stars = '***'
            elif mdi_p_val < 0.01:
                stars = '**'
            elif mdi_p_val < 0.05:
                stars = '*'
            else:
                stars = ' '

        ax.text(annotation_x, y_pos, stars, va='center', ha='center', fontsize=20, fontweight='bold', color=star_color)
        stats_results.append({
            'disease': d, 'n_case': len(case_vals), 'n_ctrl': len(ctrl_vals),
            'MDI_pvalue': mdi_p_val, 'MDI_AUC': mdi_auc_val,
            'Shannon_pvalue': shannon_p_val, 'Shannon_AUC': shannon_auc_val
        })

    ctrl_mean = plot_df[plot_df['plot_group'] == 'ctrl']['MDI'].mean()
    print('pandisease control mean MDI:', ctrl_mean, ' | Count:', len(plot_df[plot_df['plot_group'] == 'ctrl']['MDI']))
    plt.axvline(x=ctrl_mean, color='gray', linestyle='--', alpha=0.5)

    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.tick_params(axis='both', labelsize=22)
    plt.xlabel('MDI Score', fontsize=24)
    plt.ylabel('')
    sns.despine()
    plt.xlim(left=-2, right=annotation_x + 0.5)
    fig.subplots_adjust(left=0.35)

    plt.savefig(f'{SAVE_DIR}pandisease.disease.case.ctrl.mdi.pdf', dpi=800, bbox_inches='tight')
    plt.show()

    pd.DataFrame(stats_results).to_csv(f'{SAVE_DIR}pandisease.disease.case.ctrl.stats.tsv', sep='\t', index=False)


def plot_vertical_cohort_case_ctrl_mdi(df_features, label_df, disease, case_color, ctrl_color):
    """
    Plot vertical violins for MDI, grouping by cohort/project.
    Baseline (horizontal line) is dynamically set to the mean MDI of the control group.
    """
    label_df = label_df.copy()
    label_df['sample_id'] = list(label_df.index)
    label_df['disease_united'] = ['Healthy' if str(i).lower() == 'healthy' else i for i in label_df['disease_united']]

    plot_df = pd.merge(df_features, label_df[['sample_id', 'project', 'disease_united']], on='sample_id', how='left')
    plot_df['group'] = pd.Categorical(plot_df['group'], categories=['case', 'ctrl'], ordered=True)

    counts = plot_df.groupby(['project', 'disease_united']).size().reset_index(name='n')
    plot_df = pd.merge(plot_df, counts, on=['project', 'disease_united'])
    plot_df['x_label'] = plot_df.apply(lambda x: f"{x['project']} | {x['disease_united']} (n={x['n']})", axis=1)

    plot_df = plot_df.sort_values(['project', 'group'])
    x_order = list(plot_df['x_label'].unique())
    projects = plot_df['project'].unique()

    fig, ax = plt.subplots(figsize=(len(x_order) * 0.8, 10))
    color_map = {label: case_color if plot_df[plot_df['x_label'] == label]['group'].iloc[0] == 'case' else ctrl_color
                 for label in x_order}

    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    for i, proj in enumerate(projects):
        if i % 2 == 0:
            proj_labels = [label for label in x_order if label.startswith(proj)]
            indices = [x_order.index(l) for l in proj_labels]
            if indices:
                start_x, end_x = min(indices) - 0.5, max(indices) + 0.5
                rect = plt.Rectangle((start_x, -0.9), end_x - start_x, 1.9, transform=trans, color='lightgray',
                                     alpha=0.5, zorder=0, clip_on=False)
                ax.add_patch(rect)

    v = sns.violinplot(
        data=plot_df, x='x_label', y='MDI', hue='x_label',
        order=x_order, palette=color_map, dodge=False, color='#333333',
        inner=None, linewidth=1.5, cut=0, ax=ax, zorder=2, legend=False, bw_adjust=0.8
    )
    for pc in v.collections:
        pc.set_edgecolor('#333333')
        pc.set_alpha(1)

    sns.boxplot(
        data=plot_df, x='x_label', y='MDI', order=x_order, dodge=False,
        width=0.3, color='white', linewidth=1.5, fliersize=4,
        flierprops={'marker': 'o', 'markerfacecolor': 'black', 'markeredgecolor': 'black', 'markersize': 4,
                    'alpha': 0.7},
        showcaps=False, ax=ax, zorder=3,
        boxprops={'edgecolor': '#333333'}, whiskerprops={'color': '#333333', 'linewidth': 1.5},
        medianprops={'color': '#a82315', 'linewidth': 3}
    )

    y_max, y_min = plot_df['MDI'].max(), plot_df['MDI'].min()
    annotation_y = y_max + (y_max - y_min) * 0.05
    stats_results = []

    for proj in projects:
        indices = [idx for idx, label in enumerate(x_order) if label.startswith(f"{proj} |")]
        if not indices: continue
        x_pos = sum(indices) / len(indices)

        proj_data = plot_df[plot_df['project'] == proj]
        case_vals = proj_data[proj_data['group'] == 'case']['MDI']
        ctrl_vals = proj_data[proj_data['group'] == 'ctrl']['MDI']
        case_shannon = proj_data[proj_data['group'] == 'case']['shannon'].dropna()
        ctrl_shannon = proj_data[proj_data['group'] == 'ctrl']['shannon'].dropna()

        mdi_p_val, shannon_p_val, mdi_auc_val, shannon_auc_val = None, None, None, None
        stars = "—"
        star_color = '#333333'

        if len(case_vals) > 0 and len(ctrl_vals) > 0:
            _, mdi_p_val = mannwhitneyu(case_vals, ctrl_vals)
            y_true_mdi = [1] * len(case_vals) + [0] * len(ctrl_vals)
            y_score_mdi = list(case_vals) + list(ctrl_vals)
            fpr_m, tpr_m, _ = roc_curve(y_true_mdi, y_score_mdi)
            mdi_auc_val = auc(fpr_m, tpr_m)

            if len(case_shannon) > 0 and len(ctrl_shannon) > 0:
                _, shannon_p_val = mannwhitneyu(case_shannon, ctrl_shannon)
                y_true_shan = [1] * len(case_shannon) + [0] * len(ctrl_shannon)
                y_score_shan = list(case_shannon) + list(ctrl_shannon)
                fpr_s, tpr_s, _ = roc_curve(y_true_shan, y_score_shan)
                shannon_auc_val = max(auc(fpr_s, tpr_s), 1 - auc(fpr_s, tpr_s))

                if mdi_p_val < 0.05 and shannon_p_val >= 0.05:
                    star_color = '#a82315'
                elif mdi_p_val < shannon_p_val:
                    star_color = '#eb6100'

            if mdi_p_val < 0.001:
                stars = '***'
            elif mdi_p_val < 0.01:
                stars = '**'
            elif mdi_p_val < 0.05:
                stars = '*'
            else:
                stars = ''

        if stars:
            ax.text(x_pos, annotation_y, stars, va='bottom', ha='center', fontsize=20, fontweight='bold',
                    color=star_color)

        stats_results.append({
            'project': proj, 'n_case': len(case_vals), 'n_ctrl': len(ctrl_vals),
            'MDI_pvalue': mdi_p_val, 'MDI_AUC': mdi_auc_val,
            'Shannon_pvalue': shannon_p_val, 'Shannon_AUC': shannon_auc_val
        })

    ctrl_mean = plot_df[plot_df['group'] == 'ctrl']['MDI'].mean()
    print(f'{disease} control mean MDI:', ctrl_mean, ' | Count:', len(plot_df[plot_df['group'] == 'ctrl']['MDI']))
    plt.axhline(y=ctrl_mean, color='gray', linestyle='--', alpha=0.5)

    plt.setp(ax.get_xticklabels(), rotation=90, ha='right', rotation_mode='anchor')
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=22)
    plt.ylabel('MDI Score', fontsize=24, labelpad=15)
    plt.xlabel('')
    sns.despine()
    ax.yaxis.set_major_locator(MultipleLocator(1))
    plt.ylim(top=annotation_y + (y_max - y_min) * 0.1)
    fig.subplots_adjust(bottom=0.35)

    plt.savefig(f'{SAVE_DIR}{disease}.project.individual.case.ctrl.mdi.pdf', dpi=800, bbox_inches='tight')
    plt.show()

    pd.DataFrame(stats_results).to_csv(f'{SAVE_DIR}{disease}.project.individual.case.ctrl.project.stats.tsv', sep='\t',
                                       index=False)


def plot_horizontal_cohort_case_ctrl_mdi(df_features, label_df, disease, case_color, ctrl_color):
    """
    Plot horizontal violins for MDI, grouping by cohort/project.
    Baseline (vertical line) is dynamically set to the mean MDI of the control group.
    """
    label_df['sample_id'] = list(label_df.index)
    label_df['disease_united'] = ['Healthy' if str(i).lower() == 'healthy' else i for i in label_df['disease_united']]

    plot_df = pd.merge(df_features, label_df[['sample_id', 'project', 'disease_united']], on='sample_id', how='left')
    plot_df['group'] = pd.Categorical(plot_df['group'], categories=['case', 'ctrl'], ordered=True)

    counts = plot_df.groupby(['project', 'disease_united']).size().reset_index(name='n')
    plot_df = pd.merge(plot_df, counts, on=['project', 'disease_united'])
    plot_df['y_label'] = plot_df.apply(lambda x: f"{x['project']} | {x['disease_united']} (n={x['n']})", axis=1)

    plot_df = plot_df.sort_values(['project', 'group'])
    y_order = list(plot_df['y_label'].unique())
    projects = plot_df['project'].unique()

    fig, ax = plt.subplots(figsize=(12, len(y_order) * 0.7))
    color_map = {label: case_color if plot_df[plot_df['y_label'] == label]['group'].iloc[0] == 'case' else ctrl_color
                 for label in y_order}

    trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)
    for i, proj in enumerate(projects):
        if i % 2 == 0:
            proj_labels = [label for label in y_order if label.startswith(proj)]
            indices = [y_order.index(l) for l in proj_labels]
            if indices:
                start_y, end_y = min(indices) - 0.5, max(indices) + 0.5
                rect = plt.Rectangle((-0.85, start_y), 1.85, end_y - start_y, transform=trans, color='lightgray',
                                     alpha=0.5, zorder=0, clip_on=False)
                ax.add_patch(rect)

    v = sns.violinplot(
        data=plot_df, x='MDI', y='y_label', hue='y_label',
        order=y_order, palette=color_map, dodge=False, color='#333333',
        inner=None, linewidth=1.5, cut=0, ax=ax, zorder=2, legend=False, bw_adjust=0.8
    )
    for pc in v.collections:
        pc.set_edgecolor('#333333')
        pc.set_alpha(1)

    sns.boxplot(
        data=plot_df, x='MDI', y='y_label', order=y_order, dodge=False,
        width=0.3, color='white', linewidth=1.5, fliersize=4,
        flierprops={'marker': 'o', 'markerfacecolor': 'black', 'markeredgecolor': 'black', 'markersize': 4,
                    'alpha': 0.7},
        showcaps=False, ax=ax, zorder=3,
        boxprops={'edgecolor': '#333333'}, whiskerprops={'color': '#333333', 'linewidth': 1.5},
        medianprops={'color': '#a82315', 'linewidth': 3}
    )

    x_max = plot_df['MDI'].max()
    annotation_x = x_max + 0.8
    stats_results = []

    for proj in projects:
        indices = [idx for idx, label in enumerate(y_order) if label.startswith(f"{proj} |")]
        if not indices: continue
        y_pos = sum(indices) / len(indices)

        proj_data = plot_df[plot_df['project'] == proj]
        case_vals = proj_data[proj_data['group'] == 'case']['MDI']
        ctrl_vals = proj_data[proj_data['group'] == 'ctrl']['MDI']
        case_shannon = proj_data[proj_data['group'] == 'case']['shannon'].dropna()
        ctrl_shannon = proj_data[proj_data['group'] == 'ctrl']['shannon'].dropna()

        mdi_p_val, shannon_p_val, mdi_auc_val, shannon_auc_val = None, None, None, None
        stars = "—"
        star_color = '#333333'

        if len(case_vals) > 0 and len(ctrl_vals) > 0:
            _, mdi_p_val = mannwhitneyu(case_vals, ctrl_vals)
            y_true_mdi = [1] * len(case_vals) + [0] * len(ctrl_vals)
            y_score_mdi = list(case_vals) + list(ctrl_vals)
            fpr_m, tpr_m, _ = roc_curve(y_true_mdi, y_score_mdi)
            mdi_auc_val = auc(fpr_m, tpr_m)

            if len(case_shannon) > 0 and len(ctrl_shannon) > 0:
                _, shannon_p_val = mannwhitneyu(case_shannon, ctrl_shannon)
                y_true_shan = [1] * len(case_shannon) + [0] * len(ctrl_shannon)
                y_score_shan = list(case_shannon) + list(ctrl_shannon)
                fpr_s, tpr_s, _ = roc_curve(y_true_shan, y_score_shan)
                shannon_auc_val = max(auc(fpr_s, tpr_s), 1 - auc(fpr_s, tpr_s))

                if mdi_p_val < 0.05 and shannon_p_val >= 0.05:
                    star_color = '#a82315'
                elif mdi_p_val < shannon_p_val:
                    star_color = '#eb6100'

            if mdi_p_val < 0.001:
                stars = '***'
            elif mdi_p_val < 0.01:
                stars = '**'
            elif mdi_p_val < 0.05:
                stars = '*'
            else:
                stars = ''

        if stars:
            ax.text(annotation_x, y_pos, stars, va='center', ha='center', fontsize=20, fontweight='bold',
                    color=star_color)

        stats_results.append({
            'project': proj, 'n_case': len(case_vals), 'n_ctrl': len(ctrl_vals),
            'MDI_pvalue': mdi_p_val, 'MDI_AUC': mdi_auc_val,
            'Shannon_pvalue': shannon_p_val, 'Shannon_AUC': shannon_auc_val
        })

    ctrl_mean = plot_df[plot_df['group'] == 'ctrl']['MDI'].mean()
    print(f'{disease} control mean MDI:', ctrl_mean, ' | Count:', len(plot_df[plot_df['group'] == 'ctrl']['MDI']))
    plt.axvline(x=ctrl_mean, color='gray', linestyle='--', alpha=0.5)

    ax.tick_params(axis='both', labelsize=22)
    plt.xlabel('MDI Score', fontsize=24)
    plt.ylabel('')
    sns.despine()
    ax.xaxis.set_major_locator(MultipleLocator(1))
    plt.xlim(left=-2, right=annotation_x * 1.1)
    fig.subplots_adjust(left=0.35)

    plt.savefig(f'{SAVE_DIR}{disease}.project.individual.case.ctrl.mdi.pdf', dpi=800, bbox_inches='tight')
    plt.show()

    pd.DataFrame(stats_results).to_csv(f'{SAVE_DIR}{disease}.project.individual.case.ctrl.project.stats.tsv', sep='\t',
                                       index=False)


# ==========================================
# Main Execution Block
# ==========================================
if __name__ == '__main__':
    label_df_path = '/bgi-seq-model-2/datasets/zhangkexin/meta_index/metaphlan4/fine-tune/curated_LiS_CRC_20204_v0908.train_test.phe'

    # ---------------------------------------------------------
    # 1. PANDISEASE Execution
    # ---------------------------------------------------------
    print('--------PANDISEASE')
    pan_dir = '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results/network/1.8/pandisease/'
    pan_sample_list = list(pd.read_csv(pan_dir + 'pandisease.split1.valtest.samples.csv')['samples'])
    pan_label_df = pd.read_csv(label_df_path, sep='\t', index_col=0)

    pan_case_samples = [i for i in pan_sample_list if pan_label_df.loc[i, 'disease_united'] != 'healthy']
    pan_ctrl_samples = [i for i in pan_sample_list if pan_label_df.loc[i, 'disease_united'] == 'healthy']

    pan_network = np.load(pan_dir + 'pandisease.split1.valtest.pretrain.emb.cosine.individual.npy')
    pan_taxa2id = pickle.load(
        open(pan_dir + 'pandisease.split1.valtest.pretrain.emb.cosine.individual.index.pkl', 'rb'))

    pan_biomarker = pd.read_csv(
        '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results/features/11.25/sort.multidisease.mean/pandisease.biomarker.txt',
        sep='\t', header=None)
    pan_biomarker['weight.abs'] = pan_biomarker[1].abs()
    pan_biomarker = pan_biomarker.sort_values('weight.abs', ascending=False)
    pan_top30_good = list(pan_biomarker[pan_biomarker[1] < 0].iloc[:, 0])
    pan_top30_bad = list(pan_biomarker[pan_biomarker[1] > 0].iloc[:, 0])

    df_features_pan = calculate_mdi(pan_network, pan_sample_list, pan_taxa2id, pan_case_samples, pan_ctrl_samples,
                                    pan_top30_bad, pan_top30_good)

    X_pandisease = X.loc[pan_case_samples + pan_ctrl_samples, :]
    shannon_df_pan = pd.DataFrame(
        {'sample_id': X_pandisease.index, 'shannon': X_pandisease.apply(lambda x: entropy(x[x > 0]), axis=1)})
    df_features_pan = pd.merge(df_features_pan, shannon_df_pan, on='sample_id', how='left')

    plot_disease_case_ctrl_mdi_split_rows(df_features_pan, pan_label_df)

    # 🌟 强制清理 PANDISEASE 的大变量并释放内存
    plt.close('all')
    del pan_network, df_features_pan, X_pandisease, shannon_df_pan, pan_biomarker
    gc.collect()

    # ---------------------------------------------------------
    # 2. CRC Execution
    # ---------------------------------------------------------
    print('--------CRC')
    crc_dir = '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results/network/1.8/'
    crc_profile = pd.read_csv(crc_dir + 'crc.all.profile.clean.csv', index_col=0)
    crc_sample_list = list(crc_profile.index)

    crc_label_df = pd.read_csv(label_df_path, sep='\t', index_col=0)
    crc_label_df = crc_label_df[crc_label_df['project'] != 'HanniganGD_2017']

    crc_case_samples = [i for i in crc_sample_list if
                        i in crc_label_df.index and crc_label_df.loc[i, 'disease_united'] == 'CRC']
    crc_ctrl_samples = [i for i in crc_sample_list if
                        i in crc_label_df.index and crc_label_df.loc[i, 'disease_united'] == 'healthy']

    crc_network = np.load(crc_dir + 'crc.pretrain.emb.cosine.individual.npy')
    crc_taxa2id = pickle.load(open(crc_dir + 'crc.pretrain.emb.cosine.individual.index.pkl', 'rb'))

    crc_biomarker = pd.read_csv(
        '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results/features/11.25/sort.multidisease.mean/CRC.biomarker.txt',
        sep='\t', header=None)
    crc_biomarker = crc_biomarker.sort_values(by=1, key=abs, ascending=False)
    crc_good_biomarker = list(crc_biomarker[crc_biomarker[1] < 0].iloc[:, 0])
    crc_bad_biomarker = list(crc_biomarker[crc_biomarker[1] > 0].iloc[:, 0])

    df_features_crc = calculate_mdi(crc_network, crc_sample_list, crc_taxa2id, crc_case_samples, crc_ctrl_samples,
                                    crc_bad_biomarker, crc_good_biomarker)

    X_CRC = X.loc[crc_case_samples + crc_ctrl_samples, :]
    shannon_df_crc = pd.DataFrame(
        {'sample_id': X_CRC.index, 'shannon': X_CRC.apply(lambda x: entropy(x[x > 0]), axis=1)})
    df_features_crc = pd.merge(df_features_crc, shannon_df_crc, on='sample_id', how='left')

    plot_vertical_cohort_case_ctrl_mdi(df_features_crc, crc_label_df, 'CRC', '#ed91bb', '#b6dafd')

    # 🌟 强制清理 CRC 的大变量并释放内存
    plt.close('all')
    del crc_network, df_features_crc, X_CRC, shannon_df_crc, crc_biomarker
    gc.collect()

    # ---------------------------------------------------------
    # 3. IBD Execution
    # ---------------------------------------------------------
    print('--------IBD')
    ibd_dir = '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results/network/1.8/ibd/'
    ibd_network = np.load(ibd_dir + 'ibd.pretrain.emb.individual.npy')
    ibd_taxa2id = pickle.load(open(ibd_dir + 'ibd.pretrain.emb.individual.index.pkl', 'rb'))

    ibd_biomarker = pd.read_csv(
        '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results/features/11.25/sort.multidisease.mean/IBD.biomarker.txt',
        sep='\t', header=None)
    ibd_biomarker = ibd_biomarker.sort_values(by=1, key=abs, ascending=False)
    ibd_good_biomarker = list(ibd_biomarker[ibd_biomarker[1] < 0].iloc[:, 0])
    ibd_bad_biomarker = list(ibd_biomarker[ibd_biomarker[1] > 0].iloc[:, 0])

    with open(ibd_dir + 'ibd.pretrain.emb.samples.list.txt', 'r') as f:
        ibd_sample_list = [line.strip() for line in f.readlines()]

    ibd_label_df = pd.read_csv(label_df_path, sep='\t', index_col=0)
    ibd_label_df = ibd_label_df[
        ~ibd_label_df['project'].isin(['IaniroG_2022', 'LiJ_2014', 'LifeLinesDeep_2016', 'MetaCardis_2020_a'])]

    ibd_case_samples = [i for i in ibd_sample_list if
                        i in ibd_label_df.index and ibd_label_df.loc[i, 'disease_united'] == 'IBD']
    ibd_ctrl_samples = [i for i in ibd_sample_list if
                        i in ibd_label_df.index and ibd_label_df.loc[i, 'disease_united'] == 'healthy']

    df_features_ibd = calculate_mdi(ibd_network, ibd_sample_list, ibd_taxa2id, ibd_case_samples, ibd_ctrl_samples,
                                    ibd_bad_biomarker, ibd_good_biomarker)

    X_IBD = X.loc[ibd_case_samples + ibd_ctrl_samples, :]
    shannon_df_ibd = pd.DataFrame(
        {'sample_id': X_IBD.index, 'shannon': X_IBD.apply(lambda x: entropy(x[x > 0]), axis=1)})
    df_features_ibd = pd.merge(df_features_ibd, shannon_df_ibd, on='sample_id', how='left')

    plot_horizontal_cohort_case_ctrl_mdi(df_features_ibd, ibd_label_df, 'IBD', '#F0A780', '#B8DDBC')

    plt.close('all')
    del ibd_network, df_features_ibd, X_IBD, shannon_df_ibd, ibd_biomarker
    gc.collect()
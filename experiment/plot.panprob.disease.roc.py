# @Date    : 2026/6/10
# @Email   : zhangkexin2@genomics.cn

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.metrics import roc_curve, roc_auc_score, auc

import warnings

warnings.filterwarnings('ignore')

# ==========================================
# Configuration and Constants
# ==========================================
PRED_BASE_DIR = '/bgi-seq-model-2/codes/zhangkexin/meta_index/output/llama/v4/PanDisease/Split5.specific.aug.Full.sortabu.ema.12.4/'
LABEL_PATH = '/bgi-seq-model-2/datasets/zhangkexin/meta_index/metaphlan4/fine-tune/curated_LiS_CRC_20204_v0908.train_test.phe'
SAVE_DIR = '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/figures/finetune/11.25/'

COLOR_LIST = [
    '#469393', '#84C2AE', '#CDE4E4', '#1D75B5',
    '#BC388B', '#DF639C', '#EC9AB4', '#EFB1BB', '#F5D4CE',
    '#FFF7AC', '#8CA3C3', '#F8D3A9', '#818181'
]

CLASS_GROUPS = {
    'Digestive disorders': ['IBD', 'IBS', 'Adenoma'],
    'Neoplasms': ['CRC'],
    'Metabolic/endocrine diseases': ['T2D', 'MS', 'OB', 'IGT', 'BL'],
    'Other diseases': ['AS', 'ACVD', 'CKD', 'COVID-19']
}


# ==========================================
# Utility Functions
# ==========================================
def auc_ci(auc_val, label, alpha=0.05):
    """
    Calculate the 95% Confidence Interval for AUC using the analytical method.
    """
    label = np.array(label)
    n1, n2 = np.sum(label == 1), np.sum(label == 0)
    q1 = auc_val / (2 - auc_val)
    q2 = (2 * auc_val ** 2) / (1 + auc_val)
    se = np.sqrt(
        (auc_val * (1 - auc_val) + (n1 - 1) * (q1 - auc_val ** 2) + (n2 - 1) * (q2 - auc_val ** 2)) / (n1 * n2))

    z_lower, z_upper = norm.interval(1 - alpha)
    lowerb, upperb = auc_val + z_lower * se, auc_val + z_upper * se
    return lowerb, upperb


def load_and_preprocess_data():
    """
    Load predictions across 5 splits, concatenate them, and merge with metadata.
    """
    print(">>> Loading predictions and metadata...")
    pan_prob_list = []

    # Load 5-fold predictions
    for i in range(1, 6):
        file_path = os.path.join(PRED_BASE_DIR, f'split{i}/best_ckpt/result/probs/metaGPT.pandisease.test.prob.csv')
        pan_pred = pd.read_csv(file_path)
        pan_prob_list.append(pan_pred)

    pan_pred_concat = pd.concat(pan_prob_list, axis=0).reset_index(drop=True)

    # Load metadata
    label_df = pd.read_csv(LABEL_PATH, sep='\t', index_col=0)
    label_df['sample'] = list(label_df.index)

    # Merge and standardize disease names
    multi_pred_proj = pd.merge(pan_pred_concat, label_df, on='sample', how='inner')
    multi_pred_proj['disease_united'] = multi_pred_proj['disease_united'].replace({
        'metabolic_syndrome': 'MS',
        'adenoma': 'Adenoma'
    })

    return pan_prob_list, pan_pred_concat, multi_pred_proj


# ==========================================
# Plotting Functions
# ==========================================
def plot_disease_classes_roc(multi_pred_proj, save_path):
    """
    Plot ROC curves grouped by disease classes into a 2x2 grid.
    """
    print(">>> Generating 2x2 Subplots for Disease Classes ROC...")

    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 17,
        'font.family': 'Arial',
        'pdf.fonttype': 42,
        'xtick.labelsize': 17,
        'ytick.labelsize': 17
    })

    fig, axes = plt.subplots(2, 2, figsize=(10, 10), gridspec_kw={'wspace': 0.2, 'hspace': 0.35})
    axes = axes.flatten()

    color_idx = 0
    for idx, (class_name, diseases) in enumerate(CLASS_GROUPS.items()):
        ax = axes[idx]

        for disease in diseases:
            # Filter projects that contain the specific disease
            phe_project = set(multi_pred_proj[multi_pred_proj['disease_united'] == disease]['project'])
            phe_df = multi_pred_proj[multi_pred_proj['project'].isin(phe_project)]
            phe_df = phe_df[phe_df['disease_united'].isin([disease, 'healthy'])]

            case_df = phe_df[phe_df['disease_united'] == disease]
            ctrl_df = phe_df[phe_df['disease_united'] == 'healthy']

            if len(case_df) == 0 or len(ctrl_df) == 0:
                continue

            print(f"  {disease:<10} | Case: {len(case_df):<4} | Ctrl: {len(ctrl_df)}")

            # Calculate AUC and ROC
            y_true = np.concatenate([np.ones(len(case_df)), np.zeros(len(ctrl_df))])
            y_score = np.concatenate([case_df['prob'], ctrl_df['prob']])

            roc_auc = roc_auc_score(y_true, y_score)
            fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)

            # Plot individual disease ROC
            ax.plot(fpr, tpr, label=f'{disease}: AUC={roc_auc:.3f}', color=COLOR_LIST[color_idx], lw=2)
            color_idx += 1

        # Format subplot
        ax.plot([0, 1], [0, 1], ':', color='grey', alpha=0.5)
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.01)
        ax.set_aspect('equal')
        ax.set_xlabel('1-Specificity', fontsize=17)
        ax.set_ylabel('Sensitivity', fontsize=17)
        ax.set_title(class_name, fontsize=17)
        ax.legend(loc='lower right', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def plot_separate_roc_figures(pan_prob_list, pan_pred_concat):
    """
    Generate two separate plots:
    1. 5-Fold Cross Validation ROC + Mean ROC
    2. Concatenated predictions overall ROC
    """
    print("\n>>> Generating 5-Fold and Concatenated ROC plots...")

    plt.style.use('default')
    plt.rc('font', size=20)
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['font.family'] = 'Arial'

    # ---------------------------------------------------------
    # Plot 1: 5-Fold CV ROC + Mean ROC
    # ---------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(6, 6), facecolor='white')
    tprs, aucs = [], []
    mean_fpr = np.linspace(0, 1, 100)

    for i, df in enumerate(pan_prob_list):
        y_true = df['label'].astype(int).values
        y_score = df['prob'].values

        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

        ax1.plot(fpr, tpr, color='gray', lw=2, alpha=0.4, zorder=2, label=f'Fold {i + 1} (AUC = {roc_auc:.3f})')

    # Mean ROC
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    ax1.plot(mean_fpr, mean_tpr, color='#339DB5', lw=3.0, zorder=4,
             label=f'Mean ROC\nAUC = {mean_auc:.3f} $\pm$ {std_auc:.3f}')

    # Plot details
    ax1.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray', alpha=0.8, zorder=1)
    ax1.set(xlim=[-0.02, 1.02], ylim=[-0.02, 1.02], xticks=np.arange(0, 1.2, 0.2), yticks=np.arange(0, 1.2, 0.2))
    ax1.tick_params(axis='both', labelsize=20)
    for spine in ax1.spines.values(): spine.set_linewidth(2.0)
    ax1.set_xlabel('1-Specificity', fontsize=20)
    ax1.set_ylabel('Sensitivity', fontsize=20)
    ax1.legend(loc="lower right", fontsize=14, frameon=False)

    fig1.tight_layout()
    fig1.savefig(os.path.join(SAVE_DIR, 'cv_5fold_pan_roc.pdf'), dpi=800, bbox_inches='tight')
    plt.show()

    # ---------------------------------------------------------
    # Plot 2: Concatenated Overall ROC
    # ---------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(6, 6), facecolor='white')

    y_true_concat = pan_pred_concat['label'].astype(int).values
    y_score_concat = pan_pred_concat['prob'].values

    fpr_concat, tpr_concat, _ = roc_curve(y_true_concat, y_score_concat)
    auc_concat = auc(fpr_concat, tpr_concat)
    ci_lower, ci_upper = auc_ci(auc_concat, y_true_concat)

    concat_label = f"Concatenated\nAUC={auc_concat:.3f} ({ci_lower:.3f}-{ci_upper:.3f})"
    ax2.plot(fpr_concat, tpr_concat, color='#C9352B', lw=3.0, zorder=5, label=concat_label)

    # Plot details
    ax2.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray', alpha=0.8, zorder=1)
    ax2.set(xlim=[-0.02, 1.02], ylim=[-0.02, 1.02], xticks=np.arange(0, 1.2, 0.2), yticks=np.arange(0, 1.2, 0.2))
    ax2.tick_params(axis='both', labelsize=20)
    for spine in ax2.spines.values(): spine.set_linewidth(2.0)
    ax2.set_xlabel('1-Specificity', fontsize=20)
    ax2.set_ylabel('Sensitivity', fontsize=20)
    ax2.legend(loc="lower right", fontsize=14, frameon=False)

    fig2.tight_layout()
    fig2.savefig(os.path.join(SAVE_DIR, 'cv_concat_pan_roc.pdf'), dpi=800, bbox_inches='tight')
    plt.show()


# ==========================================
# Main Execution Block
# ==========================================
if __name__ == '__main__':
    # 1. Load and process data
    prob_list, pred_concat, merged_df = load_and_preprocess_data()

    # 2. Plot 2x2 grid of ROCs grouped by disease class
    plot_disease_classes_roc(
        multi_pred_proj=merged_df,
        save_path=os.path.join(SAVE_DIR, 'multidisease.sortabu.panprob.auc.5split.5.12.pdf')
    )

    # 3. Plot 5-fold CV ROC and concatenated ROC
    plot_separate_roc_figures(
        pan_prob_list=prob_list,
        pan_pred_concat=pred_concat
    )
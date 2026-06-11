import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, entropy
from sklearn.metrics import (
    roc_curve, auc, balanced_accuracy_score, confusion_matrix
)

warnings.filterwarnings('ignore')

# ==========================================
# Global visualization configuration (Top-tier journal style)
# ==========================================
plt.rc('font', size=20)
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['axes.linewidth'] = 2.0


# ==========================================
# Core metric calculation functions
# ==========================================
def calculate_mdi(individual_network, sample_list, taxa2id,
                  case_samples, ctrl_samples, bad_taxa, good_taxa):
    """
    Calculate the Microbial Dysbiosis Index (MDI) based on network features.
    Includes log transformation and weight allocation based on positive/negative biomarkers.

    Args:
        individual_network (np.ndarray): Individualized network matrix, shape (n_sample, n_taxa, n_taxa)
        sample_list (list): List of sample IDs
        taxa2id (dict): Mapping of taxa names to matrix indices
        case_samples (list): List of sample IDs in the disease group
        ctrl_samples (list): List of sample IDs in the control group
        bad_taxa (list): List of disease-promoting (PR-increasing) taxa
        good_taxa (list): List of disease-suppressing (PR-decreasing) taxa

    Returns:
        pd.DataFrame: DataFrame containing network features and MDI scores for each sample
    """
    bad_idx = [taxa2id[t] for t in bad_taxa if t in taxa2id]
    good_idx = [taxa2id[t] for t in good_taxa if t in taxa2id]
    mdi_features = []

    for i in range(individual_network.shape[0]):
        sample_feature_dict = {'sample_id': sample_list[i]}
        adj = np.abs(individual_network[i])
        adj[np.isnan(adj)] = 0
        adj[adj <= 0.01] = 0
        np.fill_diagonal(adj, 0)

        bad_internal = adj[np.ix_(bad_idx, bad_idx)].sum() / 2
        good_internal = adj[np.ix_(good_idx, good_idx)].sum() / 2
        total_edges = adj.sum() / 2

        sample_feature_dict.update({
            'f_bad_bad': bad_internal,
            'f_good_good': good_internal,
            'f_total_edges': total_edges,
            'f_bad_good': (bad_internal + 1e-5) / (good_internal + 1e-5),
        })
        mdi_features.append(sample_feature_dict)

    df_features = pd.DataFrame(mdi_features)
    df_features['group'] = df_features['sample_id'].apply(lambda x: 'case' if x in case_samples else 'ctrl')

    # Core logic for MDI calculation: log transformation
    df_features['MDI'] = np.log10(df_features['f_bad_good'] + 1e-5)
    return df_features


def calculate_cliffs_delta(case, ctrl):
    """
    Manually calculate Cliff's Delta effect size to evaluate the difference between two groups.
    """
    n1, n2 = len(case), len(ctrl)
    case, ctrl = np.array(case), np.array(ctrl)

    # Matrix broadcasting calculation
    diff = case[:, None] - ctrl
    greater = np.sum(diff > 0)
    less = np.sum(diff < 0)
    delta = (greater - less) / (n1 * n2)

    # Qualitative assessment
    if abs(delta) < 0.147:
        res = "Negligible"
    elif abs(delta) < 0.33:
        res = "Small"
    elif abs(delta) < 0.474:
        res = "Medium"
    else:
        res = "Large"

    return delta, res


def AUC_CI(auc_val, label, alpha=0.05):
    """Calculate the 95% Confidence Interval (CI) for AUC using the analytical method."""
    label = np.array(label)
    n1, n2 = np.sum(label == 1), np.sum(label == 0)
    q1 = auc_val / (2 - auc_val)
    q2 = (2 * auc_val ** 2) / (1 + auc_val)
    se = np.sqrt(
        (auc_val * (1 - auc_val) + (n1 - 1) * (q1 - auc_val ** 2) + (n2 - 1) * (q2 - auc_val ** 2)) / (n1 * n2))
    z_lower, z_upper = norm.interval(1 - alpha)
    return auc_val + z_lower * se, auc_val + z_upper * se


# ==========================================
# General evaluation module (Consolidated code)
# ==========================================
def evaluate_classification(df_features, dataset_name):
    """
    Calculate and print classification performance metrics (AUC, Confusion Matrix, Sensitivity, Specificity, etc.)
    """
    print(f"\n[{dataset_name}] Classification Performance Evaluation ---")
    nan_count = df_features['MDI'].isna().sum()
    print(f"Number of NaNs in MDI: {nan_count}")

    # Clean NaN data
    valid_mask = ~df_features['MDI'].isna()
    df_valid = df_features[valid_mask]
    y_true = (df_valid['group'] == 'case').astype(int)
    y_score = df_valid['MDI']

    # Calculate ROC and optimal threshold (Youden's Index)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    # Predict based on the optimal threshold
    y_pred = (y_score >= optimal_threshold).astype(int)

    # Calculate statistical metrics
    bacc = balanced_accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    acc = (tp + tn) / (tn + fp + fn + tp)

    print(f"AUC: {roc_auc:.4f} | Optimal Threshold: {optimal_threshold:.4f}")
    print(f"Accuracy: {acc:.4f} | Balanced Accuracy: {bacc:.4f}")
    print(f"Sensitivity (Sens): {sensitivity:.4f} | Specificity (Spec): {specificity:.4f}")
    print(f"Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}\n")
    print("=" * 50)

    return df_valid, roc_auc


# ==========================================
# Plotting module
# ==========================================
def plot_grouped_boxplot(df, disease, filename, custom_ylabel='Edge Weight', roc_auc=0):
    df_melt = pd.melt(df, id_vars=['group'],
                      value_vars=['f_bad_bad', 'f_good_good'],
                      var_name='feature', value_name='score')
    df_melt['group'] = pd.Categorical(df_melt['group'], categories=['case', 'ctrl'], ordered=True)

    fig, ax = plt.subplots(figsize=(7, 6), facecolor='white')
    custom_palette = ['#E44A33', '#4DBAD6']

    sns.boxplot(
        x='feature', y='score', hue='group', data=df_melt,
        palette=custom_palette, dodge=True, width=0.6,
        showfliers=True, fliersize=4,
        flierprops={'marker': 'o', 'markerfacecolor': 'black', 'markeredgecolor': 'black', 'markersize': 2.5,
                    'alpha': 1},
        boxprops={'edgecolor': 'black'}, whiskerprops={'color': 'black'},
        medianprops={'color': 'black'}, capprops={'color': 'black'}, ax=ax
    )

    ax.margins(x=0.1)
    ymin, ymax = ax.get_ylim()
    y_span = ymax - ymin
    y_data_max = df_melt['score'].max()
    y_pos = y_data_max + y_span * 0.05
    ax.set_ylim(ymin, y_pos + y_span * 0.1)

    offset = 0.15
    features = ['f_bad_bad', 'f_good_good']
    print(f"[{disease}] Grouped Boxplot Statistical Test Results:")

    for i, feat in enumerate(features):
        case_vals = df[df['group'] == 'case'][feat].dropna()
        ctrl_vals = df[df['group'] == 'ctrl'][feat].dropna()
        u_stat, p_value = stats.mannwhitneyu(case_vals, ctrl_vals)

        p_display = "P < 2.22e-308" if p_value == 0.0 else f"P = {p_value:.2e}"
        x1, x2 = i - offset, i + offset

        plt.plot([x1, x1, x2, x2], [y_pos * 0.96, y_pos, y_pos, y_pos * 0.96], color='black', lw=1.5)
        plt.text(i, y_pos * 1.02, p_display, ha='center', va='bottom', fontsize=18)

        print(f"  [{feat}] Mann-Whitney U = {u_stat:,.0f}, {p_display}")
        print(
            f"  Case group: {np.mean(case_vals):.2f} ± {np.std(case_vals):.2f} | Ctrl group: {np.mean(ctrl_vals):.2f} ± {np.std(ctrl_vals):.2f}")

    ax.set_xticklabels(['PR-increasing taxa', 'PR-decreasing taxa'], fontsize=20)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, [f'{disease}', f'Healthy'], loc='upper right', frameon=False, fontsize=20,
              bbox_to_anchor=(0.92, 0.88), handlelength=1.0, handleheight=0.8,
              handletextpad=0.3, borderpad=0.3, labelspacing=0.3)

    plt.ylabel(custom_ylabel)
    plt.xlabel('')
    plt.grid(True, alpha=0.3, linestyle='--', axis='y')
    plt.tight_layout()
    plt.savefig(filename, dpi=800, bbox_inches='tight')
    plt.show()


def plot_boxplot(df, col_name, disease, filename, custom_ylabel='MDI Score', roc_auc=0):
    df = df.sort_values('group')
    fig, ax = plt.subplots(figsize=(5, 6), facecolor='white')
    custom_palette = ['#E44A33', '#4DBAD6']

    sns.boxplot(
        x='group', y=col_name, data=df, hue='group',
        palette=custom_palette, order=['case', 'ctrl'], dodge=False, width=0.55,
        showfliers=True, fliersize=4,
        flierprops={'marker': 'o', 'markerfacecolor': 'black', 'markeredgecolor': 'black', 'markersize': 2.5,
                    'alpha': 1},
        legend=False, boxprops={'edgecolor': 'black'}, whiskerprops={'color': 'black'},
        medianprops={'color': 'black'}, capprops={'color': 'black'}, ax=ax
    )

    ax.margins(x=0.25)
    case_vals = df[df['group'] == 'case'][col_name]
    ctrl_vals = df[df['group'] == 'ctrl'][col_name]

    ymin, ymax = ax.get_ylim()
    y_span = ymax - ymin
    y_data_max = df[col_name].max()
    y_pos = y_data_max + y_span * 0.05
    ax.set_ylim(ymin, y_pos + y_span * 0.1)

    u_stat, p_value = stats.mannwhitneyu(case_vals, ctrl_vals)
    ks_stat, ks_p = stats.ks_2samp(case_vals, ctrl_vals)
    delta, effect_size = calculate_cliffs_delta(case_vals, ctrl_vals)

    p_display = "P < 2.22e-308" if p_value == 0.0 else f"P = {p_value:.2e}"

    plt.plot([0, 0, 1, 1], [y_pos * 0.98, y_pos, y_pos, y_pos * 0.98], color='black', lw=1.5)
    plt.text(0.5, y_pos, p_display, ha='center', va='bottom', fontsize=18)

    ax.set_xticklabels([f'{disease}\n(n={len(case_vals)})', f'Healthy\n(n={len(ctrl_vals)})'], fontsize=20)
    plt.ylabel(custom_ylabel)
    plt.xlabel('')
    plt.grid(True, alpha=0.3, linestyle='--', axis='y')
    plt.tight_layout()
    plt.savefig(filename, dpi=800, bbox_inches='tight')
    plt.show()

    print(f"[{disease}] Single Boxplot Statistical Test Results:")
    print(f"  Mann-Whitney U: U = {u_stat:,.0f}, p = {p_value:.2e}")
    print(f"  KS: D = {ks_stat:.4f}, p = {ks_p:.2e}")
    print(f"  Cliff's Delta: {delta:.4f} ({effect_size})")
    print("-" * 50)


def plot_multiple_roc_analytic_ci(data_dict, filename):
    fig, ax = plt.subplots(figsize=(6, 6), facecolor='white')
    colors = ['#C9352B', '#339DB5', '#E69F00', '#009E73']

    for i, (disease, df) in enumerate(data_dict.items()):
        color = colors[i % len(colors)]
        y_true = (df['group'] == 'case').astype(int).values
        y_score = df['MDI'].values

        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        ci_lower, ci_upper = AUC_CI(roc_auc, y_true)

        label_text = f"{disease}\nAUC={roc_auc:.3f} ({ci_lower:.3f}-{ci_upper:.3f})"
        ax.plot(fpr, tpr, color=color, linewidth=2.5, label=label_text, zorder=3)

    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray', alpha=0.8, zorder=1)

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis='both', labelsize=20)

    ax.set_xlabel('1-Specificity', fontsize=20)
    ax.set_ylabel('Sensitivity', fontsize=20)
    ax.legend(loc="lower right", fontsize=14, frameon=False)
    plt.tight_layout()
    plt.savefig(filename, dpi=800, bbox_inches='tight')
    plt.show()


# ==========================================
# Main program entry (Split by disease cohort)
# ==========================================
if __name__ == '__main__':
    # Base path and metadata loading
    path = 'tmp'
    fig_base_dir = '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/figures/finetune/11.25/networks/4.1/'
    label_df = pd.read_csv(
        '/bgi-seq-model-2/datasets/zhangkexin/meta_index/metaphlan4/fine-tune/curated_LiS_CRC_20204_v0908.train_test.phe',
        sep='\t', index_col=0)

    # ---------------------------------------------------------
    # 1. CRC Cohort Experiment
    # ---------------------------------------------------------
    print(">>> Running Experiment: CRC")
    crc_dir = '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results/network/1.8/'
    with open(crc_dir + 'crc.pretrain.emb.samples.list.txt', 'r') as f:
        crc_sample_list = [line.strip() for line in f.readlines()]

    crc_case_samples = [i for i in crc_sample_list if label_df.loc[i, 'disease_united'] == 'CRC']
    crc_ctrl_samples = [i for i in crc_sample_list if label_df.loc[i, 'disease_united'] == 'healthy']

    crc_network = np.load(crc_dir + 'crc.pretrain.emb.cosine.individual.npy')
    crc_taxa2id = pickle.load(open(crc_dir + 'crc.pretrain.emb.cosine.individual.index.pkl', 'rb'))

    # Load Biomarker
    crc_biomarker = pd.read_csv(
        '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results/features/11.25/sort.multidisease.mean/CRC.biomarker.txt',
        sep='\t', header=None)
    crc_biomarker['weight.abs'] = crc_biomarker[1].abs()
    crc_biomarker = crc_biomarker.sort_values('weight.abs', ascending=False)

    top30_crc_good_biomarker = list(crc_biomarker[crc_biomarker[1] < 0].iloc[:30, 0])
    top30_crc_bad_biomarker = list(crc_biomarker[crc_biomarker[1] > 0].iloc[:30, 0])
    CRC_top30_df_features = calculate_mdi(crc_network, crc_sample_list, crc_taxa2id, crc_case_samples, crc_ctrl_samples,
                                          top30_crc_bad_biomarker, top30_crc_good_biomarker)
    plot_grouped_boxplot(CRC_top30_df_features, 'CRC', f'{fig_base_dir}/{path}/CRC.top30.biomarker.edge.weight.pdf',
                         custom_ylabel='Edge Weight')

    crc_good_biomarker = list(crc_biomarker[crc_biomarker[1] < 0].iloc[:, 0])
    crc_bad_biomarker = list(crc_biomarker[crc_biomarker[1] > 0].iloc[:, 0])
    CRC_df_features = calculate_mdi(crc_network, crc_sample_list, crc_taxa2id, crc_case_samples, crc_ctrl_samples,
                                    crc_bad_biomarker, crc_good_biomarker)
    plot_boxplot(CRC_df_features, 'MDI', 'CRC', f'{fig_base_dir}/{path}/CRC.index.pdf', custom_ylabel='MDI Score')
    CRC_df_features, crc_auc = evaluate_classification(CRC_df_features, "CRC")

    # ---------------------------------------------------------
    # 2. CRC Validation Cohort (PRJNA758208 + Owen)
    # ---------------------------------------------------------
    print("\n>>> Running Experiment: CRC PRJNA758208 & Owen Validation")
    # PRJNA758208
    prj_label = pd.read_csv(
        '/bgi-seq-model-2/datasets/zhangkexin/meta_index/metaphlan4/validate/7.25/CRC/PRJNA758208/PRJNA758208.info',
        sep='\t', index_col=0)
    prj_case = list(prj_label[prj_label['Group'] == 'Disease'].index)
    prj_ctrl = list(prj_label[prj_label['Group'] == 'Health'].index)

    prj_dir = '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results/network/1.8/crc.PRJNA758208/'
    with open(prj_dir + 'samples.list.txt', 'r') as f:
        prj_samples = [line.strip() for line in f.readlines()]
    prj_network = np.load(prj_dir + 'pretrain.emb.individual.npy')
    prj_taxa2id = pickle.load(open(prj_dir + 'pretrain.emb.individual.index.pkl', 'rb'))

    prj_df = calculate_mdi(prj_network, prj_samples, prj_taxa2id, prj_case, prj_ctrl, crc_bad_biomarker,
                           crc_good_biomarker)

    # Owen
    with open(crc_dir + 'crc.owen.samples.txt', 'r') as f:
        owen_samples = [line.strip() for line in f.readlines()]
    owen_network = np.load(crc_dir + 'crc.owen.emb.individual.npy')
    owen_taxa2id = pickle.load(open(crc_dir + 'crc.owen.emb.individual.index.pkl', 'rb'))

    owen_df = calculate_mdi(owen_network, owen_samples, owen_taxa2id, owen_samples, owen_samples, crc_bad_biomarker,
                            crc_good_biomarker)

    val_df_features = pd.concat([prj_df, owen_df], axis=0)
    val_df_features, val_auc = evaluate_classification(val_df_features, "CRC Val (PRJNA758208+Owen)")
    plot_boxplot(val_df_features, 'MDI', 'CRC', f'{fig_base_dir}/{path}/crc.PRJNA758208.owen.index.pdf',
                 custom_ylabel='MDI Score')

    # ---------------------------------------------------------
    # 3. IBD Cohort Experiment
    # ---------------------------------------------------------
    print("\n>>> Running Experiment: IBD")
    ibd_dir = '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results/network/1.8/ibd/'
    ibd_network = np.load(ibd_dir + 'ibd.pretrain.emb.individual.npy')
    ibd_taxa2id = pickle.load(open(ibd_dir + 'ibd.pretrain.emb.individual.index.pkl', 'rb'))

    with open(ibd_dir + 'ibd.pretrain.emb.samples.list.txt', 'r') as f:
        ibd_samples = [line.strip() for line in f.readlines()]

    ibd_case_samples = [i for i in ibd_samples if label_df.loc[i, 'disease_united'] == 'IBD']
    ibd_ctrl_samples = [i for i in ibd_samples if label_df.loc[i, 'disease_united'] == 'healthy']

    ibd_biomarker = pd.read_csv(
        '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results/features/11.25/sort.multidisease.mean/IBD.biomarker.txt',
        sep='\t', header=None)
    ibd_biomarker['weight.abs'] = ibd_biomarker[1].abs()
    ibd_biomarker = ibd_biomarker.sort_values('weight.abs', ascending=False)

    top30_ibd_good_biomarker = list(ibd_biomarker[ibd_biomarker[1] < 0].iloc[:30, 0])
    top30_ibd_bad_biomarker = list(ibd_biomarker[ibd_biomarker[1] > 0].iloc[:30, 0])
    IBD_top30_df_features = calculate_mdi(ibd_network, ibd_samples, ibd_taxa2id, ibd_case_samples, ibd_ctrl_samples,
                                          top30_ibd_bad_biomarker, top30_ibd_good_biomarker)
    plot_grouped_boxplot(IBD_top30_df_features, 'IBD', f'{fig_base_dir}/{path}/IBD.top30.biomarker.edge.weight.pdf',
                         custom_ylabel='Edge Weight')

    ibd_good_biomarker = list(ibd_biomarker[ibd_biomarker[1] < 0].iloc[:, 0])
    ibd_bad_biomarker = list(ibd_biomarker[ibd_biomarker[1] > 0].iloc[:, 0])
    IBD_df_features = calculate_mdi(ibd_network, ibd_samples, ibd_taxa2id, ibd_case_samples, ibd_ctrl_samples,
                                    ibd_bad_biomarker, ibd_good_biomarker)
    plot_boxplot(IBD_df_features, 'MDI', 'IBD', f'{fig_base_dir}/{path}/IBD.index.pdf', custom_ylabel='MDI Score')
    IBD_df_features, ibd_auc = evaluate_classification(IBD_df_features, "IBD")

    # ---------------------------------------------------------
    # 4. IBD Validation Cohort (Ning)
    # ---------------------------------------------------------
    print("\n>>> Running Experiment: IBD Ning Validation")
    ning_network = np.load(ibd_dir + 'Ning.pretrain.emb.individual.npy')
    ning_taxa2id = pickle.load(open(ibd_dir + 'Ning.pretrain.emb.individual.index.pkl', 'rb'))

    with open(ibd_dir + 'Ning.pretrain.emb.samples.list.txt', 'r') as f:
        ning_samples = [line.strip() for line in f.readlines()]

    ning_label = pd.read_csv(
        '/bgi-seq-model-2/datasets/zhangkexin/meta_index/metaphlan4/validate/7.25/IBD/13_Ning_2023/13_Ning_2023.info',
        sep='\t', index_col=0)
    ning_case = list(ning_label[ning_label['Group'] == 'Disease'].index)
    ning_ctrl = list(ning_label[ning_label['Group'] == 'Health'].index)

    ning_profile = pd.read_csv(
        '/bgi-seq-model-2/datasets/zhangkexin/meta_index/metaphlan4/validate/7.25/IBD/13_Ning_2023/13_Ning_2023.profile',
        sep='\t', index_col=0).T
    index = [i for i in range(len(ning_profile.columns)) if
             ning_profile.columns[i].split('|')[-1].split('__')[0] == 's']
    ning_profile = ning_profile.iloc[:, index]
    ning_profile.columns = [i.split('|')[-1].replace('__', '-').replace(' ', '-').replace('_', '-').lower() for i in
                            ning_profile.columns]

    ning_profile = ning_profile.loc[ning_samples, :]
    shannon_df = pd.DataFrame({
        'sample_id': ning_samples,
        'shannon': ning_profile.apply(lambda x: entropy(x[x > 0]), axis=1)
    })

    ning_df_features = calculate_mdi(ning_network, ning_samples, ning_taxa2id, ning_case, ning_ctrl, ibd_bad_biomarker,
                                     ibd_good_biomarker)
    ning_df_features = pd.merge(ning_df_features, shannon_df, on='sample_id', how='left')
    ning_df_features, ning_auc = evaluate_classification(ning_df_features, "IBD Val (Ning)")
    plot_boxplot(ning_df_features, 'MDI', 'IBD', f'{fig_base_dir}/{path}/IBD.ning.mdi.index.pdf',
                 custom_ylabel='MDI Score')

    # ---------------------------------------------------------
    # 5. Pan-disease Cohort Experiment
    # ---------------------------------------------------------
    print("\n>>> Running Experiment: PANDISEASE")
    pan_dir = '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results/network/1.8/pandisease/'
    pan_samples = list(pd.read_csv(pan_dir + 'pandisease.split1.valtest.samples.csv')['samples'])

    pan_case_samples = [i for i in pan_samples if label_df.loc[i, 'disease_united'] != 'healthy']
    pan_ctrl_samples = [i for i in pan_samples if label_df.loc[i, 'disease_united'] == 'healthy']

    pan_network = np.load(pan_dir + 'pandisease.split1.valtest.pretrain.emb.cosine.individual.npy')
    pan_taxa2id = pickle.load(
        open(pan_dir + 'pandisease.split1.valtest.pretrain.emb.cosine.individual.index.pkl', 'rb'))

    top30_core_good_biomarker = ['s-' + i.lower().replace(" ", "-") for i in
                                 list(pd.read_csv(pan_dir + 'top30.negative.o3.csv', index_col=0).index)]
    top30_core_bad_biomarker = ['s-' + i.lower().replace(" ", "-") for i in
                                list(pd.read_csv(pan_dir + 'top30.positive.o3.csv', index_col=0).index)]

    pan_biomarker = pd.read_csv(
        '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results/features/11.25/sort.multidisease.mean/pandisease.biomarker.txt',
        sep='\t', header=None)
    pan_biomarker['weight.abs'] = pan_biomarker[1].abs()
    pan_biomarker = pan_biomarker.sort_values('weight.abs', ascending=False)

    pan_good_biomarker = list(pan_biomarker[pan_biomarker[1] < 0].iloc[:, 0])
    pan_bad_biomarker = list(pan_biomarker[pan_biomarker[1] > 0].iloc[:, 0])

    filtered_samples = label_df[((label_df['From'] == 'LiS') & (label_df['project'] == 'QinJ_2012'))].index.tolist()

    PAN_df_features = calculate_mdi(pan_network, pan_samples, pan_taxa2id, pan_case_samples, pan_ctrl_samples,
                                    top30_core_bad_biomarker, top30_core_good_biomarker)
    PAN_full_df_features = calculate_mdi(pan_network, pan_samples, pan_taxa2id, pan_case_samples, pan_ctrl_samples,
                                         pan_bad_biomarker, pan_good_biomarker)

    PAN_df_features = PAN_df_features[~PAN_df_features['sample_id'].isin(filtered_samples)]
    PAN_full_df_features = PAN_full_df_features[~PAN_full_df_features['sample_id'].isin(filtered_samples)]

    PAN_df_features, pan_core_auc = evaluate_classification(PAN_df_features, "Pandisease (Core)")
    PAN_full_df_features, pan_full_auc = evaluate_classification(PAN_full_df_features, "Pandisease (Full)")

    plot_boxplot(PAN_df_features, 'MDI', 'Disease', f'{fig_base_dir}/{path}/pandisease.core.index.pdf',
                 custom_ylabel='MDI Score')
    plot_boxplot(PAN_full_df_features, 'MDI', 'Disease', f'{fig_base_dir}/{path}/pandisease.full.index.pdf',
                 custom_ylabel='MDI Score')

    # ---------------------------------------------------------
    # 6. Integrated Multi-disease ROC Curve Plotting
    # ---------------------------------------------------------
    print("\n>>> Plotting Integrated ROC Curve")
    data_dict = {
        'CRC': CRC_df_features,
        'IBD': IBD_df_features,
        'Pandisease (Full)': PAN_full_df_features,
        'Pandisease (Core)': PAN_df_features
    }
    plot_multiple_roc_analytic_ci(data_dict, f'{fig_base_dir}/{path}/combined_roc.pdf')

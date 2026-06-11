import os
import pickle
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc

# Import the core MDI calculation function as requested
from network_dysbiosis_index import calculate_mdi

import warnings

warnings.filterwarnings('ignore')

# ==========================================
# Global Plotting Configuration
# ==========================================
plt.style.use('default')
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 18,
    'pdf.fonttype': 42,
    'axes.linewidth': 2.0
})


# ==========================================
# Experiment 1: Saturation AUC Analysis
# ==========================================
def experiment_saturation_auc(individual_network, sample_list, taxa2id,
                              case_samples, ctrl_samples, sorted_biomarkers_df):
    """
    Gradually increase the number of biomarkers based on feature importance
    to observe when the AUC reaches a saturation plateau.
    """
    auc_records = []
    step_range = range(10, len(sorted_biomarkers_df) + 1, 10)

    print(">>> Running Saturation Analysis...")
    for k in tqdm(step_range, desc="Saturation Steps"):
        current_top = sorted_biomarkers_df.iloc[:k]
        sub_bad = list(current_top[current_top[1] > 0][0])
        sub_good = list(current_top[current_top[1] < 0][0])

        # Skip if one of the groups is empty at this step
        if len(sub_bad) == 0 or len(sub_good) == 0:
            continue

        df_feat = calculate_mdi(individual_network, sample_list, taxa2id,
                                case_samples, ctrl_samples, sub_bad, sub_good)

        y_true = (df_feat['group'] == 'case').astype(int)
        fpr, tpr, _ = roc_curve(y_true, df_feat['MDI'])
        current_auc = auc(fpr, tpr)
        auc_records.append({'k': k, 'auc': current_auc})

    return pd.DataFrame(auc_records)


# ==========================================
# Experiment 2: Permutation Test
# ==========================================
def experiment_permutation_test(individual_network, sample_list, taxa2id,
                                case_samples, ctrl_samples,
                                n_bad, n_good, real_auc, all_taxa, iterations=100):
    """
    Randomly select the same number of taxa as biomarkers to calculate the null AUC distribution,
    demonstrating the specificity and significance of the true biomarkers.
    """
    null_aucs = []

    print(">>> Running Permutation Test...")
    for _ in tqdm(range(iterations), desc="Permutation Iterations"):
        # Randomly sample taxa for bad and good roles
        random_taxa = random.sample(all_taxa, n_bad + n_good)
        rand_bad = random_taxa[:n_bad]
        rand_good = random_taxa[n_bad:]

        df_rand = calculate_mdi(individual_network, sample_list, taxa2id,
                                case_samples, ctrl_samples, rand_bad, rand_good)

        y_true = (df_rand['group'] == 'case').astype(int)
        fpr, tpr, _ = roc_curve(y_true, df_rand['MDI'])
        a = auc(fpr, tpr)

        # Take max to account for inverse classification (if random features flip the prediction)
        null_aucs.append(max(a, 1 - a))

    # Calculate empirical P-value
    p_value = np.sum(np.array(null_aucs) >= real_auc) / iterations
    print(f">>> Permutation P-value: {p_value:.4f}")

    return null_aucs, p_value


# ==========================================
# Unified Execution & Plotting Pipeline
# ==========================================
def analyze_and_plot_disease(disease_name, sample_list, label_df, network_npy, taxa2id_pkl, biomarker_txt,
                             save_pdf_path):
    """
    Encapsulates the entire workflow to prevent global variable shadowing.
    Loads data, computes baseline AUC, runs experiments, and exports the plot.
    """
    print(f"\n{'=' * 50}\nEvaluating Cohort: {disease_name}\n{'=' * 50}")

    # 1. Filter Case/Control Samples
    case_samples = [i for i in sample_list if i in label_df.index and label_df.loc[i, 'disease_united'] == disease_name]
    ctrl_samples = [i for i in sample_list if i in label_df.index and label_df.loc[i, 'disease_united'] == 'healthy']

    # 2. Load Network and Indices
    individual_network = np.load(network_npy)
    with open(taxa2id_pkl, 'rb') as f:
        taxa2id = pickle.load(f)

    # 3. Load and Format Biomarkers
    biomarker_df = pd.read_csv(biomarker_txt, sep='\t', header=None)
    biomarker_df['weight.abs'] = biomarker_df[1].abs()
    biomarker_df = biomarker_df.sort_values('weight.abs', ascending=False)

    # Ensure biomarkers exist in the network mapping
    biomarker_df = biomarker_df.loc[biomarker_df[0].isin(list(taxa2id.keys()))]
    good_biomarkers = list(biomarker_df[biomarker_df[1] < 0].iloc[:, 0])
    bad_biomarkers = list(biomarker_df[biomarker_df[1] > 0].iloc[:, 0])

    # 4. Calculate Baseline MDI & True AUC
    df_features = calculate_mdi(individual_network, sample_list, taxa2id,
                                case_samples, ctrl_samples, bad_biomarkers, good_biomarkers)
    y_true = (df_features['group'] == 'case').astype(int)
    fpr, tpr, _ = roc_curve(y_true, df_features['MDI'])
    real_auc = auc(fpr, tpr)
    print(f">>> [{disease_name}] True Baseline AUC: {real_auc:.4f}")

    # 5. Run Experiments
    sat_df = experiment_saturation_auc(individual_network, sample_list, taxa2id,
                                       case_samples, ctrl_samples, biomarker_df)

    null_aucs, p_val = experiment_permutation_test(
        individual_network, sample_list, taxa2id, case_samples, ctrl_samples,
        len(bad_biomarkers), len(good_biomarkers), real_auc, list(biomarker_df[0]), iterations=100
    )

    # 6. Plotting
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6, 10))

    # Top Plot: Saturation curve
    ax1 = axes[0]
    ax1.plot(sat_df['k'], sat_df['auc'], marker='o', color='#E58579', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Top Biomarkers', fontsize=18)
    ax1.set_ylabel('AUC', fontsize=18)
    ax1.set_ylim(0, 1.05)
    ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # ax1.set_title(f'{disease_name} Saturation Analysis', fontsize=18, fontweight='bold', pad=15)

    # Bottom Plot: Permutation test
    ax2 = axes[1]
    sns.histplot(null_aucs, kde=True, ax=ax2, color='#8AB1D2', edgecolor="black")
    ax2.axvline(real_auc, color='#E44A33', linestyle='--', linewidth=2.5, label=f'True AUC: {real_auc:.4f}')
    # ax2.set_title(f'Permutation Test (P = {p_val:.4f})', fontsize=18, fontweight='bold', pad=15)
    ax2.set_xlabel('AUC', fontsize=18)
    ax2.set_ylabel('Frequency', fontsize=18)
    ax2.legend(frameon=False, fontsize=14)

    plt.tight_layout()

    # Save and display
    os.makedirs(os.path.dirname(save_pdf_path), exist_ok=True)
    # plt.savefig(save_pdf_path, dpi=800, bbox_inches='tight')
    plt.show()


# ==========================================
# Main Execution Block
# ==========================================
if __name__ == '__main__':
    # Base configuration
    LABEL_PHE_PATH = '/bgi-seq-model-2/datasets/zhangkexin/meta_index/metaphlan4/fine-tune/curated_LiS_CRC_20204_v0908.train_test.phe'
    FEATURE_DIR = '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results/features/11.25/sort.multidisease.mean/'
    NETWORK_DIR = '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results/network/1.8/'
    SAVE_BASE_DIR = '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/figures/finetune/11.25/networks/4.1/abs/'

    # Load unified phenotype labels
    label_dataframe = pd.read_csv(LABEL_PHE_PATH, sep='\t', index_col=0)

    # ---------------------------------------------------------
    # 1. Execute CRC Analysis
    # ---------------------------------------------------------
    with open(NETWORK_DIR + '/crc.pretrain.emb.samples.list.txt', 'r') as f:
        crc_samples = [line.strip() for line in f.readlines()]

    analyze_and_plot_disease(
        disease_name='CRC',
        sample_list=crc_samples,
        label_df=label_dataframe,
        network_npy=NETWORK_DIR + 'crc.pretrain.emb.cosine.individual.npy',
        taxa2id_pkl=NETWORK_DIR + 'crc.pretrain.emb.cosine.individual.index.pkl',
        biomarker_txt=FEATURE_DIR + 'CRC.biomarker.txt',
        save_pdf_path=SAVE_BASE_DIR + 'crc.saturation.permutation.combined.5.6.pdf'
    )

    # ---------------------------------------------------------
    # 2. Execute IBD Analysis
    # ---------------------------------------------------------
    with open(NETWORK_DIR + 'ibd/ibd.pretrain.emb.samples.list.txt', 'r') as f:
        ibd_samples = [line.strip() for line in f.readlines()]

    analyze_and_plot_disease(
        disease_name='IBD',
        sample_list=ibd_samples,
        label_df=label_dataframe,
        network_npy=NETWORK_DIR + 'ibd/ibd.pretrain.emb.individual.npy',
        taxa2id_pkl=NETWORK_DIR + 'ibd/ibd.pretrain.emb.individual.index.pkl',
        biomarker_txt=FEATURE_DIR + 'IBD.biomarker.txt',
        save_pdf_path=SAVE_BASE_DIR + 'ibd.saturation.permutation.combined.5.6.pdf'
    )
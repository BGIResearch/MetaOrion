# @Date    : 2026/6/8
# @Email   : zhangkexin2@genomics.cn

import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, entropy

# Import the core MDI calculation function from the dysbiosis module
from network_dysbiosis_index import calculate_mdi

import warnings

warnings.filterwarnings('ignore')


# ==========================================
# Module 1: MDI vs Shannon Correlation Analysis
# ==========================================
def load_biomarker(path):
    """
    Load biomarker files and separate them into disease-promoting (bad)
    and disease-suppressing (good) taxa based on their weights.
    """
    df = pd.read_csv(path, sep='\t', header=None)
    df['abs'] = df[1].abs()
    df = df.sort_values('abs', ascending=False)

    good = list(df[df[1] < 0].iloc[:, 0])
    bad = list(df[df[1] > 0].iloc[:, 0])
    return good, bad


def calculate_shannon(X_sub):
    """
    Calculate the Shannon diversity index for a given subset of microbiome profiles.
    """
    return pd.DataFrame({
        'sample_id': X_sub.index,
        'shannon': X_sub.apply(lambda x: entropy(x[x > 0]), axis=1)
    })


def calculate_mdi_shannon(individual_network, sample_list, taxa2id,
                          case_samples, ctrl_samples,
                          good_biomarker, bad_biomarker, X):
    """
    Calculate both the MDI score and Shannon diversity index, returning a merged DataFrame.
    """
    # Calculate MDI using the imported function
    df_mdi = calculate_mdi(
        individual_network, sample_list, taxa2id,
        case_samples, ctrl_samples,
        bad_biomarker, good_biomarker
    )

    # Calculate Shannon index
    shannon_df = calculate_shannon(X.loc[case_samples + ctrl_samples, :])

    return pd.merge(df_mdi, shannon_df, on='sample_id', how='left')


def plot_mdi_shannon(df, title, save_path):
    """
    Plot a scatter plot with regression line to show the Spearman correlation
    between Shannon diversity and MDI scores.
    """
    plot_df = df.dropna(subset=['shannon', 'MDI'])
    rho, pval = spearmanr(plot_df['shannon'], plot_df['MDI'])

    print(f'>>> Correlation for {title}')
    print(f'  Spearman r = {rho:.4f}')
    print(f'  Spearman p = {pval:.4e}')

    plt.style.use('default')
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 18, 'pdf.fonttype': 42})

    fig, ax = plt.subplots(figsize=(5.8, 5.2))

    sns.scatterplot(
        data=plot_df, x='shannon', y='MDI', hue='group',
        palette={'case': '#C9352B', 'ctrl': '#339DB5'},
        s=42, alpha=0.8, ax=ax
    )

    sns.regplot(
        data=plot_df, x='shannon', y='MDI',
        scatter=False, color='#666666', ax=ax
    )

    ax.set_title(f'Spearman r = {rho:.3f}\np = {pval:.2e}')
    ax.set_xlabel('Shannon diversity')
    ax.set_ylabel('MDI Score')
    ax.legend(frameon=False, title='Group')

    sns.despine(ax=ax)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=800)
    plt.show()

    return {'rho': rho, 'pval': pval, 'n': len(plot_df)}


def run_correlation_pipeline(disease_name, sample_list, label_df,
                             individual_network_path, taxa2id_path,
                             biomarker_path, X, save_path,
                             case_condition=None, remove_projects=None, remove_samples=None):
    """
    End-to-end pipeline wrapper for calculating features and plotting correlation.
    """
    print(f"\n==========================================")
    print(f"Running Correlation Pipeline: {disease_name}")
    print(f"==========================================")

    if remove_projects is not None:
        label_df = label_df[~label_df['project'].isin(remove_projects)]

    case_samples, ctrl_samples = [], []

    for i in sample_list:
        if i not in label_df.index:
            continue
        disease = label_df.loc[i, 'disease_united']

        if case_condition:
            if disease == case_condition:
                case_samples.append(i)
            elif disease == 'healthy':
                ctrl_samples.append(i)
        else:
            if disease != 'healthy':
                case_samples.append(i)
            elif disease == 'healthy':
                ctrl_samples.append(i)

    individual_network = np.load(individual_network_path)
    with open(taxa2id_path, 'rb') as f:
        taxa2id = pickle.load(f)

    good_biomarker, bad_biomarker = load_biomarker(biomarker_path)

    df = calculate_mdi_shannon(
        individual_network, sample_list, taxa2id,
        case_samples, ctrl_samples,
        good_biomarker, bad_biomarker, X
    )

    if remove_samples is not None:
        df = df[~df['sample_id'].isin(remove_samples)]

    result = plot_mdi_shannon(df=df, title=disease_name, save_path=save_path)
    return df, result


# ==========================================
# Module 2: AUC Comparison Line Plots
# ==========================================
def load_auc_table(path: str, label_col: str) -> pd.DataFrame:
    """
    Load and filter the statistical TSV files to compare MDI AUC vs Shannon AUC.
    """
    df = pd.read_csv(path, sep='\t')

    # Note: Updated 'MNDI_AUC' to 'MDI_AUC' to match the standard nomenclature
    keep_cols = [label_col, 'MDI_AUC', 'Shannon_AUC', 'n_case', 'n_ctrl']

    # Use existing columns if user TSV hasn't been updated yet, but prefer MDI_AUC
    if 'MNDI_AUC' in df.columns:
        df.rename(columns={'MNDI_AUC': 'MDI_AUC'}, inplace=True)

    df = df[keep_cols].copy()
    df = df.dropna(subset=['MDI_AUC', 'Shannon_AUC'])
    df = df[(df['n_case'] > 0) & (df['n_ctrl'] > 0)].copy()
    df = df.sort_values('MDI_AUC', ascending=False).reset_index(drop=True)
    return df


def plot_auc_lines(df: pd.DataFrame, label_col: str, title: str, output_path: str) -> None:
    """
    Plot a line chart comparing the AUC of MDI against Shannon diversity.
    """
    plt.rc('font', size=22)
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['pdf.fonttype'] = 42
    sns.set_style('ticks')

    fig_width = max(10, 0.65 * len(df) + 3)
    fig, ax = plt.subplots(figsize=(fig_width, 6.5), facecolor='white')

    x = range(len(df))
    ax.plot(x, df['MDI_AUC'], color='#C9352B', marker='o', linewidth=3.0, markersize=8, label='MDI')
    ax.plot(x, df['Shannon_AUC'], color='#339DB5', marker='s', linewidth=3.0, markersize=7, label='Shannon')

    ax.fill_between(x, df['MDI_AUC'], df['Shannon_AUC'], color='#E9EEF6', alpha=0.45)
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.8)

    ax.set_xticks(list(x))
    ax.set_xticklabels(df[label_col], rotation=50, ha='right')
    ax.set_ylabel('AUC')
    ax.set_xlabel('')
    ax.set_title(title, pad=18, fontsize=24)
    ax.set_ylim(0.45, min(1.0, max(df[['MDI_AUC', 'Shannon_AUC']].max()) + 0.06))

    for idx, row in df.iterrows():
        diff = row['MDI_AUC'] - row['Shannon_AUC']
        ax.text(
            idx, max(row['MDI_AUC'], row['Shannon_AUC']) + 0.015,
            f'{diff:+.02f}', ha='center', va='bottom', fontsize=12,
            color='#7A1F1A' if diff > 0 else '#1E5E6F'
        )

    ax.legend(frameon=False, loc='upper right', fontsize=18)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.25)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=800, bbox_inches='tight')
    plt.show()


# ==========================================
# Main Execution Block
# ==========================================
if __name__ == '__main__':

    # ---------------------------------------------------------
    # 1. Global Data Loading
    # ---------------------------------------------------------
    print(">>> Loading Global Microbiome Profile and Labels...")
    X = pd.read_csv(
        '/bgi-seq-model-2/datasets/zhangkexin/meta_index/metaphlan4/fine-tune/curated_LiS_CRC_20204_v1121.train_test.profile',
        sep='\t', index_col=0).transpose()
    species_indices = [i for i in range(len(X.columns)) if X.columns[i].split('|')[-1].startswith('s__')]
    X = X.iloc[:, species_indices]
    X.columns = [col.split('|')[-1].replace('__', '-').replace(' ', '-').replace('_', '-').lower() for col in X.columns]

    label_df = pd.read_csv(
        '/bgi-seq-model-2/datasets/zhangkexin/meta_index/metaphlan4/fine-tune/curated_LiS_CRC_20204_v0908.train_test.phe',
        sep='\t', index_col=0)

    SAVE_DIR_SCATTER = '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/figures/finetune/11.25/networks/4.1/tmp/'

    # ---------------------------------------------------------
    # 2. Run Pan-Disease Correlation Analysis
    # ---------------------------------------------------------
    pan_dir = '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results/network/1.8/pandisease/'
    pan_samples = list(pd.read_csv(pan_dir + 'pandisease.split1.valtest.samples.csv')['samples'])
    filtered_samples = label_df[((label_df['From'] == 'LiS') & (label_df['project'] == 'QinJ_2012'))].index.tolist()

    df_pan, result_pan = run_correlation_pipeline(
        disease_name='PANDISEASE',
        sample_list=pan_samples,
        label_df=label_df,
        individual_network_path=pan_dir + 'pandisease.split1.valtest.pretrain.emb.cosine.individual.npy',
        taxa2id_path=pan_dir + 'pandisease.split1.valtest.pretrain.emb.cosine.individual.index.pkl',
        biomarker_path='/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results/features/11.25/sort.multidisease.mean/pandisease.biomarker.txt',
        X=X,
        save_path=SAVE_DIR_SCATTER + 'pandisease.shannon_vs_mdi.pdf',
        remove_samples=filtered_samples
    )

    # ---------------------------------------------------------
    # 3. Run CRC Correlation Analysis
    # ---------------------------------------------------------
    crc_dir = '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results/network/1.8/'
    with open(crc_dir + 'crc.pretrain.emb.samples.list.txt', 'r') as f:
        crc_samples = [line.strip() for line in f.readlines()]

    df_crc, result_crc = run_correlation_pipeline(
        disease_name='CRC',
        sample_list=crc_samples,
        label_df=label_df,
        individual_network_path=crc_dir + 'crc.pretrain.emb.cosine.individual.npy',
        taxa2id_path=crc_dir + 'crc.pretrain.emb.cosine.individual.index.pkl',
        biomarker_path='/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results/features/11.25/sort.multidisease.mean/CRC.biomarker.txt',
        X=X,
        save_path=SAVE_DIR_SCATTER + 'CRC.shannon_vs_mdi.pdf',
        case_condition='CRC',
    )

    # ---------------------------------------------------------
    # 4. Run IBD Correlation Analysis (Bug Fixed)
    # ---------------------------------------------------------
    ibd_dir = '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results/network/1.8/ibd/'
    with open(ibd_dir + 'ibd.pretrain.emb.samples.list.txt', 'r') as f:
        ibd_samples = [line.strip() for line in f.readlines()]

    df_ibd, result_ibd = run_correlation_pipeline(
        disease_name='IBD',
        sample_list=ibd_samples,
        label_df=label_df,
        individual_network_path=ibd_dir + 'ibd.pretrain.emb.individual.npy',
        taxa2id_path=ibd_dir + 'ibd.pretrain.emb.individual.index.pkl',
        biomarker_path='/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results/features/11.25/sort.multidisease.mean/IBD.biomarker.txt',
        X=X,
        save_path=SAVE_DIR_SCATTER + 'IBD.shannon_vs_mdi.pdf',
        case_condition='IBD',
    )

    # ---------------------------------------------------------
    # 5. AUC Comparison Lines (MDI vs Shannon)
    # ---------------------------------------------------------
    print("\n==========================================")
    print("Running AUC Comparison Plots...")
    print("==========================================")

    BASE_DIR_AUC = '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/figures/finetune/11.25/networks/4.1/tmp/'

    PLOT_CONFIGS = [
        {
            'input': os.path.join(BASE_DIR_AUC, 'pandisease.disease.case.ctrl.stats.tsv'),
            'output': os.path.join(BASE_DIR_AUC, 'pandisease.disease.case.ctrl.auc.line.pdf'),
            'label_col': 'disease',
            'title': 'Pan-disease cohorts',
        },
        {
            'input': os.path.join(BASE_DIR_AUC, 'CRC.project.individual.case.ctrl.project.stats.tsv'),
            'output': os.path.join(BASE_DIR_AUC, 'CRC.project.individual.case.ctrl.auc.line.pdf'),
            'label_col': 'project',
            'title': 'CRC cohorts',
        },
        {
            'input': os.path.join(BASE_DIR_AUC, 'IBD.project.individual.case.ctrl.project.stats.tsv'),
            'output': os.path.join(BASE_DIR_AUC, 'IBD.project.individual.case.ctrl.auc.line.pdf'),
            'label_col': 'project',
            'title': 'IBD cohorts',
        },
    ]

    for cfg in PLOT_CONFIGS:
        if not os.path.exists(cfg['input']):
            print(f">>> Missing input file, skipping: {cfg['input']}")
            continue

        df_auc = load_auc_table(cfg['input'], cfg['label_col'])
        if df_auc.empty:
            print(f">>> Skip empty or invalid file: {cfg['input']}")
            continue

        plot_auc_lines(df_auc, cfg['label_col'], cfg['title'], cfg['output'])
        print(f">>> Saved Line Plot: {cfg['output']}")
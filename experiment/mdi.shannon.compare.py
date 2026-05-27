import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


PROFILE_PATH = '/bgi-seq-model-2/datasets/zhangkexin/meta_index/metaphlan4/fine-tune/curated_LiS_CRC_20204_v1121.train_test.profile'
LABEL_PATH = '/bgi-seq-model-2/datasets/zhangkexin/meta_index/metaphlan4/fine-tune/curated_LiS_CRC_20204_v0908.train_test.phe'
PANDISEASE_DIR = '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results/network/1.8/pandisease/'
FULL_BIOMARKER_PATH = '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results/features/11.25/sort.multidisease.mean/pandisease.biomarker.txt'

OUTPUT_DIR = '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results/shannon_compare'
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'Arial'


def normalize_taxa_name(name: str) -> str:
    name = str(name).strip()
    if '|s__' in name:
        name = name.split('|')[-1]
    name = name.replace('__', '-').replace(' ', '-').replace('_', '-').lower()
    if not name.startswith('s-'):
        if name.startswith('s__'):
            name = 's-' + name[3:]
        else:
            name = 's-' + name
    return name


def load_profile(profile_path: str) -> pd.DataFrame:
    X = pd.read_csv(profile_path, sep='\t', index_col=0).T
    species_idx = []
    for i in range(len(X.columns)):
        tail = X.columns[i].split('|')[-1]
        if tail.split('__')[0] == 's':
            species_idx.append(i)
    X = X.iloc[:, species_idx]
    X.columns = [normalize_taxa_name(col) for col in X.columns]
    X.index = [idx.split('.mp4')[0] for idx in X.index]
    X = X.groupby(level=0, axis=1).sum()
    return X


def load_pandisease_samples() -> list:
    path = os.path.join(PANDISEASE_DIR, 'pandisease.split1.valtest.samples.csv')
    return list(pd.read_csv(path)['samples'])


def load_mdi_strategies_and_network():
    taxa2id = pickle.load(open(os.path.join(PANDISEASE_DIR, 'pandisease.split1.valtest.pretrain.emb.cosine.individual.index.pkl'), 'rb'))

    top30_core_good = [
        normalize_taxa_name(i) for i in pd.read_csv(os.path.join(PANDISEASE_DIR, 'top30.negative.o3.csv'), index_col=0).index
    ]
    top30_core_bad = [
        normalize_taxa_name(i) for i in pd.read_csv(os.path.join(PANDISEASE_DIR, 'top30.positive.o3.csv'), index_col=0).index
    ]
    strategy_b = sorted(set(top30_core_good + top30_core_bad))

    pan_biomarker = pd.read_csv(FULL_BIOMARKER_PATH, sep='\t', header=None)
    pan_biomarker['weight_abs'] = pan_biomarker[1].abs()
    pan_biomarker = pan_biomarker.sort_values('weight_abs', ascending=False)
    good_biomarker = [normalize_taxa_name(i) for i in pan_biomarker.loc[pan_biomarker[1] < 0, 0]]
    bad_biomarker = [normalize_taxa_name(i) for i in pan_biomarker.loc[pan_biomarker[1] > 0, 0]]
    strategy_c = sorted(set(good_biomarker + bad_biomarker))

    individual_network = np.load(os.path.join(PANDISEASE_DIR, 'pandisease.split1.valtest.pretrain.emb.cosine.individual.npy'))

    return {
        'Top30core_biomarker': strategy_b,
        'Full_biomarker': strategy_c,
    }, {
        'MDI_top30core': {
            'bad_taxa': top30_core_bad,
            'good_taxa': top30_core_good,
        },
        'MDI_full': {
            'bad_taxa': bad_biomarker,
            'good_taxa': good_biomarker,
        },
    }, taxa2id, individual_network


def shannon_index(abundance: pd.Series) -> float:
    abundance = abundance.astype(float)
    abundance = abundance[abundance > 0]
    if abundance.empty:
        return np.nan
    total = abundance.sum()
    if total <= 0:
        return np.nan
    p = abundance / total
    return float(-(p * np.log(p)).sum())


def calculate_mndi(individual_network, sample_list, taxa2id, case_samples, bad_taxa, good_taxa) -> pd.DataFrame:
    bad_idx = [taxa2id[t] for t in bad_taxa if t in taxa2id]
    good_idx = [taxa2id[t] for t in good_taxa if t in taxa2id]

    rows = []
    for i in range(individual_network.shape[0]):
        sample_id = sample_list[i]
        adj = np.abs(individual_network[i])
        adj[np.isnan(adj)] = 0
        np.fill_diagonal(adj, 0)
        bad_internal = adj[np.ix_(bad_idx, bad_idx)].sum() / 2 if bad_idx else np.nan
        good_internal = adj[np.ix_(good_idx, good_idx)].sum() / 2 if good_idx else np.nan
        mndi = np.log10(((bad_internal if pd.notna(bad_internal) else 0) + 1e-5) / ((good_internal if pd.notna(good_internal) else 0) + 1e-5))
        rows.append({
            'sample_id': sample_id,
            'group': 'case' if sample_id in case_samples else 'ctrl',
            'MNDI': mndi,
            'f_bad_bad': bad_internal,
            'f_good_good': good_internal,
        })
    return pd.DataFrame(rows)


def compute_auc(df: pd.DataFrame, score_col: str):
    valid = df[score_col].notna()
    y = df.loc[valid, 'label'].astype(int)
    score = df.loc[valid, score_col].astype(float)
    if y.nunique() < 2:
        return np.nan, np.nan
    raw_auc = roc_auc_score(y, score)
    best_auc = max(raw_auc, 1 - raw_auc)
    return raw_auc, best_auc


def plot_scatter(shannon_df, mndi_df, shannon_name, mndi_name, save_path):
    merged = pd.merge(
        shannon_df[['sample_id', 'group', 'shannon']],
        mndi_df[['sample_id', 'MNDI']],
        on='sample_id',
        how='inner'
    ).dropna(subset=['shannon', 'MNDI'])

    if merged.empty:
        return {'shannon_strategy': shannon_name, 'mndi_strategy': mndi_name, 'n': 0, 'spearman_r': np.nan, 'spearman_p': np.nan}

    rho, pval = spearmanr(merged['shannon'], merged['MNDI'])

    plt.style.use('default')
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 18, 'pdf.fonttype': 42})
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    sns.scatterplot(
        data=merged,
        x='shannon',
        y='MNDI',
        hue='group',
        palette={'case': '#C9352B', 'ctrl': '#339DB5'},
        s=42,
        alpha=0.8,
        ax=ax,
    )
    sns.regplot(data=merged, x='shannon', y='MNDI', scatter=False, color='#666666', ax=ax)
    ax.set_title(f'{shannon_name} vs {mndi_name}\nSpearman r={rho:.3f}, p={pval:.3e}')
    ax.set_xlabel(f'{shannon_name} Shannon')
    ax.set_ylabel(mndi_name)
    ax.legend(frameon=False, title='Group')
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

    return {
        'shannon_strategy': shannon_name,
        'mndi_strategy': mndi_name,
        'n': len(merged),
        'spearman_r': rho,
        'spearman_p': pval,
        'plot_path': save_path,
    }


def plot_auc_comparison(auc_df: pd.DataFrame, save_path: str):
    plot_df = auc_df.copy()
    plot_df['display_name'] = plot_df['metric_type'] + ': ' + plot_df['strategy']
    plot_df['display_auc'] = plot_df['best_direction_auc']
    plot_df = plot_df.sort_values('display_auc', ascending=False)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    sns.barplot(data=plot_df, x='display_auc', y='display_name', hue='metric_type', dodge=False,
                palette={'Shannon': '#7AA6DC', 'MNDI': '#E08B69'}, ax=ax)
    for i, row in plot_df.reset_index(drop=True).iterrows():
        ax.text(row['display_auc'] + 0.01, i, f"{row['display_auc']:.3f}", va='center', fontsize=15)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel('Best-direction AUC')
    ax.set_ylabel('Metric')
    ax.set_title('AUC comparison of Shannon and MNDI metrics')
    # ax.legend(frameon=False, title='Metric type', loc='lower right')
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def main():
    print('Loading label and profile data...')
    label_df = pd.read_csv(LABEL_PATH, sep='\t', index_col=0)
    profile_df = load_profile(PROFILE_PATH)
    pandisease_samples = load_pandisease_samples()

    valid_samples = [s for s in pandisease_samples if s in label_df.index and s in profile_df.index]
    print(f'PANDISEASE samples in list: {len(pandisease_samples)}')
    print(f'PANDISEASE samples with profile + label: {len(valid_samples)}')

    sample_label_df = label_df.loc[valid_samples].copy()
    sample_label_df['group'] = np.where(sample_label_df['disease_united'] == 'healthy', 'ctrl', 'case')
    sample_label_df['label'] = np.where(sample_label_df['disease_united'] == 'healthy', 0, 1)
    case_samples = set(sample_label_df.index[sample_label_df['group'] == 'case'])

    profile_df = profile_df.loc[valid_samples]
    shannon_strategies = {
        'All_taxa': sorted(profile_df.columns.tolist()),
    }
    mdi_taxa_strategies, mndi_strategies, taxa2id, individual_network = load_mdi_strategies_and_network()
    shannon_strategies.update(mdi_taxa_strategies)

    sample_order = load_pandisease_samples()
    network_sample_list = [s for s in sample_order if s in sample_label_df.index]
    if len(network_sample_list) != individual_network.shape[0]:
        network_sample_list = list(sample_label_df.index[:individual_network.shape[0]])

    shannon_results = {}
    summary_rows = []
    combined_rows = []

    for strategy_name, taxa_list in shannon_strategies.items():
        overlap_taxa = [t for t in taxa_list if t in profile_df.columns]
        print(f'\n[{strategy_name}] Input taxa: {len(taxa_list)} | Overlap taxa in profile: {len(overlap_taxa)}')

        sub_df = profile_df[overlap_taxa].copy() if overlap_taxa else pd.DataFrame(index=profile_df.index)
        result_df = sample_label_df[['disease_united', 'group', 'label']].copy()
        result_df['sample_id'] = result_df.index
        result_df['strategy'] = strategy_name
        result_df['n_taxa_overlap'] = len(overlap_taxa)
        result_df['shannon'] = sub_df.apply(shannon_index, axis=1) if overlap_taxa else np.nan

        raw_auc, best_auc = compute_auc(result_df, 'shannon')
        result_df.to_csv(os.path.join(OUTPUT_DIR, f'{strategy_name}.sample_shannon.tsv'), sep='\t', index=False)
        shannon_results[strategy_name] = result_df.copy()
        combined_rows.append(result_df[['sample_id', 'group', 'label', 'strategy', 'shannon']])
        summary_rows.append({
            'strategy': strategy_name,
            'metric_type': 'Shannon',
            'n_samples': len(result_df),
            'n_taxa_overlap': len(overlap_taxa),
            'raw_auc': raw_auc,
            'best_direction_auc': best_auc,
        })

    mndi_results = {}
    for strategy_name, taxa_dict in mndi_strategies.items():
        result_df = calculate_mndi(
            individual_network=individual_network,
            sample_list=network_sample_list,
            taxa2id=taxa2id,
            case_samples=case_samples,
            bad_taxa=taxa_dict['bad_taxa'],
            good_taxa=taxa_dict['good_taxa'],
        )
        result_df = pd.merge(result_df, sample_label_df[['label', 'disease_united']], left_on='sample_id', right_index=True, how='left')
        raw_auc, best_auc = compute_auc(result_df, 'MNDI')
        result_df['strategy'] = strategy_name
        result_df.to_csv(os.path.join(OUTPUT_DIR, f'{strategy_name}.sample_mndi.tsv'), sep='\t', index=False)
        mndi_results[strategy_name] = result_df.copy()
        summary_rows.append({
            'strategy': strategy_name,
            'metric_type': 'MNDI',
            'n_samples': len(result_df),
            'n_taxa_overlap': np.nan,
            'raw_auc': raw_auc,
            'best_direction_auc': best_auc,
        })

    spearman_rows = []
    for shannon_name, shannon_df in shannon_results.items():
        for mndi_name, mndi_df in mndi_results.items():
            save_path = os.path.join(OUTPUT_DIR, f'{shannon_name}.vs.{mndi_name}.scatter.pdf')
            spearman_rows.append(plot_scatter(shannon_df, mndi_df, shannon_name, mndi_name, save_path))

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(OUTPUT_DIR, 'mdi.shannon.compare.summary.tsv')
    summary_df.to_csv(summary_path, sep='\t', index=False)

    spearman_df = pd.DataFrame(spearman_rows)
    spearman_path = os.path.join(OUTPUT_DIR, 'mdi.shannon.compare.spearman.tsv')
    spearman_df.to_csv(spearman_path, sep='\t', index=False)

    combined_path = os.path.join(OUTPUT_DIR, 'mdi.shannon.compare.all_samples.tsv')
    pd.concat(combined_rows, axis=0, ignore_index=True).to_csv(combined_path, sep='\t', index=False)

    auc_plot_path = os.path.join(OUTPUT_DIR, 'mdi.shannon.compare.auc_comparison.pdf')
    plot_auc_comparison(summary_df[['strategy', 'metric_type', 'raw_auc', 'best_direction_auc']].copy(), auc_plot_path)

    print('\nDone.')
    print(f'Summary saved to: {summary_path}')
    print(f'Spearman results saved to: {spearman_path}')
    print(f'AUC comparison figure saved to: {auc_plot_path}')


# if __name__ == '__main__':
#     main()


# 画MNDI和shannon在pandisease crc ibd的AUC比较
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/figures/finetune/11.25/networks/4.1/abs'

PLOT_CONFIGS = [
    {
        'input': os.path.join(BASE_DIR, 'new.pandisease.core.disease.case.ctrl.stats1.tsv'),
        'output': os.path.join(BASE_DIR, 'new.pandisease.core.disease.case.ctrl.auc.line.pdf'),
        'label_col': 'disease',
        'title': 'Pan-disease cohorts',
    },
    # {
    #     'input': os.path.join(BASE_DIR, 'new.CRC.project.individual.case.ctrl.project.stats1.tsv'),
    #     'output': os.path.join(BASE_DIR, 'new.CRC.project.individual.case.ctrl.auc.line.pdf'),
    #     'label_col': 'project',
    #     'title': 'CRC cohorts',
    # },
    # {
    #     'input': os.path.join(BASE_DIR, 'new.IBD.project.individual.case.ctrl.project.stats1.tsv'),
    #     'output': os.path.join(BASE_DIR, 'new.IBD.project.individual.case.ctrl.auc.line.pdf'),
    #     'label_col': 'project',
    #     'title': 'IBD cohorts',
    # },
]


def load_auc_table(path: str, label_col: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep='\t')
    keep_cols = [label_col, 'MNDI_AUC', 'Shannon_AUC', 'n_case', 'n_ctrl']
    df = df[keep_cols].copy()
    df = df.dropna(subset=['MNDI_AUC', 'Shannon_AUC'])
    df = df[(df['n_case'] > 0) & (df['n_ctrl'] > 0)].copy()
    df = df.sort_values('MNDI_AUC', ascending=False).reset_index(drop=True)
    return df



def plot_auc_lines(df: pd.DataFrame, label_col: str, title: str, output_path: str) -> None:
    plt.rc('font', size=22)
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['pdf.fonttype'] = 42
    sns.set_style('ticks')

    fig_width = max(10, 0.65 * len(df) + 3)
    fig, ax = plt.subplots(figsize=(fig_width, 6.5), facecolor='white')

    x = range(len(df))
    ax.plot(
        x, df['MNDI_AUC'], color='#C9352B', marker='o', linewidth=3.0,
        markersize=8, label='MNDI'
    )
    ax.plot(
        x, df['Shannon_AUC'], color='#339DB5', marker='s', linewidth=3.0,
        markersize=7, label='Shannon'
    )

    ax.fill_between(x, df['MNDI_AUC'], df['Shannon_AUC'], color='#E9EEF6', alpha=0.45)
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.8)

    ax.set_xticks(list(x))
    ax.set_xticklabels(df[label_col], rotation=50, ha='right')
    ax.set_ylabel('AUC')
    ax.set_xlabel('')
    ax.set_title(title, pad=18, fontsize=24)
    ax.set_ylim(0.45, min(1.0, max(df[['MNDI_AUC', 'Shannon_AUC']].max()) + 0.06))

    for idx, row in df.iterrows():
        diff = row['MNDI_AUC'] - row['Shannon_AUC']
        ax.text(
            idx, max(row['MNDI_AUC'], row['Shannon_AUC']) + 0.015,
            f'{diff:+.02f}', ha='center', va='bottom', fontsize=12,
            color='#7A1F1A' if diff > 0 else '#1E5E6F'
        )

    ax.legend(frameon=False, loc='upper right', fontsize=18)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=800, bbox_inches='tight')
    plt.close(fig)



def main() -> None:
    for cfg in PLOT_CONFIGS:
        df = load_auc_table(cfg['input'], cfg['label_col'])
        if df.empty:
            print(f'Skip empty file: {cfg["input"]}')
            continue
        plot_auc_lines(df, cfg['label_col'], cfg['title'], cfg['output'])
        print(f'Saved: {cfg["output"]}')


if __name__ == '__main__':
    main()

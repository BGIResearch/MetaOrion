import os
import pickle
import random
import colorsys
import warnings

warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.lines import Line2D
import networkx as nx
import scipy.stats as stats

from network_dysbiosis_index import calculate_mdi

# =============================================================================
# Global Configuration Section
# Centralized management of file paths, chart parameters, and color palettes
# for easy modification.
# =============================================================================
CONFIG = {
    'paths': {
        'network_dir': '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results/network/1.8/Antibiotic.intervention.longi.complete/',
        'biomarker_dir': '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results/network/1.8/pandisease/',
        'save_dir': '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/figures/finetune/11.25/networks/4.1/test.change.5.9/antibiotic.longi/',
        'profile_path': '/bgi-seq-model-2/datasets/zhangkexin/meta_index/metaphlan4/fine-tune/Antibiotic.intervention.longi/merged_abundance_table_supp.profile',
        'label_path': '/bgi-seq-model-2/datasets/zhangkexin/meta_index/metaphlan4/fine-tune/Antibiotic.intervention.longi/supp.tsv',
        'output_gml_dir': '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results/network/1.8/Antibiotic.intervention.longi/core.selected.individuals/'
    }
}

def calculate_shannon(row):
    """Calculate the Shannon diversity index for a single sample."""
    row = row[row > 0]
    if len(row) == 0:
        return 0
    p = row / row.sum()
    return -np.sum(p * np.log(p))


# =============================================================================
# Data Preprocessing Module
# =============================================================================
def prepare_abundance_plot_data(profile_df, label_df, good_cols, bad_cols, canon_order):
    """
    Calculate the relative abundance of core biomarkers and convert the dataframe
    into a long format supported by Seaborn for plotting.
    """
    df = profile_df.copy()

    row_sums = df.sum(axis=1)
    df['PR-decreasing'] = (df[good_cols].sum(axis=1) / row_sums) * 100
    df['PR-increasing'] = (df[bad_cols].sum(axis=1) / row_sums) * 100

    abund_summary = df[['PR-decreasing', 'PR-increasing']].reset_index().rename(columns={'index': 'sample_id'})
    merged_df = pd.merge(abund_summary, label_df[['sample_id', 'canon', 'abx']], on='sample_id', how='inner')
    merged_df = merged_df.dropna(subset=['canon'])

    merged_df['canon'] = pd.Categorical(merged_df['canon'], categories=canon_order, ordered=True)

    melted_df = merged_df.melt(
        id_vars=['sample_id', 'canon'],
        value_vars=['PR-decreasing', 'PR-increasing'],
        var_name='Consensus',
        value_name='Total Abundance (%)'
    )

    return melted_df.sort_values('canon')
def filter_profile(X):
    """Retain species-level (s__) features and simplify column names."""
    index = [i for i, col in enumerate(X.columns) if col.split('|')[-1].split('__')[0] == 's']
    X = X.iloc[:, index]
    X.columns = [col.split('|')[-1] for col in X.columns]
    return X


def clean_species_name(col):
    """Format species names: convert to lowercase, replace prefix and underscores with hyphens."""
    name = str(col).split('|')[-1].lower()
    return name.replace('s__', 's-').replace('_', '-').replace(' ', '-')


def set_canon_order(df):
    """Define the categorical order for time stages (canon)."""
    canon_order = sorted(df['canon'].dropna().unique())
    df['canon'] = pd.Categorical(df['canon'], categories=canon_order, ordered=True)
    return canon_order


def prepare_metric_df(metadata_columns, label_df, shannon_df, df_core_features):
    """Integrate metadata with calculated metrics and apply Z-score standardization."""
    core_mdi = df_core_features[['sample_id', 'MDI']].rename(columns={'MDI': 'Core_MDI'})

    metric_df = label_df[metadata_columns].copy()
    metric_df = metric_df.merge(shannon_df, on='sample_id', how='inner')
    metric_df = metric_df.merge(core_mdi, on='sample_id', how='inner')

    # Standardize core metrics using Z-score
    for metric in ['Shannon', 'Core_MDI']:
        metric_df[f'{metric}_Z'] = stats.zscore(metric_df[metric], nan_policy='omit')

    return metric_df


# =============================================================================
# Visualization Helpers Module
# =============================================================================
def apply_figure_style():
    """Apply global chart style settings."""
    plt.rcParams['font.family'] = 'Arial'
    plt.rc('font', size=24)
    plt.rcParams['pdf.fonttype'] = 42


def generate_random_colors(n=100, seed=111):
    """Generate a specified number of random colors with low saturation and high value."""
    random.seed(seed)
    colors = []
    for _ in range(n):
        h = random.random()
        s = random.uniform(0.2, 0.6)  # Low saturation
        v = random.uniform(0.8, 1.0)  # High value/brightness
        colors.append(colorsys.hsv_to_rgb(h, s, v))

    colors_hex = ['#{:02x}{:02x}{:02x}'.format(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in colors]
    return sns.color_palette(colors_hex)


# =============================================================================
# Network Processing Module
# =============================================================================
def construct_graph(corr_matrix, threshold):
    """Construct an undirected graph based on an absolute correlation threshold."""
    G = nx.Graph()
    for i in range(len(corr_matrix)):
        for j in range(i + 1, len(corr_matrix)):
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) > threshold:
                G.add_edge(corr_matrix.index[i], corr_matrix.columns[j], weight=corr_value)
    return G


def rename_func(old_name):
    """Format network node names: remove prefix and abbreviate the genus name."""
    name = old_name[2:].replace('-', ' ')
    parts = name.split(maxsplit=1)
    if len(parts) == 2:
        return f"{parts[0][0].upper()}. {parts[1]}"
    return name


def extract_and_annotate_subgraph(G, target_nodes, good_bm, bad_bm):
    """Extract a subgraph of target nodes, remove isolated nodes, and annotate Biomarker types."""
    valid_nodes = [n for n in target_nodes if n in G]
    subG = G.subgraph(valid_nodes).copy()

    attrs = {}
    for node in subG.nodes():
        if node in good_bm:
            attrs[node] = {'type': 'Beneficial Biomarker'}
        elif node in bad_bm:
            attrs[node] = {'type': 'Pathogenic Biomarker'}
        else:
            attrs[node] = {'type': 'Unknown'}

    nx.set_node_attributes(subG, attrs)
    subG = nx.relabel_nodes(subG, rename_func, copy=False)
    subG.remove_nodes_from(list(nx.isolates(subG)))
    return subG


def add_node_size_by_rank(G, good_list, bad_list):
    """Calculate node weights based on their rank in the provided lists."""
    attrs = {}
    for bm_list in [good_list, bad_list]:
        for index, old_name in enumerate(bm_list):
            new_name = rename_func(old_name)
            if new_name in G.nodes():
                rank = index + 1
                attrs[new_name] = {'biomarker_rank': rank, 'node_weight': 50 - rank + 1}
    nx.set_node_attributes(G, attrs)


def add_edge_prev(G, ind_net_tensor, t2id):
    """Calculate and assign the edge prevalence across the entire sample population."""
    for u, v in G.edges():
        if u in t2id and v in t2id:
            i, j = t2id[u], t2id[v]
            sub = ind_net_tensor[:, i, j]
            prev = np.sum(~np.isnan(sub)) / len(ind_net_tensor)
            G.edges[u, v]['edge_prev'] = float(prev)


# =============================================================================
# Main Execution Logic
# =============================================================================
if __name__ == "__main__":

    # ---------------------------------------------------------
    # 1. Data Loading and Basic Processing
    # ---------------------------------------------------------
    # Load sample list
    with open(CONFIG['paths']['network_dir'] + 'pretrain.emb.samples.list.txt', 'r') as f:
        sample_list = f.read().splitlines()

    # Load abundance profile and labels
    profile = filter_profile(pd.read_csv(CONFIG['paths']['profile_path'], sep='\t', index_col=0).transpose())
    label_df = pd.read_csv(CONFIG['paths']['label_path'], sep='\t', index_col=1)
    label_df['sample_id'] = list(label_df.index)

    # Load individual networks and indices
    individual_network = np.load(CONFIG['paths']['network_dir'] + 'pretrain.emb.individual.npy')
    with open(CONFIG['paths']['network_dir'] + 'pretrain.emb.individual.index.pkl', 'rb') as f:
        taxa2id = pickle.load(f)

    # Load core Biomarkers (Top 30)
    top30_good_df = pd.read_csv(CONFIG['paths']['biomarker_dir'] + 'top30.negative.o3.csv', index_col=0)
    top30_bad_df = pd.read_csv(CONFIG['paths']['biomarker_dir'] + 'top30.positive.o3.csv', index_col=0)
    top30_core_good_biomarker = ['s-' + i.lower().replace(" ", "-") for i in top30_good_df.index]
    top30_core_bad_biomarker = ['s-' + i.lower().replace(" ", "-") for i in top30_bad_df.index]

    # ---------------------------------------------------------
    # 2. Core Metrics Calculation (MDI & Shannon)
    # ---------------------------------------------------------
    df_core_features = calculate_mdi(individual_network, sample_list, taxa2id, sample_list, sample_list,
                                        top30_core_bad_biomarker, top30_core_good_biomarker)

    shannon_series = profile.apply(calculate_shannon, axis=1)
    shannon_df = pd.DataFrame({
        'sample_id': [i.split('.mp4')[0] for i in shannon_series.index],
        'Shannon': shannon_series.values
    })

    # =========================================================
    # Experiment 1: Longitudinal trends vs. Core Biomarker Abundance (Figure 1)
    # =========================================================
    apply_figure_style()
    plot_df = prepare_metric_df(['sample_id', 'canon', 'subject', 'abx'], label_df, shannon_df,
                      df_core_features)

    melted_df = plot_df.melt(
        id_vars=['sample_id', 'canon'],
        value_vars=['Shannon_Z', 'Core_MDI_Z'],
        var_name='Metric', value_name='Z-Score'
    )
    melted_df['Metric'] = melted_df['Metric'].map({'Shannon_Z': 'Shannon', 'Core_MDI_Z': 'MDI (Core)'})
    canon_order = set_canon_order(melted_df)

    # Abundance calculation
    abundance_df = profile.copy()
    abundance_df.index = [str(idx).split('.mp4')[0] for idx in abundance_df.index]
    abundance_df.columns = [clean_species_name(c) for c in abundance_df.columns]

    valid_good_cols = [clean_species_name(c) for c in top30_core_good_biomarker if
                       clean_species_name(c) in abundance_df.columns]
    valid_bad_cols = [clean_species_name(c) for c in top30_core_bad_biomarker if
                      clean_species_name(c) in abundance_df.columns]
    print(f"Matched Core Good Biomarkers: {len(valid_good_cols)} | Core Bad Biomarkers: {len(valid_bad_cols)}")

    melted_line_df = prepare_abundance_plot_data(
        profile_df=abundance_df,
        label_df=label_df,
        good_cols=valid_good_cols,
        bad_cols=valid_bad_cols,
        canon_order=canon_order
    )

    # Plotting - Figure 1
    fig1, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 14), sharex=True)
    custom_colors_abund = {'PR-decreasing': '#4DBAD6', 'PR-increasing': '#E44A33'}

    sns.lineplot(
        data=melted_line_df, x='canon', y='Total Abundance (%)', hue='Consensus', style='Consensus',
        markers={'PR-decreasing': 'o', 'PR-increasing': 's'}, dashes=False, palette=custom_colors_abund,
        linewidth=2.5, markersize=10, errorbar=('se', 1), err_style='bars',
        err_kws={'capsize': 5, 'linewidth': 2}, ax=ax1
    )
    ax1.grid(True, linestyle='--', linewidth=1.5, alpha=0.5, zorder=0)
    ax1.legend(frameon=False, loc='upper right', fontsize=20)

    custom_palette_trend = {'Shannon': '#9E9E9E', 'MDI (Core)': '#F5C96B'}
    sns.lineplot(
        data=melted_df, x='canon', y='Z-Score', hue='Metric', marker='o', markersize=12,
        linewidth=4, errorbar=('ci', 95), err_style='band', palette=custom_palette_trend, ax=ax2
    )
    ax2.grid(True, linestyle='--', linewidth=1.5, alpha=0.5, zorder=0)
    ax2.set_ylabel('Normalized Value\n(Z-Score)')
    ax2.set_xlabel('Canon Stages (Time)')
    ax2.legend(title='', loc='upper right', frameon=False, fontsize=20)
    ax2.tick_params(axis='x', rotation=45)

    fig1.align_ylabels([ax1, ax2])
    plt.tight_layout()
    fig1.subplots_adjust(hspace=0.08)
    fig1.savefig(os.path.join(CONFIG['paths']['save_dir'], 'Trend_and_Abundance_Combined_6.1.pdf'), dpi=800,
                 bbox_inches='tight')
    plt.show()

    # =========================================================
    # Experiment 2: Subject-specific Shannon and MDI trajectories (Figure 2)
    # =========================================================
    selected_subjects = ['CAK', 'CAA', 'CAM', 'CAC', 'CAN', 'EBR', 'BAG', 'EAC']

    df_ind = plot_df[plot_df['subject'].isin(selected_subjects)].dropna(subset=['canon']).copy()
    set_canon_order(df_ind)
    df_ind = df_ind.sort_values(['subject', 'canon'])
    df_ind['Group'] = df_ind['abx'].map({'Y': 'Case Subject', 'N': 'Control Subject'})

    subject_colors = ['#8AB1D2', '#F5AA61', '#9DD0C7', '#D0D0D0', '#C7988C', '#E58579', '#D9BDDB', '#F1DFA4']
    custom_palette_ind = dict(zip(selected_subjects, subject_colors))
    group_markers = {'Case Subject': 'o', 'Control Subject': '^'}
    group_dashes = {'Case Subject': '', 'Control Subject': (4, 2)}

    # Plotting - Figure 2
    fig2, (ax3, ax4) = plt.subplots(nrows=2, ncols=1, figsize=(15, 14), sharex=True)

    sns.lineplot(
        data=df_ind, x='canon', y='Shannon_Z', hue='subject', style='Group',
        markers=group_markers, dashes=group_dashes, alpha=1, linewidth=4,
        markersize=12, palette=custom_palette_ind, ax=ax3, legend=False
    )
    ax3.set_ylabel("Shannon (Z-Score)")
    ax3.axhline(0, color='gray', linestyle='--', linewidth=2, alpha=0.5, zorder=0)

    # Custom legends
    custom_lines_sub = [Line2D([0], [0], color=c, lw=5) for c in subject_colors]
    ax3.add_artist(ax3.legend(custom_lines_sub, selected_subjects, loc='lower left', ncol=3, frameon=False, fontsize=20,
                              title="Subjects"))

    custom_lines_group = [
        Line2D([0], [0], color='gray', lw=4, linestyle='-', marker='o', markersize=12),
        Line2D([0], [0], color='gray', lw=4, linestyle='--', marker='^', markersize=12)
    ]
    ax3.legend(custom_lines_group, ['Case Subject', 'Control Subject'], loc='lower right', frameon=False, fontsize=20,
               title="Group")

    sns.lineplot(
        data=df_ind, x='canon', y='Core_MDI_Z', hue='subject', style='Group',
        markers=group_markers, dashes=group_dashes, alpha=1, linewidth=4,
        markersize=12, palette=custom_palette_ind, ax=ax4, legend=False
    )
    ax4.set_ylabel("MDI (Z-Score)")
    ax4.set_xlabel("Canon Stages (Time)")
    ax4.tick_params(axis='x', rotation=45)
    ax4.axhline(0, color='gray', linestyle='--', linewidth=2, alpha=0.5, zorder=0)

    fig2.align_ylabels([ax3, ax4])
    plt.tight_layout()
    fig2.subplots_adjust(hspace=0.08)
    fig2.savefig(os.path.join(CONFIG['paths']['save_dir'], "Individual_Trajectories_Combined.pdf"), dpi=800,
                 bbox_inches='tight')
    plt.show()

    # =========================================================
    # Experiment 3: Core Biomarker Subgraph Export
    # =========================================================
    os.makedirs(CONFIG['paths']['output_gml_dir'], exist_ok=True)
    all_target_biomarkers = top30_core_good_biomarker + top30_core_bad_biomarker
    selected_sample_ids = df_ind['sample_id'].unique()

    print(f"\nStarting to process and export network subgraphs for {len(selected_sample_ids)} selected samples...")

    for sample_id in selected_sample_ids:
        sample_idx = sample_list.index(sample_id)
        canon_label = df_ind.loc[df_ind['sample_id'] == sample_id, 'canon'].values[0]

        if canon_label in ['A1', 'B1', 'B3', 'B5', 'C1', 'C3', 'C5', 'C6']:
            # Reconstruct correlation matrix
            taxa_keys = list(taxa2id.keys())
            corr_matrix = pd.DataFrame(individual_network[sample_idx, :, :], columns=taxa_keys, index=taxa_keys)

            G_full = construct_graph(corr_matrix, threshold=0.01)
            subG = extract_and_annotate_subgraph(G_full, all_target_biomarkers, top30_core_good_biomarker,
                                                 top30_core_bad_biomarker)

            if len(subG.nodes()) > 0:
                add_node_size_by_rank(subG, top30_core_good_biomarker, top30_core_bad_biomarker)
                add_edge_prev(subG, individual_network, taxa2id)

                save_path = os.path.join(CONFIG['paths']['output_gml_dir'], f"core_{sample_id}_{canon_label}.gml")
                nx.write_gml(subG, save_path)  # Uncomment to save the file physically
                print(f"Saved: {sample_id}_{canon_label} | Nodes: {len(subG.nodes())}, Edges: {len(subG.edges())}")
            else:
                print(f"Skipped: {sample_id}_{canon_label} | Reason: Subgraph is empty after filtering.")



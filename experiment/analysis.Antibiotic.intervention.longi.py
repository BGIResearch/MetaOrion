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

def calculate_mndi_v2(individual_network, sample_list, taxa2id,
                      case_samples, ctrl_samples,
                      bad_taxa, good_taxa):
    """
    individual_network: (n_sample, n_taxa, n_taxa)
    """
    bad_idx = [taxa2id[t] for t in bad_taxa if t in taxa2id]
    good_idx = [taxa2id[t] for t in good_taxa if t in taxa2id]

    mndi_features = []

    for i in range(individual_network.shape[0]):
        sample_feature_dict = {'sample_id': sample_list[i]}
        adj = np.abs(individual_network[i])
        adj[np.isnan(adj)] = 0
        np.fill_diagonal(adj, 0)

        bad_internal = adj[np.ix_(bad_idx, bad_idx)].sum() / 2
        bad_all = adj[bad_idx, :].sum()
        good_internal = adj[np.ix_(good_idx, good_idx)].sum() / 2
        good_all = adj[good_idx, :].sum()
        bad_good_internal = adj[np.ix_(bad_idx, good_idx)].sum()
        total_edges = adj.sum() / 2

        sample_feature_dict.update({
            'f_bad_bad': bad_internal,
            'f_bad_all': bad_all,
            'f_bad_dominance': bad_all / (total_edges + 1e-5),
            'f_good_good': good_internal,
            'f_good_all': good_all,
            'f_good_dominance': good_all / (total_edges + 1e-5),
            'f_total_edges': total_edges,
            'f_tmp': (bad_internal + 1e-5) / (good_internal + 1e-5),
            'f_bad_good_internal': bad_good_internal,
            'f_good_loss': 1 / (good_all + 1e-5)
        })

        mndi_features.append(sample_feature_dict)

    df_features = pd.DataFrame(mndi_features)
    df_features['group'] = df_features['sample_id'].apply(lambda x: 'case' if x in case_samples else 'ctrl')

    df_features['MNDI'] = np.log10(df_features['f_tmp'] + 1e-5)

    return df_features

def generate_random_colors(n=100, seed=111):
    random.seed(seed)
    colors = []
    for _ in range(n):
        h = random.random()
        s = random.uniform(0.2, 0.6)  # 低饱和度
        v = random.uniform(0.8, 1.0)  # 高明度
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        colors.append((r, g, b))

    colors_hex = [
        '#{:02x}{:02x}{:02x}'.format(int(r * 255), int(g * 255), int(b * 255))
        for r, g, b in colors
    ]
    sns_palette = sns.color_palette(colors_hex)
    return sns_palette

def filter_profile(X):
    index = []
    for i in range(len(X.axes[1])):
        if X.columns[i].split('|')[-1].split('__')[0] == 's':
            index.append(i)
    X = X.iloc[:, index]
    X.columns = [i.split('|')[-1] for i in X.columns]
    return X

network_dir = '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results/network/1.8/Antibiotic.intervention.longi.complete/'
biomarker_dir = '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results/network/1.8/pandisease/'
save_dir = '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/figures/finetune/11.25/networks/4.1/abs/Antibiotic.longi/'
profile_path = '/bgi-seq-model-2/datasets/zhangkexin/meta_index/metaphlan4/fine-tune/Antibiotic.intervention.longi/merged_abundance_table_supp.profile'
sample_list = []
with open(network_dir + 'pretrain.emb.samples.list.txt', 'r') as f:
    for line in f.readlines():
        sample_list.append(line.strip())
profile = filter_profile(pd.read_csv(profile_path, sep='\t', index_col=0).transpose())
label_df = pd.read_csv(
    '/bgi-seq-model-2/datasets/zhangkexin/meta_index/metaphlan4/fine-tune/Antibiotic.intervention.longi/supp.tsv',
    sep='\t', index_col=1)
label_df['sample_id'] = list(label_df.index)

individual_network = np.load(network_dir + 'pretrain.emb.individual.npy')
taxa2id = pickle.load(open(network_dir + 'pretrain.emb.individual.index.pkl', 'rb'))
top30_core_good_biomarker = ['s-' + i.lower().replace(" ", "-") for i in
                        list(pd.read_csv(biomarker_dir + 'top30.negative.o3.csv', index_col=0).index)]
top30_core_bad_biomarker = ['s-' + i.lower().replace(" ", "-") for i in
                       list(pd.read_csv(biomarker_dir + 'top30.positive.o3.csv', index_col=0).index)]

pan_biomarker = pd.read_csv(
    '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results/features/11.25/sort.multidisease.mean/pandisease.biomarker.txt',
    sep='\t', header=None)
pan_biomarker['weight.abs'] = pan_biomarker[1].abs()
pan_biomarker = pan_biomarker.sort_values('weight.abs', ascending=False)
good_biomarker = list(pan_biomarker[pan_biomarker[1] < 0].iloc[:, 0])
bad_biomarker = list(pan_biomarker[pan_biomarker[1] > 0].iloc[:, 0])

df_full_features = calculate_mndi_v2(individual_network, sample_list, taxa2id,
                                sample_list, sample_list,
                                bad_biomarker, good_biomarker)
df_core_features = calculate_mndi_v2(individual_network, sample_list, taxa2id,
                                sample_list, sample_list,
                                top30_core_bad_biomarker, top30_core_good_biomarker)
# merge_df = pd.merge(df_full_features, label_df, how='inner', on='sample_id')
# merge_df = merge_df.loc[merge_df['MNDI'] < 5] # 去除有一个很极端的样本


# Experiment 1: longitudinal MNDI/Shannon trends and biomarker abundance.
def calculate_shannon(row):
    row = row[row > 0]
    if len(row) == 0:
        return 0
    p = row / row.sum()
    return -np.sum(p * np.log(p))


def set_figure_style():
    plt.rcParams['font.family'] = 'Arial'
    plt.rc('font', size=24)
    plt.rcParams['pdf.fonttype'] = 42


def set_canon_order(df):
    canon_order = sorted(df['canon'].dropna().unique())
    df['canon'] = pd.Categorical(df['canon'], categories=canon_order, ordered=True)
    return canon_order


def prepare_metric_df(metadata_columns):
    full_mndi = df_full_features[['sample_id', 'MNDI']].rename(columns={'MNDI': 'Full_MNDI'})
    core_mndi = df_core_features[['sample_id', 'MNDI']].rename(columns={'MNDI': 'Core_MNDI'})

    metric_df = label_df[metadata_columns].copy()
    metric_df = metric_df.merge(shannon_df, on='sample_id', how='inner')
    metric_df = metric_df.merge(full_mndi, on='sample_id', how='inner')
    metric_df = metric_df.merge(core_mndi, on='sample_id', how='inner')

    for metric in ['Shannon', 'Full_MNDI', 'Core_MNDI']:
        metric_df[f'{metric}_Z'] = stats.zscore(metric_df[metric], nan_policy='omit')

    return metric_df

shannon_series = profile.apply(calculate_shannon, axis=1)
shannon_df = pd.DataFrame({
    'sample_id': [i.split('.mp4')[0] for i in shannon_series.index],
    'Shannon': shannon_series.values
})

plot_df = prepare_metric_df(['sample_id', 'canon'])
set_canon_order(plot_df)
melted_df = plot_df.melt(
    id_vars=['sample_id', 'canon'],
    value_vars=['Shannon_Z', 'Core_MNDI_Z'],
    var_name='Metric',
    value_name='Z-Score'
)

metric_name_map = {
    'Shannon_Z': 'Shannon',
    'Core_MNDI_Z': 'MNDI (Core)'
}
melted_df['Metric'] = melted_df['Metric'].map(metric_name_map)

set_canon_order(melted_df)


def clean_species_name(col):
    name = str(col).split('|')[-1]
    name = name.lower()
    name = name.replace('s__', 's-')
    name = name.replace('_', '-')
    name = name.replace(' ', '-')
    return name

canon_order = sorted(label_df['canon'].dropna().unique())

abundance_df = profile.copy()
abundance_df.index = [str(idx).split('.mp4')[0] for idx in abundance_df.index]
abundance_df.columns = [clean_species_name(c) for c in abundance_df.columns]

formatted_core_good = [clean_species_name(c) for c in top30_core_good_biomarker]
formatted_core_bad = [clean_species_name(c) for c in top30_core_bad_biomarker]

valid_good_cols = [col for col in formatted_core_good if col in abundance_df.columns]
valid_bad_cols = [col for col in formatted_core_bad if col in abundance_df.columns]

print(f"匹配到的 Core Good Biomarkers 数量: {len(valid_good_cols)}")
print(f"匹配到的 Core Bad Biomarkers 数量: {len(valid_bad_cols)}")

row_sums = abundance_df.sum(axis=1)
abundance_df['Good_Abundance'] = (abundance_df[valid_good_cols].sum(axis=1) / row_sums) * 100
abundance_df['Bad_Abundance'] = (abundance_df[valid_bad_cols].sum(axis=1) / row_sums) * 100

abund_summary = abundance_df[['Good_Abundance', 'Bad_Abundance']].reset_index()
abund_summary.rename(columns={'index': 'sample_id'}, inplace=True)

line_df = pd.merge(abund_summary, label_df[['sample_id', 'canon', 'abx']], on='sample_id', how='inner')
line_df = line_df.dropna(subset=['canon'])

line_df['canon'] = pd.Categorical(line_df['canon'], categories=canon_order, ordered=True)

melted_line_df = line_df.melt(
    id_vars=['sample_id', 'canon'],
    value_vars=['Good_Abundance', 'Bad_Abundance'],
    var_name='Consensus', value_name='Total Abundance (%)'
)

good_label = 'PR-decreasing'
bad_label = 'PR-increasing'

melted_line_df['Consensus'] = melted_line_df['Consensus'].map({'Good_Abundance': good_label, 'Bad_Abundance': bad_label})
melted_line_df = melted_line_df.sort_values('canon')

# =========================================================
# Figure 1: 联合绘制 Trend (MNDI vs Shannon) 和 Core Abundance
# =========================================================
set_figure_style()

fig1, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 14), sharex=True)

custom_colors_abund = {good_label: '#4DBAD6', bad_label: '#E44A33'}
custom_markers_abund = {good_label: 'o', bad_label: 's'}
sns.lineplot(
    data=melted_line_df, x='canon', y='Total Abundance (%)', hue='Consensus', style='Consensus',
    markers=custom_markers_abund, dashes=False, palette=custom_colors_abund, linewidth=2.5, markersize=10,
    errorbar=('se', 1), err_style='bars', err_kws={'capsize': 5, 'linewidth': 2}, ax=ax1
)
ax1.grid(True, which='major', axis='both', linestyle='--', linewidth=1.5, alpha=0.5, zorder=0)
ax1.set_facecolor('white')
ax1.set_ylabel('Total Abundance (%)')
ax1.set_xlabel('')
ax1.legend(frameon=False, loc='upper right', fontsize=20)

custom_palette_trend = {'Shannon': '#FF7F00', 'MNDI (Core)': '#4DAF4A'}
sns.lineplot(
    data=melted_df, x='canon', y='Z-Score', hue='Metric',
    marker='o', markersize=12, linewidth=4, errorbar=('ci', 95),
    err_style='band', palette=custom_palette_trend, ax=ax2
)
ax2.grid(True, which='major', axis='both', linestyle='--', linewidth=1.5, alpha=0.5, zorder=0)
ax2.set_ylabel('Normalized Value\n(Z-Score)')
ax2.set_xlabel('Canon Stages (Time)')
ax2.legend(title='', loc='upper right', frameon=False, fontsize=20)
ax2.tick_params(axis='x', rotation=45)

fig1.align_ylabels([ax1, ax2])

plt.tight_layout()
fig1.subplots_adjust(hspace=0.08)

fig1.savefig(save_dir + 'Trend_and_Abundance_Combined_5.20.pdf', dpi=800, bbox_inches='tight')
plt.show()

# Experiment 2: selected subject-level Shannon/MNDI trajectories.
plot_df = prepare_metric_df(['sample_id', 'canon', 'subject', 'abx'])
stable_subjects = ['CAK', 'DAL', 'CAM']
volatile_subjects = ['EBR', 'BAG', 'EAC']
selected_subjects = stable_subjects + volatile_subjects

df_ind = plot_df[plot_df['subject'].isin(selected_subjects)].copy()
df_ind = df_ind.dropna(subset=['canon'])
set_canon_order(df_ind)
df_ind = df_ind.sort_values(['subject', 'canon'])
df_ind['Group'] = df_ind['abx'].map({'Y': 'Case Subject', 'N': 'Control Subject'})

subject_colors = ['#8AB1D2', '#F5AA61', '#9DD0C7', '#E58579', '#D9BDDB', '#F1DFA4']
custom_palette_ind = dict(zip(selected_subjects, subject_colors))
group_markers = {'Case Subject': 'o', 'Control Subject': '^'}
group_dashes = {'Case Subject': '', 'Control Subject': (4, 2)}

set_figure_style()
fig2, (ax3, ax4) = plt.subplots(nrows=2, ncols=1, figsize=(15, 14), sharex=True)

sns.lineplot(
    data=df_ind, x='canon', y='Shannon_Z', hue='subject', style='Group',
    markers=group_markers, dashes=group_dashes,
    alpha=1, linewidth=4, markersize=12, palette=custom_palette_ind,
    ax=ax3, legend=False
)
ax3.set_ylabel("Shannon (Z-Score)")
ax3.set_xlabel("")
ax3.axhline(0, color='gray', linestyle='--', linewidth=2, alpha=0.5, zorder=0)

custom_lines_sub = [Line2D([0], [0], color=c, lw=5, linestyle='-') for c in subject_colors]
leg1 = ax3.legend(custom_lines_sub, selected_subjects, loc='lower left', ncol=3,
                  frameon=False, fontsize=20, title="Subjects", title_fontsize=22)
ax3.add_artist(leg1)

custom_lines_group = [
    Line2D([0], [0], color='gray', lw=4, linestyle='-', marker='o', markersize=12),
    Line2D([0], [0], color='gray', lw=4, linestyle='--', marker='^', markersize=12)
]
ax3.legend(custom_lines_group, ['Case Subject', 'Control Subject'], loc='lower right',
           frameon=False, fontsize=20, title="Group", title_fontsize=22)


sns.lineplot(
    data=df_ind, x='canon', y='Core_MNDI_Z', hue='subject', style='Group',
    markers=group_markers, dashes=group_dashes,
    alpha=1, linewidth=4, markersize=12, palette=custom_palette_ind,
    ax=ax4, legend=False
)
ax4.set_ylabel("MNDI (Z-Score)")
ax4.set_xlabel("Canon Stages (Time)")
ax4.tick_params(axis='x', rotation=45)
ax4.axhline(0, color='gray', linestyle='--', linewidth=2, alpha=0.5, zorder=0)

fig2.align_ylabels([ax3, ax4])

plt.tight_layout()
fig2.subplots_adjust(hspace=0.08)

save_path = save_dir + "Individual_Shannon_and_MNDI_Zscore_Combined_4.21.pdf"
fig2.savefig(save_path, dpi=800, bbox_inches='tight')
plt.show()


# 提取所选择样本的个体网络
def construct_graph(corr_matrix, threshold):
    """根据相关性矩阵和阈值构建无向图"""
    G = nx.Graph()
    for i in range(len(corr_matrix)):
        for j in range(i + 1, len(corr_matrix)):
            corr_value = corr_matrix.iloc[i, j]
            # 过滤绝对值小于阈值的边
            if abs(corr_value) > threshold:
                node1 = corr_matrix.index[i]
                node2 = corr_matrix.columns[j]
                G.add_edge(node1, node2, weight=corr_value)
    return G


def rename_func(old_name):
    """格式化菌名：去除前缀 s- 并缩写属名"""
    name_cut = old_name[2:]

    name = name_cut.replace('-', ' ')
    parts = name.split(maxsplit=1)

    if len(parts) == 2:
        return f"{parts[0][0].upper()}. {parts[1]}"

    return name


def extract_and_annotate_subgraph(G, target_nodes, good_bm, bad_bm):
    """
    提取诱导子图，添加好/坏菌属性，并按规则重命名节点。
    """
    valid_nodes = [n for n in target_nodes if n in G]
    subG = G.subgraph(valid_nodes).copy()

    attrs = {}
    for node in subG.nodes():
        if node in good_bm:
            status = 'Beneficial Biomarker'
        elif node in bad_bm:
            status = 'Pathogenic Biomarker'
        else:
            status = 'Unknown'
        attrs[node] = {'type': status}

    nx.set_node_attributes(subG, attrs)

    subG = nx.relabel_nodes(subG, rename_func, copy=False)

    subG.remove_nodes_from(list(nx.isolates(subG)))

    return subG


def add_node_size_by_rank(G, good_list, bad_list):
    """根据 biomarker 排名添加 node_weight 和 rank 属性"""
    attrs = {}
    for index, old_name in enumerate(good_list):
        new_name = rename_func(old_name)
        if new_name in G.nodes():
            rank = index + 1
            attrs[new_name] = {'biomarker_rank': rank, 'node_weight': 50 - rank + 1}

    for index, old_name in enumerate(bad_list):
        new_name = rename_func(old_name)
        if new_name in G.nodes():
            rank = index + 1
            attrs[new_name] = {'biomarker_rank': rank, 'node_weight': 50 - rank + 1}

    nx.set_node_attributes(G, attrs)


def add_edge_prev(G, ind_net_tensor, t2id):
    """计算边在参考样本张量中的流行率（非 NaN 比例）"""

    for u, v in G.edges():
        if u in t2id and v in t2id:
            i, j = t2id[u], t2id[v]
            sub = ind_net_tensor[:, i, j]
            prev = np.sum(~np.isnan(sub)) / len(ind_net_tensor)
            G.edges[u, v]['edge_prev'] = float(prev)

# Experiment 3: export core biomarker subgraphs for selected samples.
output_gml_dir = '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results/network/1.8/Antibiotic.intervention.longi/core.selected.individuals/'
os.makedirs(output_gml_dir, exist_ok=True)

all_target_biomarkers = top30_core_good_biomarker + top30_core_bad_biomarker
selected_sample_ids = df_ind['sample_id'].unique()
print(f"开始处理 {len(selected_sample_ids)} 个选定样本的网络...")

for sample_id in selected_sample_ids:
    sample_idx = sample_list.index(sample_id)
    canon_label = df_ind.loc[df_ind['sample_id'] == sample_id, 'canon'].values[0]
    if canon_label in ['A1', 'B1', 'B3', 'B5', 'C1', 'C3', 'C5']:
        corr_matrix = pd.DataFrame(individual_network[sample_idx, :, :], columns=list(taxa2id.keys()), index=list(taxa2id.keys()))
        corr_matrix.columns = list(taxa2id.keys())
        corr_matrix.index = list(taxa2id.keys())

        G_full = construct_graph(corr_matrix, threshold=0.01)

        subG = extract_and_annotate_subgraph(
            G=G_full,
            target_nodes=all_target_biomarkers,
            good_bm=top30_core_good_biomarker,
            bad_bm=top30_core_bad_biomarker
        )

        if len(subG.nodes()) > 0:
            add_node_size_by_rank(subG, top30_core_good_biomarker, top30_core_bad_biomarker)
            add_edge_prev(subG, individual_network, taxa2id)
            save_path = os.path.join(output_gml_dir, f"core_{sample_id}_{canon_label}.gml")
            # nx.write_gml(subG, save_path)
            print(f"Saved: {sample_id}_{canon_label} | 节点数: {len(subG.nodes())}, 边数: {len(subG.edges())}")
        else:
            print(f"Skipped: {sample_id}_{canon_label} | 原因: 过滤阈值或孤立点剔除后子网络为空。")

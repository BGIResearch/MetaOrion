# @Date    : 2026/6/10 14:37
# @Email   : zhangkexin2@genomics.cn

import os
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. Global Configurations & Paths
# ==========================================
# Define input and output paths for both diseases to prevent variable pollution
BASE_DIR = '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results'
# OUT_DIR = os.path.join(BASE_DIR, 'network/1.8/network.case.ctrl')
OUT_DIR = '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/figures/finetune/11.25/networks/4.1/tmp/'

CONFIG = {
    'IBD': {
        'case_corr': f'{BASE_DIR}/network/1.8/ibd/ibd.pretrain.case.edge.cv.4.9.csv',
        'ctrl_corr': f'{BASE_DIR}/network/1.8/ibd/ibd.pretrain.ctrl.edge.cv.4.9.csv',
        'biomarker': f'{BASE_DIR}/features/11.25/sort.multidisease.mean/IBD.biomarker.txt',
        'sample_list': f'{BASE_DIR}/network/1.8/ibd/ibd.pretrain.emb.samples.list.txt',
        'indiv_net': f'{BASE_DIR}/network/1.8/ibd/ibd.pretrain.emb.individual.npy',
        'taxa2id': f'{BASE_DIR}/network/1.8/ibd/ibd.pretrain.emb.individual.index.pkl',
        'out_case': f'{OUT_DIR}/ibd.pretrain.case.2biomarker.IG.0.01.gml',
        'out_ctrl': f'{OUT_DIR}/ibd.pretrain.ctrl.2biomarker.IG.0.01.gml'
    },
    'CRC': {
        # ⚠️ 请确认下方这两个 CRC 的 edge.cv.4.9.csv 路径是否正确
        'case_corr': f'{BASE_DIR}/network/1.8/crc.pretrain.case.edge.cv.csv',
        'ctrl_corr': f'{BASE_DIR}/network/1.8/crc.pretrain.ctrl.edge.cv.csv',
        'biomarker': f'{BASE_DIR}/features/11.25/sort.multidisease.mean/CRC.biomarker.txt',
        'sample_list': f'{BASE_DIR}/network/1.8/crc.pretrain.emb.samples.list.txt',
        'indiv_net': f'{BASE_DIR}/network/1.8/crc.pretrain.emb.cosine.individual.npy',
        'taxa2id': f'{BASE_DIR}/network/1.8/crc.pretrain.emb.cosine.individual.index.pkl',
        'out_case': f'{OUT_DIR}/crc.pretrain.case.2biomarker.IG.0.01.gml',
        'out_ctrl': f'{OUT_DIR}/crc.pretrain.ctrl.2biomarker.IG.0.01.gml'
    }
}


# ==========================================
# 2. Utility Functions
# ==========================================
def rename_func(old_name):
    """
    Format taxa names from 's__Bacteroides_fragilis' to 'B. fragilis'.
    """
    name = old_name[2:].replace('-', ' ')
    parts = name.split(maxsplit=1)
    if len(parts) == 2:
        return f"{parts[0][0].upper()}. {parts[1]}"
    return name


def construct_graph(corr_matrix, threshold):
    """
    Construct a networkx graph from a correlation matrix using a weight threshold.
    """
    G = nx.Graph()
    for i in range(len(corr_matrix)):
        for j in range(i + 1, len(corr_matrix)):
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) > threshold:
                node1 = corr_matrix.index[i]
                node2 = corr_matrix.columns[j]
                G.add_edge(node1, node2, weight=corr_value)
    return G


def extract_and_annotate_subgraph(G, target_nodes, good_biomarker, bad_biomarker):
    """
    Extract induced subgraph, annotate nodes with biomarker types, and rename.
    """
    valid_nodes = [n for n in target_nodes if n in G]
    subG = G.subgraph(valid_nodes).copy()

    attrs = {}
    for node in subG.nodes():
        if node in good_biomarker:
            status = 'Beneficial Biomarker'
        elif node in bad_biomarker:
            status = 'Pathogenic Biomarker'
        else:
            status = 'Unknown'
        attrs[node] = {'type': status}

    nx.set_node_attributes(subG, attrs)
    subG = nx.relabel_nodes(subG, rename_func, copy=False)
    subG.remove_nodes_from(list(nx.isolates(subG)))
    return subG


def edge_prevalence(ind_net, sample_indices, i, j):
    """Calculate the non-NaN percentage for a specific edge across samples."""
    if len(sample_indices) == 0: return np.nan
    sub = ind_net[sample_indices, i, j]
    present = np.sum(~np.isnan(sub))
    return present / len(sample_indices)


def node_prevalence(profile_df, samples, taxon):
    """Calculate the percentage of samples where the taxon abundance > 0."""
    valid_samples = [s for s in samples if s in profile_df.index]
    if len(valid_samples) == 0: return np.nan
    values = profile_df.loc[valid_samples, taxon]
    return (values > 0).sum() / len(valid_samples)


def annotate_case_ctrl_networks(case_graph, ctrl_graph):
    """
    Compare case and ctrl networks to annotate shared/unique nodes and edges.
    """
    # Edges
    case_edges = {frozenset(e) for e in case_graph.edges()}
    ctrl_edges = {frozenset(e) for e in ctrl_graph.edges()}
    shared_edges = case_edges & ctrl_edges

    for u, v in case_graph.edges():
        case_graph.edges[u, v]['origin'] = 'shared' if frozenset((u, v)) in shared_edges else 'only'
    for u, v in ctrl_graph.edges():
        ctrl_graph.edges[u, v]['origin'] = 'shared' if frozenset((u, v)) in shared_edges else 'only'

    # Nodes
    case_nodes = set(case_graph.nodes())
    ctrl_nodes = set(ctrl_graph.nodes())
    shared_nodes = case_nodes & ctrl_nodes

    for n in case_graph.nodes():
        case_graph.nodes[n]['origin'] = 'shared' if n in shared_nodes else 'only'
    for n in ctrl_graph.nodes():
        ctrl_graph.nodes[n]['origin'] = 'shared' if n in shared_nodes else 'only'


def add_node_size_by_rank(G, good_list, bad_list):
    """
    Add ranking and visual node_size attributes based on biomarker importance.
    """
    attrs = {}
    for index, old_name in enumerate(good_list):
        new_name = rename_func(old_name)
        if new_name in G.nodes():
            attrs[new_name] = {'biomarker_rank': index + 1, 'node_weight': 50 - index}

    for index, old_name in enumerate(bad_list):
        new_name = rename_func(old_name)
        if new_name in G.nodes():
            attrs[new_name] = {'biomarker_rank': index + 1, 'node_weight': 50 - index}

    nx.set_node_attributes(G, attrs)


# ==========================================
# 3. Main Pipeline Function
# ==========================================
def process_disease_network(disease, config, X, label_df):
    """
    End-to-end pipeline to generate, annotate, and save networkx GML files for a specific disease.
    """
    print(f"\n{'=' * 40}\nProcessing Network for: {disease}\n{'=' * 40}")

    # File Checks
    if not os.path.exists(config['case_corr']):
        print(f">>> [WARNING] File missing, skipping {disease}: {config['case_corr']}")
        return

    # 1. Load Correlation Matrices and Build Base Graphs
    print(">>> Loading correlation matrices and building base graphs...")
    case_corr_matrix = pd.read_csv(config['case_corr'], index_col=0)
    ctrl_corr_matrix = pd.read_csv(config['ctrl_corr'], index_col=0)
    case_graph = construct_graph(case_corr_matrix, 0.01)
    ctrl_graph = construct_graph(ctrl_corr_matrix, 0.01)

    # 2. Load Biomarkers (Top 30 Good/Bad)
    print(">>> Extracting Biomarker subgraphs...")
    biomarker_df = pd.read_csv(config['biomarker'], sep='\t', header=None)
    biomarker_df['weight.abs'] = biomarker_df[1].abs()
    biomarker_df = biomarker_df.sort_values('weight.abs', ascending=False)

    good_biomarkers = list(biomarker_df[biomarker_df[1] < 0].iloc[:30, 0])
    bad_biomarkers = list(biomarker_df[biomarker_df[1] > 0].iloc[:30, 0])
    top_biomarkers = bad_biomarkers + good_biomarkers

    # 3. Extract Subgraphs and Rename Nodes
    case_biomarker_graph = extract_and_annotate_subgraph(case_graph, top_biomarkers, good_biomarkers, bad_biomarkers)
    ctrl_biomarker_graph = extract_and_annotate_subgraph(ctrl_graph, top_biomarkers, good_biomarkers, bad_biomarkers)

    # 4. Load Sample Information & Filter Data
    print(">>> Calculating prevalence attributes...")
    with open(config['sample_list'], 'r') as f:
        sample_list = [line.strip() for line in f.readlines()]

    disease_profile = X.loc[sample_list, :]
    disease_profile = disease_profile.loc[:, ~(disease_profile == 0).all()]
    disease_profile.columns = [rename_func(i) for i in disease_profile.columns]

    case_samples = [i for i in sample_list if label_df.loc[i, 'disease_united'] == disease]
    ctrl_samples = [i for i in sample_list if label_df.loc[i, 'disease_united'] == 'healthy']

    # 5. Load Individual Networks & Taxa Dict
    individual_network = np.load(config['indiv_net'])
    with open(config['taxa2id'], 'rb') as f:
        taxa2id = pickle.load(f)
    taxa2id = {rename_func(k): v for k, v in taxa2id.items()}

    sample2idx = {s: i for i, s in enumerate(sample_list)}
    case_idx = [sample2idx[s] for s in case_samples if s in sample2idx]
    ctrl_idx = [sample2idx[s] for s in ctrl_samples if s in sample2idx]

    # 6. Add Node & Edge Prevalence
    for u, v in case_biomarker_graph.edges():
        case_biomarker_graph.edges[u, v]['prev'] = edge_prevalence(individual_network, case_idx, taxa2id[u], taxa2id[v])
    for u, v in ctrl_biomarker_graph.edges():
        ctrl_biomarker_graph.edges[u, v]['prev'] = edge_prevalence(individual_network, ctrl_idx, taxa2id[u], taxa2id[v])

    for node in case_biomarker_graph.nodes():
        case_biomarker_graph.nodes[node]['node_prev'] = node_prevalence(disease_profile, case_samples, node)
    for node in ctrl_biomarker_graph.nodes():
        ctrl_biomarker_graph.nodes[node]['node_prev'] = node_prevalence(disease_profile, ctrl_samples, node)

    # 7. Final Network Annotations
    print(">>> Annotating shared/unique entities and node sizes...")
    annotate_case_ctrl_networks(case_biomarker_graph, ctrl_biomarker_graph)
    add_node_size_by_rank(case_biomarker_graph, good_biomarkers, bad_biomarkers)
    add_node_size_by_rank(ctrl_biomarker_graph, good_biomarkers, bad_biomarkers)

    # 8. Save to GML
    os.makedirs(OUT_DIR, exist_ok=True)
    nx.write_gml(case_biomarker_graph, config['out_case'], stringizer=str)
    nx.write_gml(ctrl_biomarker_graph, config['out_ctrl'], stringizer=str)
    print(f">>> Successfully saved {disease} Case GML to: {config['out_case']}")
    print(f">>> Successfully saved {disease} Ctrl GML to: {config['out_ctrl']}")


# ==========================================
# 4. Main Execution Block
# ==========================================
if __name__ == '__main__':
    print(">>> Loading Global Microbiome Profile & Metadata...")
    # Load Abundance Profile
    X = pd.read_csv(
        '/bgi-seq-model-2/datasets/zhangkexin/meta_index/metaphlan4/fine-tune/curated_LiS_CRC_20204_v1121.train_test.profile',
        sep='\t', index_col=0).transpose()
    species_indices = [i for i in range(len(X.columns)) if X.columns[i].split('|')[-1].startswith('s__')]
    X = X.iloc[:, species_indices]
    X.columns = [col.split('|')[-1].replace('__', '-').replace(' ', '-').replace('_', '-').lower() for col in X.columns]

    # Load Metadata
    label_df = pd.read_csv(
        '/bgi-seq-model-2/datasets/zhangkexin/meta_index/metaphlan4/fine-tune/curated_LiS_CRC_20204_v0908.train_test.phe',
        sep='\t', index_col=0)

    # Execute Pipeline for both CRC and IBD
    for target_disease, conf in CONFIG.items():
        process_disease_network(target_disease, conf, X, label_df)

    print("\n>>> All tasks completed successfully.")

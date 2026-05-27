import pickle
import networkx as nx
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from scipy.stats import spearmanr, pearsonr
import seaborn as sns
from scipy import stats

from community import community_louvain
from collections import Counter
from networkx.algorithms.community import label_propagation_communities

def jaccard_similarity(G1, G2):
    # 统一边的顺序（无向图）
    edges1 = {tuple(sorted((u, v))): weight for u, v, weight in G1.edges(data='weight')}
    edges2 = {tuple(sorted((u, v))): weight for u, v, weight in G2.edges(data='weight')}
    intersection = set(edges1.keys()) & set(edges2.keys())
    union = set(edges1.keys()) | set(edges2.keys())

    consistent = 0
    inconsistent = 0
    for u, v in intersection:
        weight1 = edges1[(u, v)]
        weight2 = edges2[(u, v)]
        if (weight1 * weight2) > 0:  # 符号相同
            consistent += 1
        else:  # 符号相反（包括一正一零或一负一零）
            inconsistent += 1
    jaccard = len(intersection) / len(union) if union else 1.0

    edges1_w = [edges1.get(i, 0) for i in intersection]
    edges2_w = [edges2.get(i, 0) for i in intersection]
    edges_corr, p = pearsonr(edges1_w, edges2_w)
    print(
        f"G1中共有 {G1.number_of_nodes()} 个节点，{G1.number_of_edges()} 条边\n"
        f"G2中共有 {G2.number_of_nodes()} 个节点，{G2.number_of_edges()} 条边\n"
        f"G1 和 G2 共有 {len(intersection)} 条重复边\n"
        f"其中方向一致的有 {consistent} 条，方向不一致的有 {inconsistent} 条\n"
        f"G1和G2的边相关性为 {edges_corr}，p值为 {p} \n"
        f"Jaccard 相似度为: {jaccard:.4f}"
    )
    return len(intersection), consistent, inconsistent, jaccard


def matrix_to_edge_list(df, is_p=False):
    """
    将 n*n 矩阵转换为边列表，并过滤掉自相关(对角线)和重复边
    """
    # 确保索引和列名一致
    nodes = df.index.tolist()

    # 提取上三角阵（不含对角线 k=1）
    # mask 掉不需要的部分
    mask = np.triu(np.ones(df.shape), k=1).astype(bool)

    # 将矩阵转换为长表格格式 (node1, node2, weight)
    if is_p:
        edge_list = df.where(mask).stack(dropna=False).reset_index()
        edge_list.columns = ['node1', 'node2', 'weight']
        edge_list = edge_list.dropna(subset=['weight'])
    else:
        edge_list = df.where(mask).stack().reset_index()
        edge_list.columns = ['node1', 'node2', 'weight']

    # 生成唯一边ID，用于匹配
    # 对 node1 和 node2 进行行内排序，确保顺序固定
    edge_list = edge_list[edge_list['weight'] != 0]
    edge_list['edge'] = edge_list.apply(lambda x: " - ".join(sorted([str(x['node1']), str(x['node2'])])), axis=1)

    return edge_list[['edge', 'weight']]


def define_combined_status(row):
    s1 = row['status_1']
    s2 = row['status_2']
    if s1 == 'Significant' and s2 == 'Significant':
        return 'Both Significant'
    elif s1 == 'Significant':
        return 'C1 Only'
    elif s2 == 'Significant':
        return 'C2 Only'
    else:
        return 'Neither'


def contrust_graph(corr_matrix, threshold):
    G = nx.Graph()
    for i in range(len(corr_matrix)):
        for j in range(i + 1, len(corr_matrix)):
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) > threshold:
                node1 = corr_matrix.index[i]
                node2 = corr_matrix.columns[j]
                G.add_edge(node1, node2, weight=corr_value)
    # G = G.subgraph([n for n in G.nodes() if G.degree[n] >= 3])
    return G


def plot_matrix_correlation(matrix_file1, matrix_file2, p_file1=None, p_file2=None, label1="Condition A",
                            label2="Condition B", pvalue=0.01):
    # 1. 加载矩阵 (假设第一行和第一列是节点名)
    df1 = pd.read_csv(matrix_file1, index_col=0)
    df2 = pd.read_csv(matrix_file2, index_col=0)

    # 2. 转换为边列表
    edges1 = matrix_to_edge_list(df1)
    edges2 = matrix_to_edge_list(df2)

    # 加载 P-value 并标记显著性
    if p_file1:
        df1_p = pd.read_csv(p_file1, index_col=0)
        edges1_p = matrix_to_edge_list(df1_p, is_p=True)
        edges1 = pd.merge(edges1, edges1_p, on='edge', suffixes=('', '_p'))
        # 定义颜色分类：P <= 0.001 为显著，否则为不显著
        edges1['status'] = edges1['weight_p'].apply(lambda x: 'Significant' if x < pvalue else 'Non-significant')
    else:
        edges1['status'] = 'Significant'  # 如果没有P值文件，默认全部显著

    if p_file2:
        df2_p = pd.read_csv(p_file2, index_col=0)
        edges2_p = matrix_to_edge_list(df2_p, is_p=True)
        edges2 = pd.merge(edges2, edges2_p, on='edge', suffixes=('', '_p'))
        # 定义颜色分类：P <= 0.001 为显著，否则为不显著
        edges2['status'] = edges2['weight_p'].apply(lambda x: 'Significant' if x < pvalue else 'Non-significant')
    else:
        edges2['status'] = 'Significant'  # 如果没有P值文件，默认全部显著

    # 3. 取共有边
    merged = pd.merge(edges1, edges2, on='edge', how='inner', suffixes=('_1', '_2'))
    merged = merged[(merged['weight_1'] != 0) & (merged['weight_2'] != 0)]
    merged_filterp = merged[(merged['status_1'] == 'Significant') & (merged['status_2'] == 'Significant')]

    if merged.empty:
        print("没有找到共有边！请检查两个矩阵的节点名称是否一致。")
        return

    # 4. 计算相关性
    r, p = stats.pearsonr(merged_filterp['weight_1'], merged_filterp['weight_2'])

    # 5. 可视化
    plt.figure(figsize=(8, 8))
    plt.rcParams['font.family'] = 'Arial'
    plt.rc('font', size=16)
    plt.rcParams['pdf.fonttype'] = 42

    sns.set_style("ticks")
    # g = sns.jointplot(data=merged, x='weight_1', y='weight_2',
    #                   kind='reg',
    #                   scatter_kws={'alpha': 0.3, 's': 10, 'color': '#457B9D'},
    #                   line_kws={'color': '#E63946', 'lw': 2},
    #                   height=8)

    # 修改绘图颜色映射
    merged['combined_status'] = merged.apply(define_combined_status, axis=1)
    palette_map = {
        'Both Significant': '#1D3557',  # 深蓝
        'C1 Only': '#457B9D',  # 中蓝
        'C2 Only': '#A8DADC',  # 浅青/蓝
        'Neither': '#D3D3D3'  # 灰色
    }
    g = sns.jointplot(data=merged, x='weight_1', y='weight_2',
                      hue='combined_status',
                      palette=palette_map,
                      hue_order=['Both Significant', 'C1 Only', 'C2 Only', 'Neither'],  # 控制图例顺序
                      kind='scatter',
                      alpha=0.6, s=20,
                      height=8)

    # 由于使用了 hue，回归线需要手动在 ax_joint 上添加
    # sns.regplot(data=merged[merged['status'] == 'Significant'],
    #             x='weight_1', y='weight_2',
    #             ax=g.ax_joint, scatter=False, color='#E63946', truncate=False)

    # 添加 y=x 参考线
    ax = g.ax_joint
    # low, high = ax.get_xlim()[0], ax.get_xlim()[1]
    # ax.plot([low, high], [low, high], color='gray', linestyle='--', alpha=0.8, label='y=x')
    ax.axhline(0, color='gray', linestyle='--', linewidth=1, zorder=1)
    ax.axvline(0, color='gray', linestyle='--', linewidth=1, zorder=1)

    # 设置轴标签
    G1 = contrust_graph(df1, threshold=0)
    G2 = contrust_graph(df2, threshold=0)
    _, pos, neg, _ = jaccard_similarity(G1, G2)
    if p_file1 is not None and p_file2 is None:
        print('\nfiltered')
        df1[df1_p >= pvalue] = 0
        G1_p = contrust_graph(df1, 0)
        _, pos, neg, _ = jaccard_similarity(G1_p, G2)
    if p_file2 is not None and p_file1 is None:
        print('\nfiltered')
        df2[df2_p >= pvalue] = 0
        G2_p = contrust_graph(df2, 0)
        _, pos, neg, _ = jaccard_similarity(G1, G2_p)
    if p_file1 is not None and p_file2 is not None:
        print('\nfiltered')
        df1[df1_p >= pvalue] = 0
        G1_p = contrust_graph(df1, 0)

        df2[df2_p >= pvalue] = 0
        G2_p = contrust_graph(df2, 0)
        _, pos, neg, _ = jaccard_similarity(G1_p, G2_p)

    g.set_axis_labels(f'{label1} Correlation (C1)', f'{label2} Correlation (C2)')

    # 添加统计信息文本框
    text_str = f'Common Edges: {len(merged_filterp)} (+{pos};-{neg})\nPearson r: {r:.3f}\np-value: {p:.2e}'
    ax.text(0.05, 1, text_str, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.legend(loc='upper right', frameon=False, markerscale=1.5, bbox_to_anchor=(1.05, 0.35))

    plt.suptitle(f'{label1} vs {label2}')
    plt.tight_layout()
    # plt.savefig(f'/home/share/huadjyin/home/zhangkexin2/code/meta_index/experiment/figures/finetune/11.25/networks/{label1} vs {label2}.pdf', dpi=800)
    plt.show()


# 使用示例
dir = '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results/network/1.8/'
# df_common = plot_matrix_correlation(dir+'crc.test.pooling.spearman.prev.bootstrap.ci.edge.csv', dir+'crc.test.pooling.cosine.prev.bootstrap.ci.edge.csv',
#                                     # dir+'crc.test.split1.profile.clean.prev.p.change.new.csv',
#                                     label1="Emb Spearman", label2="Emb Cosine")
# df_common = plot_matrix_correlation(dir+'crc.test.split1.profile.clean.prev.spearman.0.001.csv', dir+'crc.test.pooling.cosine.prev.bootstrap.ci.edge.csv',
#                                     # dir+'crc.test.split1.profile.clean.prev.p.change.new.csv',
#                                     label1="Profile spearman (p <0.001)", label2="Emb Cosine.1")
# df_common = plot_matrix_correlation(dir+'fastspar.correlation.csv', dir+'crc.test.pooling.cosine.prev.bootstrap.ci.edge.csv',
#                                     dir+'fastspar.pvalues.csv',
#                                     "Profile Fastspar", "Emb Cosine",pvalue=0.05)
# df_common = plot_matrix_correlation(dir+'crc.test.split1.profile.clean.prev.spearman.change.new.csv', dir+'crc.test.pooling.cosine.prev.bootstrap.ci.edge.csv',
#                                     dir+'crc.test.split1.profile.clean.prev.p.change.new.csv',
#                                     label1="Profile Filtered0.spearman", label2="Emb cosine.onlyp")
# df_common = plot_matrix_correlation(dir+'crc.test.single.pooling.mean.cosine.nofilter.corr.csv', dir+'crc.test.pooling.cosine.prev.bootstrap.ci.edge.csv',
#                                     # dir+'crc.test.split1.profile.clean.prev.p.change.new.csv',
#                                     label1="Emb Cosine(nofiltered)", label2="Emb cosine")
# df_common = plot_matrix_correlation(dir+'pretrain.crc.test.single.pooling.mean.cosine.prev01.corr.csv', dir+'crc.test.pooling.cosine.prev.bootstrap.ci.edge.csv',
#                                     # dir+'crc.test.split1.profile.clean.prev.p.change.new.csv',
#                                     label1="Emb Cosine(pretrain)", label2="Emb cosine (finetune)")
# df_common = plot_matrix_correlation(dir+'crc.test.split1.profile.clean.prev.spearman.change.new.csv', dir+'pretrain.crc.test.single.pooling.mean.cosine.prev01.corr.csv',
#                                     dir+'crc.test.split1.profile.clean.prev.p.change.new.csv',
#                                     label1="Profile Filtered0.spearman", label2="Emb cosine (pretrain)")

# df_common = plot_matrix_correlation(dir+'fastspar.correlation.1e6.tsv', dir+'crc.test.split1.profile.clean.prev.spearman.0.001.csv',
#                                     dir+'fastspar.pvalues.1e6.tsv',
#                                     label1="Relative Abundance Fastspar", label2="Relative Abundance Spearman", pvalue=0.05)

# plot_matrix_correlation(dir+'crc.all.binabu.clean.spearman.csv', dir+'crc.all.predict.abu.spearman.csv',
#                                     dir+'crc.all.binabu.clean.p.csv',dir+'crc.all.predict.abu.p.csv',
#                                     label1="Binning Abu Spearman", label2="Predicted Abu Spearman")

plot_matrix_correlation(dir+'crc.all.predict.abu.spearman.csv', dir+'crc.all.profile.clean.spearman.csv',
                                    dir+'crc.all.predict.abu.p.csv',dir+'crc.all.profile.clean.p.csv',
                                    label1="Predicted Abu Spearman", label2="Relative Abu Spearman")

# plot_matrix_correlation(dir+'crc.pretrain.pooling.cosine.prev.bootstrap.ci.edge.cv.csv', dir+'crc.all.profile.clean.spearman.csv',
#                                     p_file2=dir+'crc.all.profile.clean.p.csv',
#                                     label1="Emb cosine", label2="Relative Abu Spearman")

# plot_matrix_correlation('/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results/network/12.18/crc.test.pretrain.pooling.cosine.prev.bootstrap.ci.edge.csv',
#         dir+'crc.all.profile.clean.spearman.1873.csv',
#                                     p_file2=dir+'crc.all.profile.clean.p.1873.csv',
#                                     label1="Emb cosine+bootstrap", label2="Relative Abu Spearman+bootstrap")


# plot_matrix_correlation(dir + 'tmp/crc.fusion.pooling.cosine.prev.bootstrap.ci.edge.cv.csv',
#                         dir + 'crc.all.profile.clean.spearman.bootstrap.csv',
#                         p_file2=dir + 'crc.all.profile.clean.p.bootstrap.csv',
#                         label1="Fusion Emb cosine+bootstrap", label2="Relative Abundance Spearman+bootstrap")



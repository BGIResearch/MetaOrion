import os
import random
import pickle
import warnings

import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from tqdm import tqdm
from functools import reduce
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

def merge_graphs_by_freq_weight(G_list, min_freq=0.1):
    """
    合并图列表，每条边的权重 = 共现频率 × 平均边权重
    :param G_list: List[networkx.Graph]，每个图代表一个样本
    :param min_freq: float，最小保留频率阈值（例如0.1 = 至少10%的样本中出现）
    :return: 合并后的图 G_combined
    """
    edge_weights = defaultdict(list)
    num_graphs = len(G_list)

    # 收集所有边的权重
    for G in G_list:
        for u, v, data in G.edges(data=True):
            edge = tuple(sorted((u, v)))
            edge_weights[edge].append(data.get("weight", 1))

    # 构建新的图
    G_combined = nx.Graph()

    for (u, v), weights in edge_weights.items():
        freq = len(weights) / num_graphs
        if freq >= min_freq:
            # 这里可以取边权重的中位数，可以取共现频率 × 平均边权重
            mean_weight = np.mean(weights)
            # std_weight = np.std(weights, ddof=1)
            # CV = std_weight / mean_weight
            # if CV < 0.4: # 越小越好
            # mean_weight = np.median(weights)
            combined_weight = mean_weight  # freq *
            G_combined.add_edge(u, v, weight=combined_weight, freq=freq)

    return G_combined


def filter_freq(input_dir, threshold):
    taxa_embed_dict = {}
    data_path = os.listdir(input_dir)
    for i in data_path:
        if i.endswith('.pkl'):
            data = pickle.load(open(input_dir + i, 'rb'))
            for m in range(len(data['taxa'])):
                if data['taxa'][m] not in taxa_embed_dict.keys():
                    taxa_embed_dict[data['taxa'][m]] = [data['embedding'][m]]
                else:
                    taxa_embed_dict[data['taxa'][m]].append(data['embedding'][m])
    for i in list(taxa_embed_dict.keys()):
        if len(taxa_embed_dict[i]) / len(data_path) < threshold:  # 0.01
            del taxa_embed_dict[i]
    taxa_list = list(taxa_embed_dict.keys())
    return taxa_list


def plot_graph(G):
    pos = nx.spring_layout(G, seed=42, k=0.8)

    node_color = "#9ecae1"  # 柔和蓝色
    node_size = 1000
    edges = G.edges(data=True)
    edge_colors = ['#2ca25f' if data['weight'] > 0 else '#de2d26' for _, _, data in edges]  # 绿为正，红为负
    edge_widths = [abs(data['weight']) * 3 for _, _, data in edges]  # 根据相关性调整粗细

    plt.figure(figsize=(10, 10))
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_color, alpha=0.9)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, alpha=0.7)
    nx.draw_networkx_labels(G, pos, font_size=8, font_color="black", font_weight='bold')
    plt.title("Microbial Correlation Network", fontsize=18, weight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def contrust_graph(corr_matrix, threshold):
    G = nx.Graph()
    for i in range(len(corr_matrix)):
        for j in range(i + 1, len(corr_matrix)):
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) >= threshold:
                node1 = corr_matrix.index[i]
                node2 = corr_matrix.columns[j]
                G.add_edge(node1, node2, weight=corr_value)
    # G = G.subgraph([n for n in G.nodes() if G.degree[n] >= 3])
    return G

def print_corr(df, p_value):
    triu_indices = np.triu_indices_from(df, k=1)
    pairs = [(df.index[i], df.columns[j], df.values[i, j]) for i, j in zip(*triu_indices)]
    top_pairs = sorted(pairs, key=lambda x: abs(x[2]), reverse=True)[:10]
    print("Top 10 strongest correlations:")
    for i, (s1, s2, val) in enumerate(top_pairs, 1):
        print(f"{i}. {s1} - {s2}: {val:.4f} ; p-value : {p_value.loc[s1, s2]:.4f}")


def gml_to_matrix(G, weight_attr='weight'):
    """
    将 GML 文件转换为带有节点名称的 pandas DataFrame (邻接矩阵)
    """

    # 2. 获取所有节点并排序 (确保 Case 和 Control 矩阵的行列顺序完全一致)
    nodes = sorted(list(G.nodes()))

    # 3. 转换为 pandas 邻接矩阵
    # weight 参数指定 GML 中存储数值的属性名，默认通常是 'weight'
    matrix_df = nx.to_pandas_adjacency(G, nodelist=nodes, weight=weight_attr)

    return matrix_df

# profile info
# profile_filtered = pd.read_csv(
#     '/home/share/huadjyin/home/zhangkexin2/data/meta_index/metaphlan4/pre-train/SZ-4D_3.3k_gut.profile',index_col=0, sep='\t').T
# col_idx = [i for i in profile_filtered.columns.tolist() if 's__' in i and 't__' not in i]
# profile_gut_filtered = profile_filtered.loc[:,col_idx]
# profile_gut_clean=profile_gut_filtered.iloc[1:, :]
# profile_gut_clean.columns = [i.split('|')[-1].replace('__', '-').replace('_', '-').replace(' ', '-').lower() for i in profile_gut_clean.columns.tolist()]


'''
# dir = '/home/share/huadjyin/home/zhangkexin2/code/meta_index/output/llama/v4/pretrain/crc.test/split1/'
dir = '/home/share/huadjyin/home/zhangkexin2/code/meta_index/output/llama/v4/PanDisease/Split5.specific.aug.Full.sortabu.ema.12.4/split1/crc.freeze.1.ema.12.9/best_ckpt/result/emb/'
label_df = pd.read_csv(
        '/home/share/huadjyin/home/chenjunhong/META_AI/dataset/meta_index/data_v0905/curated_LiS_CRC_20204_v0908.train_test.phe',
        sep='\t', index_col=0)

G_list = []
data_path = os.listdir(dir)
taxa_list = filter_freq(dir, 0.1) # 0.1
# taxa_list=[]
# for i in profile_gut_clean.columns:
#     if np.median(profile_gut_clean[i]) > 0.5 and (profile_gut_clean[i] != 0).sum() > 100:
#         taxa_list.append(i)
# 合并个体网络为群体网络
for i in tqdm(data_path):
    if i.endswith('.pkl'):
        try:
            data = pickle.load(open(dir + i, 'rb'))
            # if label_df.loc[i.split('.pkl')[0],'disease_united'] == 'CRC':
            df = pd.DataFrame(data['embedding'].T, columns=data['taxa'])

            # filter low-freq taxa
            indices = []
            for x in range(len(data['taxa'])):
                if data['taxa'][x] in taxa_list:
                    indices.append(x)
            df = df.iloc[:, indices]

            # spearman to contrust network
            # corr_matrix, p_matrix = spearmanr(df)
            # corr_matrix[p_matrix >= 0.001] = 0
            # corr_matrix = pd.DataFrame(corr_matrix, index=df.columns, columns=df.columns)
            # print_corr(corr_matrix, p_matrix)


            # cosine to contrust network
            corr_matrix = cosine_similarity(df.T)
            corr_matrix = pd.DataFrame(corr_matrix, index=df.columns, columns=df.columns)
            corr_matrix = corr_matrix.astype(float)

            G = contrust_graph(corr_matrix, threshold=0) # 0.3
            G_list.append(G)
        except:
            continue


G_combined = merge_graphs_by_freq_weight(G_list, min_freq=0) #0.01
cosine_df = gml_to_matrix(G_combined)
cosine_df.to_csv('/home/share/huadjyin/home/zhangkexin2/code/meta_index/experiment/results/network/12.18/crc.test.single.pooling.mean.cosine.prev01.corr.csv')
# nx.write_gml(G_combined, '/home/share/huadjyin/home/zhangkexin2/code/meta_index/experiment/results/network/12.18/crc.test.single.pooling.mean.cosine.nofilter.gml')

'''
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils import resample
from tqdm import tqdm


def bootstrap_network_variable_taxa(sample_data_list, save_dir, n_iterations=1000, alpha=0.05, cv_threshold=0.5, prevalence_threshold = 0.1):
    """
    处理不同样本菌种不一致情况下的 Bootstrap 网络计算

    参数:
    ----------
    sample_data_list : list of pd.DataFrame
        每个 DataFrame 的 index 是菌名，columns 是 Embedding 维度。
        例如：[df_sample1, df_sample2, ...]
    """

    # 1. 提取全局所有出现的唯一菌名
    all_taxa = sorted(list(set().union(*(df.index for df in sample_data_list))))
    n_taxa = len(all_taxa)
    n_samples = len(sample_data_list)
    taxa_to_idx = {taxa: i for i, taxa in enumerate(all_taxa)}

    print(f"检测到全局共有 {n_taxa} 种菌，样本量为 {n_samples}")

    # 2. 预计算每个样本的“对齐相似度矩阵”
    # 即使样本 A 只有 10 种菌，我们也会生成一个全局 (n_taxa, n_taxa) 的矩阵
    print("正在预计算并对齐每个个体的余弦相似度矩阵...")
    individual_networks = np.full((n_samples, n_taxa, n_taxa), np.nan) # 初始化为 NaN


    for i, df in enumerate(sample_data_list):
        # 计算该样本内存在的菌的相似度
        current_taxa = df.index
        sim_matrix = cosine_similarity(df.values)
        # sim_matrix, p_matrix = spearmanr(df.T.values)
        # sim_matrix[p_matrix >= 0.001] = 0

        # 将局部相似度映射到全局矩阵中
        indices = [taxa_to_idx[t] for t in current_taxa]
        # 使用 numpy 的花式索引进行填充
        # 这会自动处理缺失：没出现在该样本的菌，对应的行列保持为 0
        ixgrid = np.ix_(indices, indices)
        individual_networks[i][ixgrid] = sim_matrix

    # save
    import pickle
    pickle.dump(taxa_to_idx, open(save_dir+'pretrain.emb.individual.index.pkl', 'wb'))
    np.save(save_dir+'pretrain.emb.individual.npy', individual_networks)

    present_mask = ~np.isnan(individual_networks) & (individual_networks != 0)
    edge_prevalence = np.mean(present_mask, axis=0)
    keep_prevalence = edge_prevalence >= prevalence_threshold
    individual_networks[:, ~keep_prevalence] = np.nan

    # 3. Bootstrap 重采样 (方案一)
    boot_means = np.zeros((n_iterations, n_taxa, n_taxa))
    print(f"正在进行 {n_iterations} 次 Bootstrap...")
    for b in tqdm(range(n_iterations)):
        indices = resample(np.arange(n_samples))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            boot_means[b] = np.nanmean(individual_networks[indices], axis=0)
            boot_means[b][np.isnan(boot_means[b])] = 0
        # boot_means[b] = np.mean(individual_networks[indices], axis=0) # 这里不是对所有不为0的计算的均值，而是包含了0
    # np.save(save_dir + 'crc.case.pretrain.emb.individual.popu.bootstrap.npy', boot_means)

    # 4. 统计汇总
    final_mean = np.nanmean(individual_networks, axis=0) # 这里也是
    final_mean[np.isnan(final_mean)] = 0
    lower_bound = np.percentile(boot_means, (alpha / 2) * 100, axis=0)
    upper_bound = np.percentile(boot_means, (1 - alpha / 2) * 100, axis=0)

    boot_mu = np.mean(boot_means, axis=0)
    boot_sigma = np.std(boot_means, axis=0)
    cv = np.divide(boot_sigma, boot_mu, out=np.zeros_like(boot_sigma), where=boot_mu != 0)

    # 判断显著性：
    not_significant = (lower_bound <= 0) & (upper_bound >= 0)# | (cv >= cv_threshold)
    not_significant_stable = (lower_bound <= 0) & (upper_bound >= 0) | (cv >= cv_threshold)

    # 克隆一份均值矩阵，并将不显著的边设为 0
    filtered_mean = final_mean.copy()
    filtered_mean[not_significant] = 0

    filtered_cv_mean = final_mean.copy()
    filtered_cv_mean[not_significant_stable] = 0

    # 将结果转回 DataFrame 方便查看
    return {
        "mean": pd.DataFrame(final_mean, index=all_taxa, columns=all_taxa),
        "lower_ci": pd.DataFrame(lower_bound, index=all_taxa, columns=all_taxa),
        "upper_ci": pd.DataFrame(upper_bound, index=all_taxa, columns=all_taxa),
        "taxa_list": all_taxa,
        "mean_CI_filtered": pd.DataFrame(filtered_mean, index=all_taxa, columns=all_taxa),
        "mean_CI_CV_filtered": pd.DataFrame(filtered_cv_mean, index=all_taxa, columns=all_taxa),
        "bootstrap_popu_mean": boot_means
    }


def create_individual_emb_list(name, save_dir):
    # label_df = pd.read_csv(
    #     '/bgi-seq-model-2/datasets/zhangkexin/meta_index/metaphlan4/fine-tune/curated_LiS_CRC_20204_v0908.train_test.phe',
    #     sep='\t', index_col=0)
    label_df = pd.read_csv('/bgi-seq-model-2/datasets/zhangkexin/meta_index/metaphlan4/fine-tune/Antibiotic.intervention.longi/supp.tsv', sep='\t',index_col=1)
    # X = pd.read_csv(
    #     '/bgi-seq-model-2/datasets/zhangkexin/meta_index/metaphlan4/fine-tune/curated_LiS_CRC_20204_v1121.train_test.profile',
    #     sep='\t', index_col=0).transpose()
    X = pd.read_csv(
        '/bgi-seq-model-2/datasets/zhangkexin/meta_index/metaphlan4/fine-tune/Antibiotic.intervention.longi/merged_abundance_table_supp.profile',
        sep='\t', index_col=0).T
    index = []
    for i in range(len(X.axes[1])):
        if X.columns[i].split('|')[-1].split('__')[0] == 's':
            index.append(i)
    X = X.iloc[:, index]
    X.columns = [i.split('|')[-1].replace('__', '-').replace(' ', '-').replace('_', '-').lower() for i in X.columns]
    X.index = [i.split('.mp4')[0] for i in X.index]

    # X = pd.read_csv(
    #     '/bgi-seq-model-2/datasets/zhangkexin/meta_index/metaphlan4/fine-tune/SZ_4D_longitudinal_bmi/longitude_50.profile',
    #     sep='\t', index_col=0).T
    # X.columns = X.columns.map(lambda x: 's-'+x.split('(SGB')[0].replace('_', '-').lower())

    samples=[]
    sample_list = []
    # dir = '/bgi-seq-model-2/codes/zhangkexin/meta_index/output/llama/v4/pretrain/ibd/13_Ning_2023/'
    # dir = '/bgi-seq-model-2/codes/zhangkexin/meta_index/output/llama/v4/pretrain/crc/all.predict.abu.raw/'
    # dir = '/bgi-seq-model-2/codes/zhangkexin/meta_index/output/llama/v4/pretrain/abundance_sim/emb/'
    dir = '/bgi-seq-model-2/codes/zhangkexin/meta_index/output/llama/v4/pretrain/Antibiotic.intervention.longi/'
    data_path = os.listdir(dir)
    prev_filtered_taxa = filter_freq(dir, 0.1)  # filter prev < 0.1
    for i in tqdm(data_path):
        if i.endswith('.pkl') and i.split('.pkl')[0] in list(X.index):
            data = pickle.load(open(dir + i, 'rb'))
            # if label_df.loc[i.split('.pkl')[0], 'disease_united'] != 'healthy':
            # if label_df.loc[i.split('.pkl')[0], 'canon'][0].upper() == 'C':
            if 1:
                df = pd.DataFrame(data[name], index=data['taxa'])

                # filter abu < 0.01
                abu = X.loc[i.split('.pkl')[0], :]
                abu_filtered_taxa = list(abu.loc[abu >= 0.01].index)
                #
                indices = []
                for x in range(len(data['taxa'])):
                    if data['taxa'][x] in prev_filtered_taxa and data['taxa'][x] in abu_filtered_taxa:
                        indices.append(x)
                    # if data['taxa'][x] in abu_filtered_taxa:  # 只要个体网络的话，不过滤prev了
                    #     indices.append(x)
                df = df.iloc[indices, :]

                if len(df) == 0:
                    print(i.split('.pkl')[0])
                    continue
                else:
                    sample_list.append(df)
                    samples.append(i.split('.pkl')[0])
    with open(save_dir+'pretrain.emb.samples.list.txt', 'w') as f:
        for i in samples:
            f.write(i+'\n')
    return sample_list


# 运行
save_dir = '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results/network/1.8/Antibiotic.intervention.longi.complete/'
os.makedirs(save_dir, exist_ok=True)
samples_fusion = create_individual_emb_list('embedding', save_dir)
# samples_abu = create_individual_emb_list('abu_embedding')
# samples_hidden = create_individual_emb_list('hidden_embedding')
results_fusion = bootstrap_network_variable_taxa(samples_fusion, save_dir, n_iterations=1000)
# results_abu = bootstrap_network_variable_taxa(samples_abu, n_iterations=1000)
# results_hidden = bootstrap_network_variable_taxa(samples_hidden, n_iterations=1000)
# pickle.dump(results, open('/home/share/huadjyin/home/zhangkexin2/code/meta_index/experiment/results/network/12.18/crc.test.pooling.prev.bootstrap.pkl', 'wb'))

results_fusion['mean_CI_filtered'].to_csv('/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results/network/1.8/Antibiotic.intervention.longi/stage.C.pretrain.edge.csv')
results_fusion['mean_CI_CV_filtered'].to_csv('/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results/network/1.8/Antibiotic.intervention.longi/stage.C.pretrain.edge.cv.csv')

# results_abu['mean_CI_filtered'].to_csv('/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results/network/1.8/tmp/crc.abu.pooling.cosine.prev.bootstrap.ci.edge.csv')
# results_abu['mean_CI_CV_filtered'].to_csv('/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results/network/1.8/tmp/crc.abu.pooling.cosine.prev.bootstrap.ci.edge.cv.csv')
#
# results_hidden['mean_CI_filtered'].to_csv('/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results/network/1.8/tmp/crc.hidden.pooling.cosine.prev.bootstrap.ci.edge.csv')
# results_hidden['mean_CI_CV_filtered'].to_csv('/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results/network/1.8/tmp/crc.hidden.pooling.cosine.prev.bootstrap.ci.edge.cv.csv')

# # 查看某一对菌的置信区间
# taxa_a, taxa_b = results['taxa_list'][0], results['taxa_list'][1]
# print(f"\n{taxa_a} vs {taxa_b}:")
# print(f"均值: {results['mean'].loc[taxa_a, taxa_b]:.4f}")
# print(f"CI: [{results['lower_ci'].loc[taxa_a, taxa_b]:.4f}, {results['upper_ci'].loc[taxa_a, taxa_b]:.4f}]") # 95% 置信区间
# # print(f"是否显著: {results['is_significant'].loc[taxa_a, taxa_b}")


def jaccard_similarity(G1, G2):
    # 统一边的顺序（无向图）
    edges1 = {tuple(sorted((u, v))):weight for u, v, weight in G1.edges(data='weight')}
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


    edges1_w = [edges1.get(i,0) for i in union]
    edges2_w = [edges2.get(i,0) for i in union]
    edges_corr, p = spearmanr(edges1_w, edges2_w)
    print(
        f"G1中共有 {G1.number_of_nodes()} 个节点，{G1.number_of_edges()} 条边\n"
        f"G2中共有 {G2.number_of_nodes()} 个节点，{G2.number_of_edges()} 条边\n"
        f"G1 和 G2 共有 {len(intersection)} 条重复边\n"
        f"其中方向一致的有 {consistent} 条，方向不一致的有 {inconsistent} 条\n"
        f"G1和G2的边相关性为 {edges_corr}，p值为 {p} \n"
        f"Jaccard 相似度为: {jaccard:.4f}"
    )
    return len(intersection), consistent, inconsistent, jaccard


# # 对比使用bootstrap构建的cosine图和原来的cosine图的区别
# # 注意：原来的是没过滤的低频菌的
# bootstrap_result = pickle.load(open('/home/share/huadjyin/home/zhangkexin2/code/meta_index/experiment/results/network/12.18/crc.test.pooling.cosine.bootstrap.pkl', 'rb'))
# bootstrap_cosine_ = bootstrap_result['mean']
# raw_cosine_network = contrust_graph(bootstrap_cosine_, 0.5) # 也可以用crc.test.single.pooling.mean.cosine.nofilter.corr.csv，应该结果是一样的
# bootstrap_cosine = pd.DataFrame(index=bootstrap_cosine_.index, columns=bootstrap_cosine_.index)
# for i in bootstrap_cosine.index:
#     for j in bootstrap_cosine.columns:
#         if bootstrap_result['lower_ci'].loc[i, j] <= 0 and bootstrap_result['upper_ci'].loc[i, j] >= 0:
#             bootstrap_cosine.loc[i, j] = 0
#         else:
#             bootstrap_cosine.loc[i, j] = bootstrap_cosine_.loc[i, j]
# bootstrap_cosine_network = contrust_graph(bootstrap_cosine, 0.5)
# jaccard_similarity(raw_cosine_network, bootstrap_cosine_network)
#
# #发现profile整体计算spearman和只基于两个菌均不为0的样本计算
# # 基于两个菌均不为0的样本计算profile的相关系数，我这边发现构建出的图的边相较于用profile整体计算多了很多（corr >= 0.5, p < 0.001)，而且与整体计算的图相关性不咋高
# profile_corr = pd.read_csv('/home/share/huadjyin/home/zhangkexin2/code/meta_index/experiment/results/network/12.18/crc.test.split1.profile.clean.spearman.0.001.csv',index_col=0)
# profile_network = contrust_graph(profile_corr, 0.5)
#
# profile_clean_corr = pd.read_csv('/home/share/huadjyin/home/zhangkexin2/code/meta_index/experiment/results/network/12.18/crc.test.split1.profile.clean.spearman.change.csv', index_col=0)
# profile_clean_p = pd.read_csv('/home/share/huadjyin/home/zhangkexin2/code/meta_index/experiment/results/network/12.18/crc.test.split1.profile.clean.p.change.csv', index_col=0)
# profile_clean_corr[profile_clean_p >= 0.001]=0
# profile_network_new = contrust_graph(profile_clean_corr, 0.5)
#
# jaccard_similarity(profile_network, profile_network_new) #相关性差0.021175434613010392, p值也不显著，其中方向一致的有 306 条，方向不一致的有 228 条
#






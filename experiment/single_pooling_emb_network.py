import os
import pickle
import warnings

import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils import resample
from tqdm import tqdm

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
    label_df = pd.read_csv('/bgi-seq-model-2/datasets/zhangkexin/meta_index/metaphlan4/fine-tune/Antibiotic.intervention.longi/supp.tsv', sep='\t',index_col=1)

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


    samples=[]
    sample_list = []
    dir = '/bgi-seq-model-2/codes/zhangkexin/meta_index/output/llama/v4/pretrain/Antibiotic.intervention.longi/'
    data_path = os.listdir(dir)
    prev_filtered_taxa = filter_freq(dir, 0.1)  # filter prev < 0.1
    for i in tqdm(data_path):
        if i.endswith('.pkl') and i.split('.pkl')[0] in list(X.index):
            data = pickle.load(open(dir + i, 'rb'))
            # if label_df.loc[i.split('.pkl')[0], 'disease_united'] != 'healthy':
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


if __name__ == '__main__':
    save_dir = '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results/network/1.8/Antibiotic.intervention.longi.complete/'
    os.makedirs(save_dir, exist_ok=True)
    samples_fusion = create_individual_emb_list('embedding', save_dir)
    results_fusion = bootstrap_network_variable_taxa(samples_fusion, save_dir, n_iterations=1000)

    results_fusion['mean_CI_filtered'].to_csv('/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results/network/1.8/Antibiotic.intervention.longi/stage.C.pretrain.edge.csv')
    results_fusion['mean_CI_CV_filtered'].to_csv('/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results/network/1.8/Antibiotic.intervention.longi/stage.C.pretrain.edge.cv.csv')


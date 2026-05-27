import os
import json
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib import rcParams
from matplotlib import font_manager as fm
from tqdm import tqdm
from collections import defaultdict


# 对泛病进行分类别的特征归因


# 统计每一个split里各个疾病类别的数目
path='/bgi-seq-model-2/datasets/zhangkexin/meta_index/preprocess/metaphlan4/fine-tune/nov.specific.random.5'
sample_class_split={}
# sample_class = {}

for i in range(1, 6):
    split = 'split'+str(i)+'.change'
    sample_class = {}
    with open(os.path.join(path, split,'datapath.pandisease.test'),'r') as f:
        for line in f.readlines():
            sample = json.load(open(line.strip(), 'r'))

            if sample['disease_united'] not in sample_class.keys():
                sample_class[sample['disease_united']]=[os.path.basename(line.strip()).split('.json')[0]]
            else:
                sample_class[sample['disease_united']].append(os.path.basename(line.strip()).split('.json')[0])
    sample_class_split[split]=sample_class

dir = '/bgi-seq-model-2/codes/zhangkexin/meta_index/output/llama/v4/PanDisease/Split5.specific.aug.Full.sortabu.ema.12.4/'

# 根据这些样本名，将特征归因找到的每个样本的权重收集起来，取平均，得到五个spilt的weight
class_sorted_tax = defaultdict(dict)
for key in tqdm(list(sample_class.keys())):
    # tax_weight = {}
    for i in range(1, 6):
        split = 'split' + str(i)+'.change'
        tax_weight = {}
        file_num = len(sample_class_split[split][key])
        for i in sample_class_split[split][key]:
            if i+'.pkl' in os.listdir(os.path.join(dir,split.split('.change')[0],'best_ckpt/result/tax_weight/')):
                sample = pickle.load(open(os.path.join(dir,split.split('.change')[0],'best_ckpt/result/tax_weight/',i + '.pkl'), 'rb'))
                for j in range(len(sample['taxa'])):
                    tax = sample['taxa'][j]
                    if tax not in tax_weight.keys():
                        tax_weight[tax] = [sample['weight'][j]]
                    else:
                        tax_weight[tax].append(sample['weight'][j])

        for j in tqdm(list(tax_weight.keys())):
            if len(tax_weight[j]) / file_num < 0.05:  # 0.01
                del tax_weight[j]
        tax_mean_w = {x: np.sum(tax_weight[x])/file_num for x in list(tax_weight.keys())}
        # tax_mean_w = {x : np.median(tax_weight[x]) for x in list(taw2w2x_weight.keys())}
        sorted_tax = dict(sorted(tax_mean_w.items(), key=lambda item: item[1], reverse=True))
        class_sorted_tax[key][split] = sorted_tax

merged_data = defaultdict(lambda: defaultdict(list))
# 第一步：收集所有菌群的所有权重
for disease_key, split_data in class_sorted_tax.items():
    for split_name, taxa_weights in split_data.items():
        for taxa, weight in taxa_weights.items():
            merged_data[disease_key][taxa].append(weight)

final_result = {}
for disease_key, taxa_data in merged_data.items():
    final_result[disease_key] = {
        'merged_taxa': taxa_data,
        'mean_weights': {
            taxa: np.mean(weights)
            for taxa, weights in taxa_data.items()
        }
    }



def plot_top30_taxa_boxplots_custom(merged_result, save_path=None):
    """
    针对特定4种疾病，绘制前30个菌群权重的箱线图+散点图
    排版顺序：左上(adenoma)、右上(CRC)、左下(IBD)、右下(IBS)
    排序逻辑：先筛选绝对值Top30，负向菌群（蓝色）在前按绝对值降序，正向菌群（红色）在后按数值降序。
    """
    # 设置学术图表样式
    rcParams.update({
        'font.family': 'Arial',
        'font.size': 20,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 10
    })
    # 【核心修改区：严格按照你要求的方位调整列表顺序】
    # 索引0(左上), 索引1(右上), 索引2(左下), 索引3(右下)
    target_diseases = ['adenoma', 'CRC', 'IBD', 'IBS']
    color_pos = '#E58579'  # 正向：红色
    color_neg = '#8AB1D2'  # 负向：蓝色
    # 创建2x2大图 (高度设为 24 保证30个菌的文字不重叠)
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 24), squeeze=False)
    axes = axes.flatten()
    for idx, disease in enumerate(target_diseases):
        ax = axes[idx]
        # 兼容性处理：如果字典里既没有小写也没有大写，则跳过
        dict_key = disease
        if disease not in merged_result:
            if disease.capitalize() in merged_result:
                dict_key = disease.capitalize()
            else:
                ax.axis('off')
                continue
        data = merged_result[dict_key]
        mean_weights = data['mean_weights']
        # --- 筛选与分区排序逻辑 ---
        # 1. 先挑出绝对值最大的 Top30
        top30_items = sorted(mean_weights.items(), key=lambda x: abs(x[1]), reverse=True)[:30]
        top30_taxa_names = [taxa for taxa, _ in top30_items]
        # 2. 将 Top30 划分为负向和正向阵营
        neg_taxa = [t for t in top30_taxa_names if mean_weights[t] < 0]
        pos_taxa = [t for t in top30_taxa_names if mean_weights[t] >= 0]
        # 3. 各自内部排序：
        # 负向（蓝）：按绝对值降序（最负的在最前面）
        neg_taxa = sorted(neg_taxa, key=lambda x: abs(mean_weights[x]), reverse=True)
        # 正向（红）：按数值降序（最大的在最前面）
        pos_taxa = sorted(pos_taxa, key=lambda x: mean_weights[x], reverse=False)
        # 4. 拼合得到最终展示顺序
        final_taxa_order = neg_taxa + pos_taxa
        # ------------------------------------
        # 为每个菌群分配颜色
        taxa_palette = {taxa: color_pos if mean_weights[taxa] > 0 else color_neg for taxa in final_taxa_order}
        # 准备绘图数据
        plot_data = []
        for taxa in final_taxa_order:
            weights = data['merged_taxa'].get(taxa, [])
            for weight in weights:
                plot_data.append({
                    'Taxa': taxa,
                    'Weight': weight,
                    'Mean_Weight': mean_weights[taxa]
                })
        df = pd.DataFrame(plot_data)
        # 绘制箱线图 (使用 order 参数强制按照我们的最终顺序从上到下排列)
        sns.boxplot(data=df, y='Taxa', x='Weight', ax=ax,
                    width=0.6, showfliers=False,
                    hue='Taxa', palette=taxa_palette, dodge=False,
                    order=final_taxa_order)
        # 绘制散点图 (同样使用 order 参数对应)
        sns.stripplot(data=df, y='Taxa', x='Weight', ax=ax,
                      color='black', size=4, alpha=0.7, jitter=0.2,
                      order=final_taxa_order)
        # 隐藏 seaborn 自动生成的图例
        if ax.legend_:
            ax.legend_.remove()
        # 添加一条垂直于 x=0 的灰色虚线作为正负分界线
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=1.5, zorder=0)
        # 设置Y轴标签（斜体）
        prop = fm.FontProperties(style='italic', size=14)
        labels = [i[2:].replace('-', ' ').capitalize() for i in final_taxa_order]
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels, fontproperties=prop)
        # 设置标题和标签
        ax.set_title(disease, fontsize=20, fontweight='bold')
        ax.set_xlabel("Weight", fontsize=16)
        ax.set_ylabel("")
        ax.grid(False)
    # 调整布局
    plt.tight_layout()
    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=800, bbox_inches='tight')
    plt.show()
# 调用方式：
plot_top30_taxa_boxplots_custom(final_result, save_path='/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/figures/finetune/11.25/features/4.disease.biomarker.top30.pdf')

# @Date    : 2026/6/10
# @Email   : zhangkexin2@genomics.cn

import os
import pickle
import random
import colorsys
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import kruskal
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, silhouette_samples
import umap
import pycountry_convert as pc

warnings.filterwarnings('ignore')


# ==========================================
# 1. Utility Functions (数据与颜色处理箱)
# ==========================================
def generate_random_colors(n=100, seed=111):
    """生成低饱和度、高明度的马卡龙色系调色板"""
    random.seed(seed)
    colors = []
    for _ in range(n):
        h = random.random()
        s = random.uniform(0.4, 0.6)
        v = random.uniform(0.8, 1.0)
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        colors.append((r, g, b))

    colors_hex = ['#{:02x}{:02x}{:02x}'.format(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in colors]
    return sns.color_palette(colors_hex)


def filter_profile(X):
    """过滤丰度表，仅保留 s__ 级别的特征，并清洗列名"""
    index = [i for i in range(len(X.columns)) if X.columns[i].split('|')[-1].split('__')[0] == 's']
    X = X.iloc[:, index]
    X.columns = [i.split('|')[-1].replace('__', '-').replace(' ', '-').replace('_', '-').lower() for i in X.columns]
    return X


def get_continent_from_country(country_name):
    """将国家名转换为大洲名"""
    if pd.isna(country_name):
        return None
    try:
        country_alpha2 = pc.country_alpha3_to_country_alpha2(country_name)
        continent_code = pc.country_alpha2_to_continent_code(country_alpha2)
        return pc.convert_continent_code_to_continent_name(continent_code)
    except:
        return None


# ==========================================
# 2. Plotting Functions (绘图核心逻辑)
# ==========================================
def plot_tsne_highlight(X, y, target_labels, level):
    """绘制带特定类别标签高亮与箭头指示的 t-SNE 图"""
    plt.style.use('default')
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 28, 'pdf.fonttype': 42})

    all_X_embedded = TSNE(n_components=2, perplexity=10, learning_rate='auto', init='pca',
                          random_state=42).fit_transform(X)
    df = pd.DataFrame(all_X_embedded, columns=["Dim1", "Dim2"])
    df['label'] = y

    highlight_colors = generate_random_colors(n=len(target_labels), seed=111)
    other_color = "#E0E0E0"

    fig, ax = plt.subplots(figsize=(8, 8), facecolor='white')

    # 绘制非目标点
    df_others = df[~df['label'].isin(target_labels)]
    ax.scatter(df_others["Dim1"], df_others["Dim2"], c=other_color, s=8, alpha=0.3, label='Others', edgecolors='none')

    # 绘制目标高亮散点与标签
    base_offsets = [(20, 20), (-60, 30), (20, -40), (-40, -40)]
    for i, target in enumerate(target_labels):
        if target in df['label'].values:
            point_data = df[df['label'] == target]
            color = highlight_colors[i]

            ax.scatter(point_data["Dim1"], point_data["Dim2"], c=color, s=6, alpha=0.9, zorder=3, linewidths=0)

            first_point = point_data.iloc[0]
            offset = base_offsets[i % len(base_offsets)]
            target_text = target.lstrip('s-').replace('-', ' ').capitalize()
            ax.annotate(
                target_text,
                xy=(first_point['Dim1'], first_point['Dim2']),
                xytext=offset,
                textcoords='offset points',
                fontsize=10,
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8, ec=color, lw=1.5),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5, connectionstyle="arc3,rad=0.1"),
                zorder=5
            )

    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    sns.despine(ax=ax, left=False, bottom=False, right=False, top=False)

    save_path = f'/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/figures/pretrain/crc.owen.459.top100tax.embed.highlight.{level}.pdf'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # plt.savefig(save_path, dpi=800, bbox_inches='tight')
    plt.show()


def plot_tsne_abu(X, y, level):
    """绘制丰度连续分布的 t-SNE 图"""
    plt.style.use('default')
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 28, 'pdf.fonttype': 42})
    plt.figure(figsize=(9.5, 8))

    all_X_embedded = TSNE(n_components=2, perplexity=10, learning_rate='auto', init='pca',
                          random_state=42).fit_transform(X)
    df = pd.DataFrame(all_X_embedded, columns=["Dim1", "Dim2"])
    df['value'] = y

    # 连续染色：过滤到 0–50
    df_plot = df[(df['value'] >= 0) & (df['value'] <= 50)]

    sc = plt.scatter(
        df_plot["Dim1"], df_plot["Dim2"], c=df_plot["value"],
        cmap="coolwarm", s=6, alpha=0.8, linewidths=0
    )

    cbar = plt.colorbar(sc, shrink=0.8)
    cbar.set_ticks([0, 10, 20, 30, 40, 50])
    cbar.ax.tick_params(labelsize=20)
    cbar.ax.spines[:].set_visible(False)
    cbar.draw_all()
    cbar.set_label("Abundance Range")
    cbar.outline.set_visible(False)

    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    sns.despine(left=False, bottom=False, right=False, top=False)

    save_path = f'/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/figures/pretrain/crc.owen.459.top100tax.embed.{level}.pdf'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # plt.savefig(save_path, dpi=800, bbox_inches='tight')
    plt.show()


def plot_umap(X, y, model_name, save_dir, is_legend=None, hue_order=None):
    """绘制 UMAP 聚类分布图，并保存 Silhouette Score"""
    plt.style.use('default')
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 18, 'pdf.fonttype': 42})
    plt.figure(figsize=(12, 8), facecolor='white')

    all_X_embedded = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric='euclidean',
                               random_state=42).fit_transform(X)

    # 轮廓系数计算
    sample_silhouette_values = silhouette_samples(all_X_embedded, y)
    cluster_silhouette_scores = {}
    cluster_sizes = {}
    for label in set(y):
        mask = [_ == label for _ in y]
        cluster_silhouette_scores[label] = np.mean(sample_silhouette_values[mask])
        cluster_sizes[label] = np.sum(mask)

    df = pd.DataFrame(all_X_embedded, columns=["Dim1", "Dim2"])
    df['label'] = y

    top_ranks = list(df['label'].value_counts().index)
    if hue_order is None:
        hue_order = top_ranks

    custom_palette = generate_random_colors(40, seed=101)

    if is_legend:
        sns.lmplot(
            x="Dim1", y="Dim2", data=df[df['label'].isin(top_ranks)],
            hue="label", hue_order=hue_order, palette=sns.color_palette("tab20", 20),
            fit_reg=False, legend=False, scatter_kws={"s": 5}
        )
        sns.despine(left=False, bottom=False, right=False, top=False)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, frameon=False, labels=hue_order, markerscale=5)
    else:
        sns.lmplot(
            x="Dim1", y="Dim2", data=df[df['label'].isin(top_ranks)],
            hue="label", hue_order=hue_order, palette=custom_palette,
            fit_reg=False, legend=False, scatter_kws={"s": 5}
        )
        sns.despine(left=False, bottom=False, right=False, top=False)
        plt.legend(bbox_to_anchor=(1.0, -0.12), loc=1, ncol=4, frameon=False,
                   labelspacing=0.7, handlelength=0.5, columnspacing=0.7,
                   handletextpad=0.7, labels=hue_order, markerscale=5)

    plt.grid(False)

    os.makedirs(save_dir, exist_ok=True)
    pickle.dump(cluster_silhouette_scores, open(f"{save_dir}{model_name}.pkl", 'wb'))
    # plt.savefig(f"{save_dir}{model_name}.pdf", bbox_inches='tight', format='pdf')
    plt.show()


def add_kruskal_significance(ax, data, x_or_y, group_name, orientation='v'):
    """基于 Kruskal-Wallis 检验计算 P 值并在子图上添加显著性星星"""
    groups = [group[x_or_y].values for name, group in data.groupby(group_name)]
    stat, p = kruskal(*groups)

    if p < 0.001:
        sig = '***'
    elif p < 0.01:
        sig = '**'
    elif p < 0.05:
        sig = '*'
    else:
        sig = ''

    if orientation == 'v':
        ax.text(0.5, 0.98, sig, ha='center', va='bottom', fontsize=20, transform=ax.transAxes)
    else:
        ax.text(0.02, 0.5, sig, ha='right', va='center', fontsize=20, rotation=90, transform=ax.transAxes)


def plot_scatter_with_boxplots(X, Y, disease, save_path=None):
    """绘制主散点图 + 附带边缘分布的箱线图 (GridSpec 2x2)"""
    df = pd.DataFrame({'PC1': X[:, 0], 'PC2': X[:, 1], 'project': Y, 'disease': disease})

    project_colors = {
        'Healthy': '#8FB4DC', 'IBD': '#c899d2', 'IBS': '#ac99d2', 'Adenoma': '#99a3d2',
        'CRC': '#41c2c4', 'Melanoma': '#8fdbdc', 'T2D': '#f56162', 'MS': '#f58561',
        'OB': '#f5aa61', 'IGT': '#f5cf61', 'BL': '#f5f461', 'Others': '#FFCFD1',
        'AS': '#8FDBF3', 'ACVD': '#BFC1A5', 'CKD': '#A38277', 'COVID-19': '#818181'
    }

    df['project'] = pd.Categorical(df['project'], categories=list(project_colors.keys()), ordered=True)
    df = df.sort_values(by=['project'])

    project_unique = list(project_colors.keys())
    project_color_map = {proj: sns.color_palette(list(project_colors.values()))[i] for i, proj in
                         enumerate(project_unique)}

    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 25, 'font.family': 'Arial', 'axes.titlesize': 22, 'axes.labelsize': 22,
        'xtick.labelsize': 20, 'ytick.labelsize': 20, 'legend.fontsize': 20,
        'figure.titlesize': 24, 'pdf.fonttype': 42, 'ps.fonttype': 42
    })

    fig = plt.figure(figsize=(7, 7), facecolor='white')
    gs = fig.add_gridspec(2, 2, width_ratios=[3, 0.3], height_ratios=[3, 0.3], wspace=0.05, hspace=0.05)

    ax_scatter = fig.add_subplot(gs[0, 0])
    ax_box_pc3 = fig.add_subplot(gs[1, 0], sharex=ax_scatter)
    ax_box_pc4 = fig.add_subplot(gs[0, 1], sharey=ax_scatter)

    for i in range(len(X)):
        color = project_color_map[Y[i]]
        ax_scatter.scatter(X[i, 0], X[i, 1], marker='o', color=color, s=15)

    ax_scatter.set_xlabel('Dim1', fontsize=22)
    ax_scatter.xaxis.set_label_position('top')
    ax_scatter.xaxis.tick_top()
    ax_scatter.xaxis.set_label_coords(0.5, 1.05)

    ax_scatter.set_ylabel('Dim2', fontsize=22)
    ax_scatter.yaxis.set_label_coords(-0.05, 0.5)
    ax_scatter.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                           labelbottom=False, labeltop=False, labelleft=False, labelright=False, labelsize=20)

    sns.boxplot(y='disease', x='PC1', data=df, ax=ax_box_pc3, palette=['#4DBAD6', '#E44A33'], fliersize=1, orient='h',
                width=0.7, saturation=1)
    ax_box_pc3.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False, labeltop=False,
                           labelsize=20)
    ax_box_pc3.set_ylabel('Status', fontsize=22)
    ax_box_pc3.yaxis.set_label_coords(-0.05, 0.5)
    ax_box_pc3.yaxis.set_label_position('left')
    ax_box_pc3.tick_params(axis='y', labelleft=False, left=False, labelright=False, right=False)
    ax_box_pc3.set_xlabel('')

    sns.boxplot(x='disease', y='PC2', data=df, ax=ax_box_pc4, palette=['#4DBAD6', '#E44A33'], fliersize=1, width=0.7,
                saturation=1)
    ax_box_pc4.tick_params(axis='both', which='both', bottom=True, top=False, left=False, right=False, labelleft=False,
                           labelright=False, labelbottom=False, labeltop=False, labelsize=20)
    ax_box_pc4.set_xlabel('Status', fontsize=22)
    ax_box_pc4.xaxis.set_label_position('top')
    ax_box_pc4.xaxis.set_label_coords(0.5, 1.05)
    ax_box_pc4.set_ylabel('')

    add_kruskal_significance(ax_box_pc3, df, 'PC1', 'disease', orientation='h')
    add_kruskal_significance(ax_box_pc4, df, 'PC2', 'disease', orientation='v')

    project_legend = [Line2D([0], [0], marker='o', color='none', markerfacecolor=project_color_map[p],
                             markeredgecolor=project_color_map[p], label=p, markersize=9) for p in project_unique]
    ax_scatter.legend(handles=project_legend, loc='upper left', ncol=1, bbox_to_anchor=(1.05, 1.05), frameon=False,
                      columnspacing=0.13, handletextpad=0.1, labelspacing=0.15, fontsize=22)

    status_legend = [Patch(facecolor='#E44A33', edgecolor='none', label='case'),
                     Patch(facecolor='#4DBAD6', edgecolor='none', label='control')]
    ax_box_pc3.legend(handles=status_legend, loc='lower left', bbox_to_anchor=(0.2, -1.3), frameon=False, ncol=2,
                      handlelength=1.2, handletextpad=0.4, columnspacing=1.0, fontsize=22)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # plt.savefig(save_path, dpi=800, bbox_inches='tight', pad_inches=0.3)
    plt.show()


# ==========================================
# 3. Main Execution Block (分离执行模块)
# ==========================================
if __name__ == "__main__":

    # ---------------------------------------------------------
    # Module A: 5-Fold PanDisease UMAP & Scatter
    # ---------------------------------------------------------
    # PAN_EMB_DIR = '/bgi-seq-model-2/codes/zhangkexin/meta_index/output/llama/v4/PanDisease/Split5.specific.aug.Full.sortabu.ema.12.4/'
    # CLUSTER_OUT_DIR = '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/figures/finetune/11.25/cluster/'
    #
    # # 执行状态开关 (False即可跳过该模块)
    # RUN_MODULE_A = True
    # if RUN_MODULE_A:
    #     print("\n>>> [Module A] Running 5-Fold PanDisease UMAP...")
    #     for i in range(1, 6):
    #         X, y, status = [], [], []
    #         path = f"{PAN_EMB_DIR}split{i}/best_ckpt/result/emb/"
    #         if not os.path.exists(path):
    #             continue
    #
    #         for sample in os.listdir(path):
    #             data = pickle.load(open(path + sample, 'rb'))
    #             X.append(data['finetune_embedding'].tolist())
    #             s = data['label'].replace('metabolic_syndrome', 'MS')
    #             if s.islower(): s = s.capitalize()
    #             y.append(s)
    #             status.append('Control' if data['label'] == 'healthy' else 'Case')
    #
    #         if len(X) > 0:
    #             print(f"  -> Processing UMAP for Split {i}...")
    #             X_umap = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric='euclidean',
    #                                random_state=42).fit_transform(X)
    #             plot_scatter_with_boxplots(X_umap, y, status,
    #                                        f'{CLUSTER_OUT_DIR}metaGPT.pandisease.valtest.split{i}.umap.sortabu.ema.3.24.pdf')

    # ---------------------------------------------------------
    # Module B: Geographic Clustering (America focus)
    # ---------------------------------------------------------
    GEO_SAVE_DIR = '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/figures/pretrain/geo.cluster/'
    META_PATH = '/bgi-seq-model-2/datasets/zhangkexin/meta_index/metaphlan4/fine-tune/curated_LiS_CRC_20204_v0908.train_test.phe'
    PROFILE_PATH = '/bgi-seq-model-2/datasets/zhangkexin/meta_index/metaphlan4/fine-tune/curated_LiS_CRC_20204_v1121.train_test.profile'
    SPLIT_DIR = '/bgi-seq-model-2/codes/zhangkexin/meta_index/output/llama/v4/pretrain/pandisease.test/'

    RUN_MODULE_B = True
    if RUN_MODULE_B and os.path.exists(META_PATH) and os.path.exists(PROFILE_PATH):
        print("\n>>> [Module B] Starting Geographic Clustering...")
        label_df = pd.read_csv(META_PATH, sep='\t', index_col=0)

        for x in os.listdir(SPLIT_DIR):
            if x == 'all.samples': continue

            dir_path = f"{SPLIT_DIR}{x}/"
            if not os.path.isdir(dir_path): continue

            X_emb, continent_labels, country_labels = [], [], []

            for i in os.listdir(dir_path):
                sample_name = i.split('.pkl')[0]
                continent_name = get_continent_from_country(label_df.loc[sample_name, 'country'])

                data = pickle.load(open(dir_path + i, 'rb'))
                X_emb.append(data['sample_embedding'])
                continent_labels.append(continent_name)
                country_labels.append(label_df.loc[sample_name, 'country'])

            if len(X_emb) > 0:
                print(f"  -> Plotting samples for {x}")
                plot_umap(X_emb, continent_labels, f'pandisease.val.test.continent.umap.{x}', f'{GEO_SAVE_DIR}{x}/')
                plot_umap(X_emb, country_labels, f'pandisease.val.test.country.umap.{x}', f'{GEO_SAVE_DIR}{x}/')

    # ---------------------------------------------------------
    # Module C: Pretrain Taxa Embeddings (Top 100 Abundance)
    # ---------------------------------------------------------
    TAXA_EMB_DIR = '/bgi-seq-model-2/codes/zhangkexin/meta_index/output/llama/v4/pretrain/abundance_sim/emb/'

    RUN_MODULE_C = True
    if RUN_MODULE_C and os.path.exists(TAXA_EMB_DIR):
        print("\n>>> [Module C] Processing Taxa Embeddings (Top 100)...")
        X_taxa, y_taxa, abu_taxa = [], [], []

        for file in os.listdir(TAXA_EMB_DIR):
            data = pickle.load(open(os.path.join(TAXA_EMB_DIR, file), 'rb'))
            for i in range(len(data['taxa'])):
                X_taxa.append(data['embedding'][i])
                y_taxa.append(data['taxa'][i])
                abu_taxa.append(data['binning_abundance'][i])

        # 获取出现频率前100的 Taxa
        taxa_freq = pd.DataFrame.from_dict(dict(Counter(y_taxa)), orient='index').sort_values(0, ascending=False)
        top100_taxa = list(taxa_freq.index[:100])

        X_top100, y_top100, abu_top100 = [], [], []
        for i in range(len(y_taxa)):
            if y_taxa[i] in top100_taxa:
                X_top100.append(X_taxa[i])
                y_top100.append(y_taxa[i])
                abu_top100.append(abu_taxa[i])

        X_top100_tsne = np.vstack(X_top100)

        # 1. 绘制带有丰度范围的 t-SNE 图
        print("  -> Plotting t-SNE with Abundance Range...")
        plot_tsne_abu(X_top100_tsne, abu_top100, 'tsne.abu')

        # 2. 绘制前 10 个 Taxa 的高亮 t-SNE 图
        print("  -> Plotting t-SNE with Top 10 Taxa Highlighted...")
        my_targets = list(taxa_freq.index)[:10]
        plot_tsne_highlight(X_top100_tsne, y_top100, target_labels=my_targets, level='taxa_focus')

    # ---------------------------------------------------------
    # Module D: SZ-4D Cohort Clustering
    # ---------------------------------------------------------
    SZ4D_DIR = '/bgi-seq-model-2/codes/zhangkexin/meta_index/output/llama/v4/pretrain/SZ-4D/'
    SZ4D_SAVE_DIR = '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results/pretrain/'

    RUN_MODULE_D = True
    if RUN_MODULE_D and os.path.exists(SZ4D_DIR):
        print("\n>>> [Module D] Processing SZ-4D Cohort Clustering...")
        X_sz, y_sz = [], []

        for file in os.listdir(SZ4D_DIR):
            if file.startswith('SZ-4D'):
                label_name = file.split('_')[2].capitalize()
                path = os.path.join(SZ4D_DIR, file)
                for sample in os.listdir(path):
                    data = pickle.load(open(os.path.join(path, sample), 'rb'))
                    X_sz.append(data['sample_embedding'])
                    y_sz.append(label_name)

        if len(X_sz) > 0:
            print("  -> Plotting UMAP for SZ-4D...")
            plot_umap(X_sz, y_sz, 'SZ-4D.cluster.umap', SZ4D_SAVE_DIR)

    print("\n>>> All tasks completed successfully.")
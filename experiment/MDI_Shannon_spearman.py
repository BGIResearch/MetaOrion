import os
import pickle
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

warnings.filterwarnings('ignore')

# 引入外部 MDI 计算函数
from network_dysbiosis_index import calculate_mdi

# ==========================================
# 1. 基础配置与路径管理 (集中管理，告别硬编码散落)
# ==========================================
DIR_BASE = '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results'
DIR_NET = f'{DIR_BASE}/network/1.8/Antibiotic.intervention.longi.complete/'
DIR_PAN = f'{DIR_BASE}/network/1.8/pandisease/'
DIR_DATA = '/bgi-seq-model-2/datasets/zhangkexin/meta_index/metaphlan4/fine-tune/Antibiotic.intervention.longi'
SAVE_DIR = '/bgi-seq-model-2/codes/xiangwenjing/pic/'


# ==========================================
# 2. 核心数据处理函数
# ==========================================
def filter_profile(X):
    """过滤物种丰度表，仅保留 s__ 级别的特征"""
    index = [i for i, col in enumerate(X.columns) if col.split('|')[-1].startswith('s__')]
    X = X.iloc[:, index]
    X.columns = [col.split('|')[-1] for col in X.columns]
    return X


def calculate_shannon(row):
    """计算单样本 Shannon 指数"""
    row = row[row > 0]
    if len(row) == 0: return 0
    p = row / row.sum()
    return -np.sum(p * np.log(p))


def load_and_prepare_data():
    """统筹数据加载、MDI计算与指标合并，返回可直接绘图的 DataFrame"""
    print(">>> 正在加载丰度表与元数据...")
    # 1. 加载样本列表与丰度信息
    with open(DIR_NET + 'pretrain.emb.samples.list.txt', 'r') as f:
        sample_list = [line.strip() for line in f.readlines()]

    profile = filter_profile(
        pd.read_csv(f'{DIR_DATA}/merged_abundance_table_supp.profile', sep='\t', index_col=0).transpose())

    label_df = pd.read_csv(f'{DIR_DATA}/supp.tsv', sep='\t', index_col=1)
    label_df['sample_id'] = list(label_df.index)

    # 2. 加载网络结构与物种索引
    print(">>> 正在加载网络与 Biomarker...")
    individual_network = np.load(DIR_NET + 'pretrain.emb.individual.npy')
    with open(DIR_NET + 'pretrain.emb.individual.index.pkl', 'rb') as f:
        taxa2id = pickle.load(f)

    # 3. 提取 Core Biomarker (保持原版命名格式匹配)
    top30_core_good = ['s-' + i.lower().replace(" ", "-") for i in
                       pd.read_csv(DIR_PAN + 'top30.negative.o3.csv', index_col=0).index]
    top30_core_bad = ['s-' + i.lower().replace(" ", "-") for i in
                      pd.read_csv(DIR_PAN + 'top30.positive.o3.csv', index_col=0).index]

    pan_biomarker = pd.read_csv(f'{DIR_BASE}/features/11.25/sort.multidisease.mean/pandisease.biomarker.txt', sep='\t',
                                header=None)
    pan_biomarker['weight.abs'] = pan_biomarker[1].abs()
    pan_biomarker = pan_biomarker.sort_values('weight.abs', ascending=False)
    good_biomarker = list(pan_biomarker[pan_biomarker[1] < 0].iloc[:, 0])
    bad_biomarker = list(pan_biomarker[pan_biomarker[1] > 0].iloc[:, 0])

    # 4. 计算指标 (调用外部 calculate_mdi)
    print(">>> 正在计算 MDI 与 Shannon 指数...")
    df_full = calculate_mdi(individual_network, sample_list, taxa2id, sample_list, sample_list, bad_biomarker,
                            good_biomarker)
    df_core = calculate_mdi(individual_network, sample_list, taxa2id, sample_list, sample_list, top30_core_bad,
                            top30_core_good)

    shannon_series = profile.apply(calculate_shannon, axis=1)
    shannon_df = pd.DataFrame(
        {'sample_id': [i.split('.mp4')[0] for i in shannon_series.index], 'Shannon': shannon_series.values})

    # 5. 数据融合
    df_full_sub = df_full[['sample_id', 'MDI']].rename(columns={'MDI': 'Full_MDI'})
    df_core_sub = df_core[['sample_id', 'MDI']].rename(columns={'MDI': 'Core_MDI'})

    plot_df = label_df[['sample_id', 'canon']].copy()
    plot_df = plot_df.merge(shannon_df, on='sample_id', how='inner')
    plot_df = plot_df.merge(df_full_sub, on='sample_id', how='inner')
    plot_df = plot_df.merge(df_core_sub, on='sample_id', how='inner')

    return plot_df


# ==========================================
# 3. 绘图函数 (完全锁定你的原版绘图逻辑)
# ==========================================
def plot_antibiotic_scatter(plot_df, save_path):
    print(">>> 正在绘制散点图...")
    # --- 1. 数据预处理 (原版) ---
    scatter_df = plot_df[['sample_id', 'canon', 'Shannon', 'Core_MDI']].dropna().copy()
    scatter_df['Phase'] = scatter_df['canon'].astype(str).str[0] + ' phase'
    scatter_df = scatter_df[scatter_df['Phase'].isin(['A phase', 'B phase', 'C phase'])]

    x = scatter_df['Shannon']
    y = scatter_df['Core_MDI']

    r, p = stats.spearmanr(x, y)
    slope, intercept, _, _, _ = stats.linregress(x, y)
    line_eq = f"y={slope:.2f}x+{intercept:.2f}"
    phase_colors = {'A phase': '#729ECE', 'B phase': '#FFB579', 'C phase': '#98DF8A'}

    # --- 2. 图像全局设置 (原版) ---
    plt.rcParams['font.family'] = 'Arial'
    plt.rc('font', size=24)
    plt.rcParams['pdf.fonttype'] = 42
    plt.figure(figsize=(12, 8))

    # --- 3. 画散点 (原版) ---
    ax1 = sns.scatterplot(
        data=scatter_df, x='Shannon', y='Core_MDI', hue='Phase',
        palette=phase_colors, hue_order=['A phase', 'B phase', 'C phase'],
        s=120, alpha=0.8, edgecolor=None
    )
    ax1.set_box_aspect(1)

    # --- 4. 画回归线 (原版手动线段) ---
    x_vals = np.array([x.min(), x.max()])
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--', color='#d62728', linewidth=3, label=line_eq)

    # --- 5. 细节美化与标签 (原版) ---
    plt.grid(True, which='major', color='#E0E0E0', linestyle='--', linewidth=1)
    ax1.set_facecolor('white')

    plt.xlabel('Shannon Diversity')
    plt.ylabel('MDI Value (Core)')
    plt.title(f"Shared Biomarkers(Core):MDI vs Shannon\nr={r:.3f}, p={p:.2e}", pad=15)

    ax1.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.10),
        ncol=2,
        frameon=False,
        title='',
        columnspacing=2.0,
        handletextpad=0.4,
        labelspacing=0.5
    )

    plt.subplots_adjust(top=0.88, bottom=0.22)

    # --- 6. 保存与显示 ---
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # plt.savefig(save_path, dpi=800)
    plt.show()
    plt.close()


# ==========================================
# 4. 主执行入口
# ==========================================
if __name__ == '__main__':
    # 统筹所有数据
    final_df = load_and_prepare_data()

    output_pdf = os.path.join(SAVE_DIR, 'Scatter_CoreMNDI_vs_Shannon.pdf')
    plot_antibiotic_scatter(final_df, output_pdf)
    print(">>> 运行完成！")
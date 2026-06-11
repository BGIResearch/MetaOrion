import os
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, matthews_corrcoef, classification_report


def plot_multi_metrics(preds, labels, model_name, dir, classes=None, class_names=None):
    # sns.set()
    plt.style.use('default')
    plt.rc('font', size=16)
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['pdf.fonttype'] = 42

    f, ax = plt.subplots(figsize=(6, 6))

    if classes is None:
        classes = sorted(np.unique(np.concatenate([labels, preds])))
    if class_names is None:
        class_names = [str(c) for c in classes]
    else:
        if len(class_names) != len(classes):
            raise ValueError("class_names的长度必须与类别数量一致")
    # 计算原始混淆矩阵（整数）
    cm = confusion_matrix(labels, preds, labels=classes)
    print("原始混淆矩阵:\n", cm)
    # 按行归一化（计算比例）
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # 处理除零情况（将NaN替换为0）
    cm_normalized = np.nan_to_num(cm_normalized)
    # 使用归一化矩阵绘制热图（颜色反映比例）
    # 注释使用原始混淆矩阵的整数
    # cmap = sns.blend_palette(["#FFFFFF", "#E58579"], as_cmap=True)
    import matplotlib.colors as mcolors

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "custom",
        [
            (0.0, "#FFFFFF"),  # 0 → 白色
            (0.05, "#8AB1D2"),  # 很小值才开始变蓝 ⭐关键
            (0.5, "#D9BDDB"),
            (1.0, "#E58579")
        ]
    )
    sns.heatmap(
        cm_normalized,
        annot=cm,
        fmt='d',
        cmap=cmap,
        norm=mcolors.PowerNorm(gamma=0.8),  # ⭐关键
        cbar=False,
        annot_kws={"fontsize": 10, "color": "#333333"}
    )
    ax.set_xlabel('Predicted Label', fontsize=18)
    ax.set_ylabel('True Label', fontsize=18)
    n_classes = len(classes)
    xticks_pos = np.arange(n_classes) + 0.5
    ax.set_xticks(xticks_pos)
    ax.set_xticklabels(class_names, fontsize=16, rotation=45, ha='right')
    ax.set_yticks(xticks_pos)
    ax.set_yticklabels(class_names, fontsize=16, rotation=0)
    ax.tick_params(axis='both', which='both', direction='in', length=0, width=0)
    # plt.savefig(dir + f"{model_name}.png", dpi=800, bbox_inches='tight')  # 降低dpi加速保存
    plt.savefig(dir + f"{model_name}.pdf", bbox_inches='tight', format='pdf')
    plt.show()

# filtered T2D 117
df = pd.read_csv('/bgi-seq-model-2/datasets/zhangkexin/meta_index/metaphlan4/fine-tune/curated_LiS_CRC_20204_v0908.train_test.phe',sep='\t', index_col=0)
filtered_samples = df[((df['From']=='LiS') & (df['project']=='QinJ_2012'))].index.tolist()
pan_prob_list = []
for i in range(1, 6):
    pan_pred = pd.read_csv(
        f'/bgi-seq-model-2/codes/zhangkexin/meta_index/output/llama/v4/PanDisease/Split5.specific.aug.Full.sortabu.ema.12.4/split{str(i)}/best_ckpt/result/probs/metaGPT.multidisease.test.prob.csv')
    pan_prob_list.append(pan_pred)
pan_pred = pd.concat(pan_prob_list, axis=0)  # axis=0表示纵向堆叠
pan_pred = pan_pred[~pan_pred['Unnamed: 0'].isin(filtered_samples)]

PAN_LABELS = {
    'Healthy': 0,
    'IBD': 1,
    'CRC': 2,
    'T2D': 3,
    'MS': 4,
    'Others': 5,
    'AS': 6,
    'OB': 7,
    'IBS': 8,
    'IGT': 9,
    'BL': 10,
    'ACVD': 11,
    'CKD': 12,
    'COVID-19': 13,
    'Adenoma': 14,
    'Melanoma': 15
}
plot_multi_metrics(pan_pred['pred'], pan_pred['label'], 'metaGPT.sortabu.multidisease.test.split5.confusion.matrix.5.11',
                   '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/figures/finetune/11.25/',
                   classes=np.arange(0, len(PAN_LABELS)),
                   class_names=list(PAN_LABELS.keys()))


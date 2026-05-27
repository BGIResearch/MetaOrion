import os

import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def get_significance_stars(p_value):
        """根据p值返回显著性标记"""
        if p_value < 0.001:
            return '***'
        elif p_value < 0.01:
            return '**'
        elif p_value < 0.05:
            return '*'
        else:
            return ''

def mannwhitneyu_effect_size(group1, group2):
    """计算Mann-Whitney U检验的效应量r"""
    n1, n2 = len(group1), len(group2)
    u = stats.mannwhitneyu(group1, group2).statistic
    r = 1 - (2 * u) / (n1 * n2)
    return r

# 检验泛病的biomarker在case ctrl的富集情况
profile_dir = '/bgi-seq-model-2/datasets/zhangkexin/meta_index/metaphlan4/fine-tune/'
X = pd.read_csv(
    profile_dir+'curated_LiS_CRC_20204_v1121.train_test.profile',
    sep='\t', index_col=0).transpose()
index = []
for i in range(len(X.axes[1])):
    if X.columns[i].split('|')[-1].split('__')[0] == 's':
        index.append(i)
X = X.iloc[:, index]
X.columns = [i.split('|')[-1].replace('__','-').replace(' ','-').replace('_','-').lower() for i in X.columns]

label_df = pd.read_csv(
        profile_dir+'curated_LiS_CRC_20204_v0908.train_test.phe',
        sep='\t', index_col=0)

def case_ctrl_abundance_wilcox(X, case_sample, ctrl_sample, raw_disease, biomarker_list, project=None):
    # if raw_disease == 'pandisease':
    #     healthy_project = []
    #     for i in list(label_df['project'].value_counts().index):
    #         project_df = label_df[label_df['project'] == i]
    #         if len(project_df[project_df['disease_united'] == 'healthy']) == len(project_df):
    #             healthy_project.append(i)
    #     label_exclude_df = label_df[label_df['project'].isin(healthy_project)] # ctrl or general popu
    #     ctrl_sample = label_exclude_df.loc[label_exclude_df['disease'] == 'healthy'].index
    #     case_sample = label_df.loc[label_df['disease'] != 'healthy'].index
    # elif raw_disease == 'Healthy':
    #     ctrl_sample = label_df.loc[label_df['disease'] == 'healthy'].index
    #     case_sample = label_df.loc[label_df['disease'] != 'healthy'].index
    # else:
    #     disease = disease_mapping[raw_disease]
    #     phe_single_project = set(list(phe_df[phe_df['disease_united'] == disease]['project']))
    #     phe_single_df = phe_df[phe_df['project'].isin(phe_single_project)]
    #
    #     ctrl_sample = phe_single_df.loc[phe_single_df['disease'] == 'healthy'].index
    #     case_sample = phe_single_df.loc[phe_single_df['disease'] != 'healthy'].index

    all_data = []
    biomarker_results = ['species', 'case.num', 'case.median', 'case.mean', 'ctrl.num', 'ctrl.median', 'ctrl.mean', 'p.value', 'effect.size']
    biomarker_results = {i: [] for i in biomarker_results}

    p_values_dict = {}

    # 遍历每个biomarker
    for idx, i in enumerate(biomarker_list):
        if i not in X.columns:
            print(f'biomarker {i} not existed.')
            continue
        else:     
            try:
                case_abu = X.loc[case_sample, i].values
                ctrl_abu = X.loc[ctrl_sample, i].values

                if len(case_abu.shape) != 1:  # 有些菌能返回多个结果，所以合并
                    case_abu = np.sum(case_abu, axis=1)
                    ctrl_abu = np.sum(ctrl_abu, axis=1)

                # 去除0值的样本
                # case_abu = case_abu[case_abu != 0]
                # ctrl_abu = ctrl_abu[ctrl_abu != 0]

                if len(case_abu) == 0 or len(ctrl_abu) == 0:
                    print('去除0值后没有样本了', raw_disease, i)

                    biomarker_results['species'].append(i[2:].replace('-', ' ').capitalize())
                    biomarker_results['case.num'].append(len(case_abu))
                    biomarker_results['ctrl.num'].append(len(ctrl_abu))
                    biomarker_results['case.median'].append(np.nan)
                    biomarker_results['ctrl.median'].append(np.nan)
                    biomarker_results['case.mean'].append(np.nan)
                    biomarker_results['ctrl.mean'].append(np.nan)
                    biomarker_results['p.value'].append(np.nan)
                    biomarker_results['effect.size'].append(np.nan)

                    if idx < 10:
                        all_data.append({
                                'biomarker': i[2:].replace('-', ' ').capitalize(),
                                'group': 'case',
                                'Abundance': 0
                            })
                        all_data.append({
                                'biomarker': i[2:].replace('-', ' ').capitalize(),
                                'group': 'control',
                                'Abundance': 0
                            })

                    continue
                else:
                    median_case = np.median(case_abu)
                    median_ctrl = np.median(ctrl_abu)
                    mean_case = np.mean(case_abu)
                    mean_ctrl = np.mean(ctrl_abu)
                    u_statistic, p_value = stats.mannwhitneyu(case_abu, ctrl_abu,
                                                            alternative='two-sided')  # 双侧检验
                    effect_size_r = mannwhitneyu_effect_size(case_abu, ctrl_abu)

                    p_values_dict[i] = p_value

                    if median_case > median_ctrl:
                        enrichment = "疾病组"
                        direction = "高"
                    else:
                        enrichment = "健康组"
                        direction = "低"

                    # print(f"{i},case:{len(case_abu)}, ctrl:{len(ctrl_abu)}, 富集分析：Biomarker 在{enrichment}中显著{direction}表达",
                    #       f"P-value: {p_value:.5f}", f"\n效应量 (r): {abs(effect_size_r):.3f}")
                    biomarker_results['species'].append(i[2:].replace('-', ' ').capitalize())
                    biomarker_results['case.num'].append(len(case_abu))
                    biomarker_results['ctrl.num'].append(len(ctrl_abu))
                    biomarker_results['case.median'].append(median_case)
                    biomarker_results['ctrl.median'].append(median_ctrl)
                    biomarker_results['case.mean'].append(mean_case)
                    biomarker_results['ctrl.mean'].append(mean_ctrl)
                    biomarker_results['p.value'].append(p_value)
                    biomarker_results['effect.size'].append(effect_size_r)

                    # 为当前biomarker准备数据
                    if idx < 10:
                        for value in case_abu:
                            all_data.append({
                                'biomarker': i[2:].replace('-', ' ').capitalize(),
                                'group': 'case',
                                'Abundance': value
                            })

                        for value in ctrl_abu:
                            all_data.append({
                                'biomarker': i[2:].replace('-', ' ').capitalize(),
                                'group': 'control',
                                'Abundance': value
                            })
            except:
                print('未知错误',raw_disease, i)
                continue
    plot_data = pd.DataFrame(all_data)
    biomarker_results = pd.DataFrame(biomarker_results)
    biomarker_results.to_csv(
        f'/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results/features/11.25/sort.biomarker.wilcox.test.4.14.all/{raw_disease}.healthy.wilcox.csv')
    # plt.figure(figsize=(8, 6))
    # plt.style.use('default')
    # plt.rcParams.update({'font.size': 18})
    # plt.rcParams['font.family'] = 'Arial'
    # plt.rcParams['pdf.fonttype'] = 42
    # ax = sns.boxplot(data=plot_data, x='biomarker', y='Abundance', hue='group',
    #                  palette={'case': '#C9352B', 'control': '#339DB5'}, flierprops=dict(marker='.',  # 使用点形状
    #                                                                                     markersize=3,  # 点的大小
    #                                                                                     markerfacecolor='grey',
    #                                                                                     # 点的填充颜色
    #                                                                                     markeredgecolor='grey',
    #                                                                                     # 点的边缘颜色
    #                                                                                     alpha=0.7),
    #                  showfliers=True, width=0.5)
    # # 使用标准对数格式化
    # ax.set_yscale('log')
    # ax.yaxis.set_major_formatter(plt.LogFormatter())  # 标准对数格式
    # x_positions = {}
    #
    # for idx, biomarker in enumerate(biomarker_list[:10]):
    #     x_positions[biomarker] = idx
    # for idx, biomarker in enumerate(biomarker_list[:10]):
    #     if biomarker in p_values_dict.keys():
    #         p_value = p_values_dict[biomarker]
    #         sig_symbol = get_significance_stars(p_value)
    #         x_pos = x_positions[biomarker]
    #         line_y = plot_data['Abundance'].max() * 1.5  # 简单乘法
    #         ax.text(x_pos, line_y, sig_symbol,
    #                 ha='center', va='bottom', fontsize=16, fontweight='bold')
    # # ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    # plt.xticks(rotation=45, ha='right')
    # plt.xlabel('')
    # plt.legend(bbox_to_anchor=(-0.42, 1.2), ncol=1, loc=2, frameon=False, labelspacing=0.5, columnspacing=0.6,
    #            handletextpad=0.6, handlelength=1.6, fontsize=18)
    # plt.title(raw_disease, fontsize=20, pad=10)
    # # plt.savefig(
    # #     f'/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/figures/finetune/11.25/features/pandisease.wilcox.test.12.18/{raw_disease}.biomarker.boxplot.pdf',
    # #     bbox_inches='tight')
    # plt.tight_layout()
    # plt.show()


#
disease_mapping = {
    'Healthy': 'healthy',
    'IBD': 'IBD',
    'CRC': 'CRC',
    'T2D': 'T2D',
    'MS': 'metabolic_syndrome',
    'Others': 'others',
    'AS': 'AS',
    'OB': 'OB',
    'IBS': 'IBS',
    'IGT': 'IGT',
    'BL': 'BL',
    'ACVD': 'ACVD',
    'CKD': 'CKD',
    'COVID-19': 'COVID-19',
    'Adenoma': 'adenoma',
    'Melanoma': 'melanoma'
}
dir = '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results/features/11.25/sort.multidisease.mean/'
for i in os.listdir(dir):
    if i.endswith('.txt'):
        raw_disease = i.split('.biomarker')[0]
        # if raw_disease in ['Melanoma']:
        # if raw_disease in ['T2D']:

        biomarker_list = []
        with open(dir+f'{raw_disease}.biomarker.txt', 'r') as f:
            for line in f.readlines():
                biomarker_list.append(line.strip().split('\t')[0])

        # biomarker_list=['s-anaerobutyricum-hallii']

        if raw_disease == 'pandisease':
            # healthy_project = []
            # for i in list(label_df['project'].value_counts().index):
            #     project_df = label_df[label_df['project'] == i]
            #     if len(project_df[project_df['disease_united'] == 'healthy']) == len(project_df):
            #         healthy_project.append(i)
            # label_exclude_df = label_df[~label_df['project'].isin(healthy_project)]  # ctrl or general popu
            # ctrl_sample = label_exclude_df.loc[label_exclude_df['disease'] == 'healthy'].index
            # case_sample = label_exclude_df.loc[label_exclude_df['disease'] != 'healthy'].index
            ctrl_sample = label_df.loc[label_df['disease'] == 'healthy'].index
            case_sample = label_df.loc[label_df['disease'] != 'healthy'].index
        elif raw_disease == 'Healthy':
            ctrl_sample = label_df.loc[label_df['disease'] == 'healthy'].index
            case_sample = label_df.loc[label_df['disease'] != 'healthy'].index
        else:
            disease = disease_mapping[raw_disease]
            phe_single_project = set(list(label_df[label_df['disease_united'] == disease]['project']))
            # for one_project in phe_single_project:
            #     phe_single_df = label_df[label_df['project']==one_project]
            #     ctrl_sample = phe_single_df.loc[phe_single_df['disease'] == 'healthy'].index
            #     case_sample = phe_single_df.loc[phe_single_df['disease'] != 'healthy'].index
            phe_single_df = label_df[label_df['project'].isin(phe_single_project)]
            ctrl_sample = phe_single_df.loc[phe_single_df['disease'] == 'healthy'].index
            # case_sample = phe_single_df.loc[phe_single_df['disease'] != 'healthy'].index  # sort.biomarker.wilcox.test
            case_sample = phe_single_df.loc[phe_single_df['disease'] == disease].index # sort.biomarker.wilcox.test.tmp     4.14

        # if len(case_sample) == 0 or len(ctrl_sample) == 0:
        #     print(f'{raw_disease} is filtered')
        #     continue
        # else:
        case_ctrl_abundance_wilcox(X, case_sample, ctrl_sample, raw_disease, biomarker_list)



# # 采用五次划分datapath.test的样本，而不是所有的
# split5_samples = []
# for i in range(1, 6):
#     with open(f'/home/share/huadjyin/home/zhangkexin2/data/meta_index/preprocess/metaphlan4/fine-tune/specific.random.5/split{i}/datapath.pandisease.test', 'r') as f:
#         for line in f.readlines():
#             split5_samples.append(line.strip().split('/')[-1].split('.json')[0])
# split5_samples_uniq = list(set(split5_samples))
#
# ctrl_sample = label_df.loc[(label_df.index.isin(split5_samples_uniq)) & (label_df['disease'] == 'healthy')].index
# case_sample = label_df.loc[(label_df.index.isin(split5_samples_uniq)) & (label_df['disease'] != 'healthy')].index
#
# all_data = []
# biomarker_results = ['species', 'case.num', 'case.median', 'ctrl.num', 'ctrl.median', 'p.value', 'effect.size']
# biomarker_results = {i:[] for i in biomarker_results}
# # 遍历每个biomarker
# for i in biomarker_list:
#     case_abu = X.loc[case_sample, i].values
#     ctrl_abu = X.loc[ctrl_sample, i].values
#
#     if len(case_abu.shape) != 1:  # 有些菌能返回多个结果，所以合并
#         case_abu = np.sum(case_abu, axis=1)
#         ctrl_abu = np.sum(ctrl_abu, axis=1)
#
#     # 去除0值的样本
#     case_abu = case_abu[case_abu != 0]
#     ctrl_abu = ctrl_abu[ctrl_abu != 0]
#
#     median_case = np.median(case_abu)
#     median_ctrl = np.median(ctrl_abu)
#     u_statistic, p_value = stats.mannwhitneyu(case_abu, ctrl_abu,
#                                               alternative='two-sided')  # 双侧检验
#     effect_size_r = mannwhitneyu_effect_size(case_abu, ctrl_abu)
#
#     if median_case > median_ctrl:
#         enrichment = "疾病组"
#         direction = "高"
#     else:
#         enrichment = "健康组"
#         direction = "低"
#
#     print(f"{i},case:{len(case_abu)}, ctrl:{len(ctrl_abu)}, 富集分析：Biomarker 在{enrichment}中显著{direction}表达",
#           f"P-value: {p_value:.5f}", f"\n效应量 (r): {abs(effect_size_r):.3f}")
#     biomarker_results['species'].append(i)
#     biomarker_results['case.num'].append(len(case_abu))
#     biomarker_results['ctrl.num'].append(len(ctrl_abu))
#     biomarker_results['case.median'].append(median_case)
#     biomarker_results['ctrl.median'].append(median_ctrl)
#     biomarker_results['p.value'].append(p_value)
#     biomarker_results['effect.size'].append(effect_size_r)
#
#
#     # 为当前biomarker准备数据
#     for value in case_abu:
#         all_data.append({
#             'biomarker': i,
#             'group': 'case',
#             'Abundance': value
#         })
#
#     for value in ctrl_abu:
#         all_data.append({
#             'biomarker': i,
#             'group': 'control',
#             'Abundance': value
#         })
# plot_data = pd.DataFrame(all_data)
# biomarker_results=pd.DataFrame(biomarker_results)
# biomarker_results.to_csv('/home/share/huadjyin/home/zhangkexin2/code/meta_index/experiment/results/features/pandisease.biomarker.test5.wilcox.test.csv')
#
# # 创建大图
# plt.figure(figsize=(15,5))
# plt.style.use('default')
# plt.rcParams.update({'font.size': 20})
# plt.rcParams['font.family'] = 'Arial'
# plt.rcParams['pdf.fonttype'] = 42
#
# # 绘制箱线图
# ax = sns.boxplot(data=plot_data, x='biomarker', y='Abundance', hue='group',
#                  palette={'case': 'firebrick', 'control': 'steelblue'}, showfliers=False,
#                  width=0.7)
#
# # 添加数据点（稍微抖动避免重叠）
# # sns.stripplot(data=plot_data, x='biomarker', y='value', hue='group',
# #               dodge=True, alpha=0.5, size=3, jitter=0.2,
# #               palette={'疾病组': 'firebrick', '健康组': 'steelblue'})
# ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
# plt.xlabel('')
# plt.savefig('/home/share/huadjyin/home/zhangkexin2/code/meta_index/experiment/figures/finetune/features/pandisease.wilcox.test/pandisease.biomarker.test5.boxplot.pdf', bbox_inches='tight')
# plt.show()




# ctrl_sample = label_df.loc[label_df['disease'] == 'healthy'].index
# case_sample = label_df.loc[label_df['disease'] != 'healthy'].index

import os
import json
import random
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split



def stratified_split_IBD_subject(seed, df, label_col='IBD_label', subject_col='subject_id', val_test_ratio=0.5):
    random.seed(seed)
    subjects = df[subject_col].unique()
    subject_labels = df.groupby(subject_col)[label_col].first()  # 每个subject的标签

    global_ratio = Counter(subject_labels)[False] / Counter(subject_labels)[True]
    train_subjects, val_test_subjects = [], []
    case_subjects = subject_labels[subject_labels].index.tolist()
    control_subjects = subject_labels[~subject_labels].index.tolist()

    random.shuffle(case_subjects)
    random.shuffle(control_subjects)
    n_val_test = int(len(subjects) * val_test_ratio)
    n_case = int(n_val_test * (1 / (global_ratio + 1))) #
    n_control = n_val_test - n_case

    val_test_subjects.extend(case_subjects[:n_case])
    val_test_subjects.extend(control_subjects[:n_control])

    train_subjects = [s for s in subjects if s not in val_test_subjects]

    train_idx = df[df[subject_col].isin(train_subjects)].index.tolist()
    val_test_idx = df[df[subject_col].isin(val_test_subjects)].index.tolist()

    return train_idx, val_test_idx

def stratified_split_CRC(seed, phe_CRC_df, val_test_ratio=0.5):
    # 去除腺瘤等ctrl
    phe_CRC_df = phe_CRC_df[(phe_CRC_df['disease_united'] == 'CRC') | (phe_CRC_df['disease_united'] == 'healthy')]

    phe_CRC_df['CRC_label'] = phe_CRC_df['disease_united'] == 'CRC'
    _, _, CRC_train, CRC_val_test = train_test_split(phe_CRC_df, phe_CRC_df['CRC_label'],
                                                     train_size=1-val_test_ratio,
                                                     random_state=seed,
                                                     shuffle=True, stratify=phe_CRC_df['CRC_label'])
    return list(CRC_train.index), list(CRC_val_test.index)

def stratifed_split_IBD(seed, phe_IBD_LiS_df, val_test_ratio=0.5):
    _, _, IBD_LiS_train, IBD_LiS_val_test = train_test_split(phe_IBD_LiS_df, phe_IBD_LiS_df['IBD_label'],
                                                             train_size=1-val_test_ratio,
                                                             random_state=seed,
                                                             shuffle=True, stratify=phe_IBD_LiS_df['IBD_label'])
    return list(IBD_LiS_train.index), list(IBD_LiS_val_test.index)

def stratifed_split_pandisease_rest(seed, phe_filter_CRC_IBD_df, val_test_ratio=0.5):
    phe_filter_CRC_IBD_df = phe_filter_CRC_IBD_df.groupby('disease_united').filter(lambda x: len(x) >= 2)
    phe_single_categories = phe_filter_CRC_IBD_df.groupby('disease_united').filter(lambda x: len(x) == 1)
    _, _, y_train, y_val_test = train_test_split(phe_filter_CRC_IBD_df['group'],
                                                 phe_filter_CRC_IBD_df['disease_united'],
                                                 train_size=1-val_test_ratio,
                                                 random_state=seed,
                                                 shuffle=True, stratify=phe_filter_CRC_IBD_df['disease_united'])
    return list(y_train.index)+list(phe_single_categories.index), list(y_val_test.index)


def sampleid2jsonfile(disease_list, save_dir, samples_list, disease):
    data_files = '/home/share/huadjyin/home/zhangkexin2/data/meta_index/preprocess/metaphlan4/fine-tune/9.9/raw/'
    crc_other_files = '/home/share/huadjyin/home/zhangkexin2/data/meta_index/preprocess/metaphlan4/fine-tune/8.12/PRJNA758208/'
    files_list = []
    for i in samples_list:
        if os.path.exists(data_files + i + '.json'):
            data = json.load(open(data_files + i + '.json', 'r'))
            if data['disease_united'] in disease_list:
                if data['disease_united'] == phe_df.loc[i, 'disease_united'] and data['group'] == phe_df.loc[i, 'group']:
                    files_list.append(data_files + i + '.json')
                else:
                    print('Label Conflict: ', i, data['disease_united'], phe_df.loc[i, 'disease_united'], data['group'], phe_df.loc[i, 'group'])
            else:
                data['disease_united'] = 'others'
                with open(os.path.join(data_files, i+'.json'), 'w') as f:
                    json.dump(data, f, indent=4)
                files_list.append(os.path.join(data_files, i+'.json'))
        elif phe_df.loc[i, 'bioproject'] == 'PRJNA758208':
            # if os.path.exists(crc_other_files + i + '.json'):
            #     files_list.append(crc_other_files + i + '.json')
            # else:
                print('CRC8208 Not Found: ',i, phe_df.loc[i, 'bioproject'])
        else:
            print('Not Found: ', i, phe_df.loc[i, 'bioproject'])

    with open(save_dir + f'datapath.{disease}', 'w') as f:
        for i in files_list:
            f.write(i + '\n')

if __name__ == '__main__':
    fold=1
    for seed in [111, 222, 333, 444, 555]:
        save_dir = f'/home/share/huadjyin/home/zhangkexin2/data/meta_index/preprocess/metaphlan4/fine-tune/specific.random.5/split{str(fold)}/'
        os.makedirs(save_dir+'sample.id/', exist_ok=True)

        # phe_df = pd.read_csv(
        #     '/home/share/huadjyin/home/zhangkexin2/data/meta_index/metaphlan4/fine-tune/curated_LiS_CRC_21271.train_test.change.phe',
        #     sep='\t', index_col=0)
        # phe_df = pd.read_csv(
        #     '/home/share/huadjyin/home/chenjunhong/META_AI/dataset/meta_index/data_v0811/curated_LiS_CRC_20153_v0811.train_test.phe',
        #     sep='\t', index_col=0)
        phe_df = pd.read_csv(
            '/home/share/huadjyin/home/chenjunhong/META_AI/dataset/meta_index/data_v0905/curated_LiS_CRC_20204_v0908.train_test.phe',
            sep='\t', index_col=0)
        class_counts = phe_df['disease_united'].value_counts()
        disease_list = list(class_counts[class_counts >= 120].index)
        phe_df_diseases = phe_df[phe_df['disease_united'].isin(disease_list)]
        phe_df_others = phe_df[~phe_df['disease_united'].isin(disease_list)]
        phe_df_others = phe_df_others.sample(300, random_state=seed)
        phe_df = pd.concat([phe_df_diseases,phe_df_others])

        phe_IBD_project = set(list(phe_df[phe_df['disease_united'] == 'IBD']['project']))
        phe_CRC_project = set(list(phe_df[phe_df['disease_united'] == 'CRC']['project']))
        phe_IBD_df = phe_df[phe_df['project'].isin(phe_IBD_project)]
        phe_CRC_df = phe_df[phe_df['project'].isin(phe_CRC_project)]

        # -------------TRAIN AND VAL_TEST DATASET SPLIT
        # crc split
        CRC_train, CRC_val_test = stratified_split_CRC(seed, phe_CRC_df, 0.2)

        # ibd split
        phe_IBD_df['IBD_label'] = phe_IBD_df['disease_united'] == 'IBD'
        # -shenghui data
        phe_IBD_LiS_df = phe_IBD_df[phe_IBD_df['From'] == 'LiS']
        IBD_LiS_train, IBD_LiS_val_test = stratifed_split_IBD(seed, phe_IBD_LiS_df, 0.2)
        # -curated data
        phe_IBD_curated_df = phe_IBD_df[phe_IBD_df['From'] == 'curated']
        phe_IBD_curated_df = phe_IBD_curated_df[
            (phe_IBD_curated_df['disease_united'] == 'IBD') | (phe_IBD_curated_df['disease_united'] == 'healthy')]
        IBD_curated_train, IBD_curated_val_test = stratified_split_IBD_subject(seed, phe_IBD_curated_df,
                                                                              label_col='IBD_label',
                                                                              subject_col='subject_id', val_test_ratio=0.2)
        IBD_train = IBD_curated_train + IBD_LiS_train
        IBD_val_test = IBD_curated_val_test + IBD_LiS_val_test
        # -add ctrl sample to make IBD case:ctrl=1:1
        phe_ctrl_df = phe_df[
            (phe_df['project'].isin(['LifeLinesDeep_2016', 'MetaCardis_2020_a'])) & (phe_df['disease_united'] == 'healthy')]
        phe_ctrl_df = phe_ctrl_df.sample(frac=1, random_state=seed)
        IBD_train_supp = Counter(phe_IBD_df.loc[IBD_train, 'IBD_label'])[True] - \
                         Counter(phe_IBD_df.loc[IBD_train, 'IBD_label'])[False]
        IBD_test_supp = Counter(phe_IBD_df.loc[IBD_val_test, 'IBD_label'])[True] - \
                        Counter(phe_IBD_df.loc[IBD_val_test, 'IBD_label'])[False]
        IBD_supp_ctrl = list(phe_ctrl_df.sample(n=IBD_train_supp + IBD_test_supp, random_state=seed).index)
        IBD_train = IBD_train + IBD_supp_ctrl[:IBD_train_supp]
        IBD_val_test = IBD_val_test + IBD_supp_ctrl[IBD_train_supp:IBD_train_supp + IBD_test_supp]

        # pandisease split
        phe_filter_CRC_IBD_df = phe_df[
            ~phe_df.index.isin(IBD_train + IBD_val_test + CRC_train + CRC_val_test)]
        pandisease_rest_train, pandisease_rest_val_test = stratifed_split_pandisease_rest(seed, phe_filter_CRC_IBD_df, 0.2)

        Pandisease_train = IBD_train + CRC_train + pandisease_rest_train
        Pandisease_val_test = IBD_val_test + CRC_val_test + pandisease_rest_val_test

        # -------------VAL AND TEST DATASET SPLIT
        # crc split
        phe_CRC_df = phe_df.loc[CRC_val_test, :]
        CRC_test, CRC_val = stratified_split_CRC(seed, phe_CRC_df, 0.5)

        # ibd split
        phe_IBD_df = phe_df.loc[IBD_val_test, :]
        phe_IBD_df['IBD_label'] = phe_IBD_df['disease_united'] == 'IBD'
        # -shenghui data
        phe_IBD_LiS_df = phe_IBD_df[phe_IBD_df['From'] == 'LiS']
        IBD_LiS_test, IBD_LiS_val = stratifed_split_IBD(seed, phe_IBD_LiS_df, 0.5)
        # -curated data
        phe_IBD_curated_df = phe_IBD_df[phe_IBD_df['From'] == 'curated']
        IBD_curated_test, IBD_curated_val = stratified_split_IBD_subject(seed, phe_IBD_curated_df,
                                                                              label_col='IBD_label',
                                                                              subject_col='subject_id', val_test_ratio=0.5)
        IBD_test = IBD_curated_test + IBD_LiS_test
        IBD_val = IBD_curated_val + IBD_LiS_val

        # pandisease split
        phe_filter_CRC_IBD_df = phe_df.loc[pandisease_rest_val_test, :]
        class_counts = phe_filter_CRC_IBD_df['disease_united'].value_counts()
        valid_classes = class_counts[class_counts == 1].index
        small_samples = list(phe_filter_CRC_IBD_df[phe_filter_CRC_IBD_df['disease_united'].isin(valid_classes)].index)
        phe_filter_CRC_IBD_df = phe_filter_CRC_IBD_df[~phe_filter_CRC_IBD_df['disease_united'].isin(valid_classes)]

        phe_filter_CRC_IBD_df = phe_filter_CRC_IBD_df[
            ~phe_filter_CRC_IBD_df.index.isin(IBD_val + IBD_test + CRC_val + CRC_test)]
        pandisease_rest_test, pandisease_rest_val = stratifed_split_pandisease_rest(seed, phe_filter_CRC_IBD_df, 0.5)

        Pandisease_val = IBD_val + CRC_val + pandisease_rest_val + small_samples
        Pandisease_test = IBD_test + CRC_test + pandisease_rest_test

        # SAVE SAMPLE ID and EXTRACT JSON DATAPATH
        for disease in ['Pandisease', 'CRC', 'IBD']:
            for type in ['train', 'val_test', 'val', 'test']:
                var = disease+'_'+type
                # SAVE SAMPLE ID
                sampleid_filename = var.lower().replace('_', '.') + '.sampleid.txt'
                with open(save_dir + 'sample.id/' + sampleid_filename, 'w') as f:
                    data = locals()[var]
                    for i in data:
                        f.write(i + '\n')
                # EXTRACT JSON DATAPATH
                sampleid2jsonfile(disease_list, save_dir, data, var.lower().replace('_', '.'))
                print(var, len(phe_df.loc[data, 'disease_united']), Counter(phe_df.loc[data, 'disease_united']))
        fold+=1






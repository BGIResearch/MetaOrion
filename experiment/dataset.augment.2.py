# 扰动和dropout增强数据
import os
import json

import numpy as np

from pathlib import Path
from functools import partial
from multiprocessing import Pool


def augment_data(data, seed):
    np.random.seed(seed)
    idx = np.arange(0,len(data['taxa']))
    np.random.shuffle(idx)
    perturb_abundance = [data['abundance'][i] for i in idx]
    perturb_taxa= [data['taxa'][i] for i in idx]
    data['taxa'] = perturb_taxa
    data['abundance'] = perturb_abundance
    return data


def process_file(pth, datapath_dir, s, disease):
    pth = pth.strip()
    dirname = os.path.dirname(
        pth.replace('/raw/', f'/aug.11.25.ibd/{s}/')
    )
    filename = os.path.basename(pth)
    data = json.load(open(pth.strip(), 'r'))
    for seed in range(10, 16):  # ibd crc 要换个seed
        if seed == 10:
            aug_data_perturb = data
        else:
            aug_data_perturb = augment_data(data, seed)

            save_pth = os.path.join(dirname, f'perturb.{str(seed)}', filename)
            Path(os.path.dirname(save_pth)).mkdir(parents=True, exist_ok=True)
            with open(save_pth, 'w') as f:
                json.dump(aug_data_perturb, f, indent=4)
            with open(os.path.join(datapath_dir, s, 'datapath.' + disease + '.train.aug'),
                      'a+') as f:
                f.write(save_pth + '\n')

        abu = aug_data_perturb['abundance']
        taxa = aug_data_perturb['taxa']
        for drop_rate in [0.05, 0.08, 0.1, 0.12, 0.15]:
            rng = np.random.RandomState(seed)
            n_species = len(abu)

            # 生成丢弃掩码（1=保留，0=丢弃）
            mask = rng.binomial(1, 1 - drop_rate, size=n_species).astype(bool)

            aug_drop_abu = np.array(abu)[mask]  # 丰度向量（长度减小）
            if len(aug_drop_abu) < 7:
                continue
            else:
                aug_drop_taxa = np.array(taxa)[mask]

                aug_data_drop = aug_data_perturb.copy()
                aug_data_drop['taxa']=aug_drop_taxa.tolist()
                aug_data_drop['abundance']= aug_drop_abu.tolist()

                save_pth = os.path.join(dirname, f'perturb.{str(seed)}.dropout.{str(drop_rate)}', filename)
                Path(os.path.dirname(save_pth)).mkdir(parents=True, exist_ok=True)
                with open(save_pth, 'w') as f:
                    json.dump(aug_data_drop, f, indent=4)
                with open(os.path.join(datapath_dir, s, 'datapath.' + disease + '.train.aug'),
                          'a+') as f:
                    f.write(save_pth + '\n')


datapath_dir = '/home/share/huadjyin/home/zhangkexin2/data/meta_index/preprocess/metaphlan4/fine-tune/nov.specific.random.5/'
for s in os.listdir(datapath_dir):
    for disease in ['ibd']:
        datafiles = open(os.path.join(datapath_dir, s, 'datapath.'+disease+'.train'), 'r').readlines()
        print(len(datafiles))
        with Pool(30) as p:
            process_file_disease = partial(process_file, datapath_dir=datapath_dir, s=s, disease=disease)
            res = p.map(process_file_disease, datafiles)

        datafiles_aug = open(os.path.join(datapath_dir, s, 'datapath.' + disease + '.train.aug'), 'r').readlines()
        all_datafiles = datafiles+datafiles_aug
        with open(os.path.join(datapath_dir, s, 'datapath.' + disease + '.train.all'), 'w') as f:
            for i in all_datafiles:
                f.write(i)
        print('done!')

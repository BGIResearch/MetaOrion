#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project : MetaOrion
# @File    : datasets.py
# @Author  : zhangchao
# @Date    : 2024/7/16 13:36 
# @Email   : zhangchao5@genomics.cn
from __future__ import annotations

import json

import torch
import os.path
import pickle
import numpy as np
from torch.utils.data import Dataset
from typing import List

from src.metaorion.basic.tokenizer import MetaOrionTokenizer


class MetaOrionSequenceDataset(Dataset):
    data_path: List[str] | str
    model_name_or_path: str
    nodes: List

    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)
        self.tokenizer = MetaOrionTokenizer(model_name_or_path=self.model_name_or_path)
        self.PAN_LABELS = {
            'healthy': 0,
            'IBD': 1,
            'CRC': 2,
            'T2D': 3,
            'metabolic_syndrome': 4,
            'others': 5,
            'AS': 6,
            'OB': 7,
            'IBS': 8,
            'IGT': 9,
            'BL': 10,
            'ACVD': 11,
            'CKD': 12,
            'COVID-19': 13,
            'adenoma': 14,
            'melanoma': 15
        }

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        pth = self.data_path[idx]
        _, name = os.path.split(pth)
        raw_data = self.reader(pth=pth, ix=idx, name=name)
        return raw_data

    def reader(self, pth, ix, name):
        data = json.load(open(pth, 'r'))

        zipped = zip(data['abundance'], data['taxa'])
        sorted_pairs = sorted(zipped, reverse=True)
        abundance_sorted, taxa_sorted = zip(*sorted_pairs)
        abundance = np.array(list(abundance_sorted))
        seq = [sq.replace('__', '-').replace('_', '-').replace(' ', '-').lower() for sq in taxa_sorted]


        seq_len = len(seq)
        mask_rate = np.ones(seq_len) * 0.2
        mask_rate[0] = 0
        mask_flag = np.random.binomial(1, mask_rate, seq_len)

        return {
            'seq': seq,
            'abundance': abundance,
            'idx': ix,
            'filename': name,
            'label': data['label'] if 'label' in data.keys() and data['label'] is not None else -1,
            'mask': mask_flag
        }

    def collate_fn(self, batch_sample):
        batch_seq, batch_label = [], []
        batch_filenames, batch_indices = [], []
        batch_abu = []

        batch_masked_seq, batch_mask = [], []
        batch_age, batch_gender = [], []

        for sample in batch_sample:
            batch_seq.append(' '.join(sample['seq']))
            batch_abu.append(sample['abundance'])
            batch_label.append(sample['label'])
            batch_filenames.append(sample['filename'])
            batch_indices.append(sample['idx'])

            batch_mask.append(sample['mask'])

        seq_tokens = self.tokenizer.batch_encode_plus(
            batch_seq, add_special_tokens=True, padding=True, return_tensors='pt')
        length = seq_tokens['input_ids'].size(1)
        abundance, mask = self.batch_padding(
            abundance=batch_abu, mask=batch_mask, length=length)

        batch_label = torch.tensor(batch_label)

        return {
            'input_ids': seq_tokens['input_ids'],
            'attention_mask': mask,
            'padding_mask': seq_tokens['attention_mask'],
            'bin_abundance': abundance,
            'batch_label': batch_label,
            'batch_filenames': batch_filenames,
            'batch_indices': batch_indices,
            'batch_seqs': batch_seq,
            'batch_age': batch_age,
            'batch_gender': batch_gender
        }

    def batch_padding(self, abundance, mask=None, length=1000):
        binning_abundance = [self.binning(x) for x in abundance]
        binning_abundance = torch.tensor(
            np.array([np.pad(x, (0, length - x.shape[0]))
                      if length - x.shape[0] >= 0 else x[:length] for x in binning_abundance]), dtype=torch.float32)
        if mask is not None:
            batch_mask = torch.tensor(
                np.array([np.pad(x, (0, length - x.shape[0])) for x in mask]), dtype=torch.float32)
        else:
            batch_mask = torch.zeros_like(binning_abundance)
        return binning_abundance, batch_mask

    def z_score(self, data: torch.Tensor):
        return (data - data.mean(1)[..., None]) / data.std(1)[..., None]

    def binning(self, x, n_bins=50):
        if x.min() <= 0:
            non_zero_ids = x.nonzero()
            non_zero_x = x[non_zero_ids]
            bins = np.quantile(non_zero_x, np.linspace(0, 1, n_bins - 1))
            nonzero_digits = self._digitize(non_zero_x, bins)
            binned_x = np.zeros_like(x, dtype=np.int64)
            binned_x[non_zero_ids] = nonzero_digits
        else:
            bins = np.quantile(x, np.linspace(0, 1, n_bins - 1))
            binned_x = self._digitize(x, bins)
        return binned_x

    def _digitize(self, x, n_bins):
        left_digits = np.digitize(x, bins=n_bins)
        right_digits = np.digitize(x, bins=n_bins, right=True)
        rands = np.random.rand(len(x))
        digits = rands * (right_digits - left_digits) + left_digits
        digits = np.ceil(digits).astype(np.int64)
        return digits


class MetaOrionSortedSequenceDataset(MetaOrionSequenceDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        data_paths = kwargs.get('data_path')
        # assert isinstance(data_paths, str) and data_paths.endswith('json')
        files = []
        with open(data_paths, 'r') as fp:
            for line in fp.readlines():
                files.append(line.strip())
        self.data_path = files

    def __len__(self):
        return len(self.data_path)


class MetaOrionPhenotypeDataset(MetaOrionSortedSequenceDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def reader(self, pth, ix, name):
        data = json.load(open(pth, 'r'))

        zipped = zip(data['abundance'], data['taxa'])
        sorted_pairs = sorted(zipped, reverse=True)
        abundance_sorted, taxa_sorted = zip(*sorted_pairs)
        abundance = np.array(list(abundance_sorted))
        seq = [sq.replace('__', '-').replace('_', '-').replace(' ', '-').lower() for sq in taxa_sorted]


        if 'gender' in data.keys():
            if isinstance(data['gender'], int):
                gender = data['gender']
            else:
                if data['gender'] == 'male':
                    gender = 1
                elif data['gender'] == 'female':
                    gender = 0
                else:
                    gender = 2
        else:
            gender = 2

        if 'age' in data.keys() and data['age'] is not None:
            age = data['age']
        else:
            age = -1.

        label = data.get('disease_united')
        if label is None:
            label = data.get('label', -1)

        return {
            'seq': seq,
            'abundance': abundance,
            'idx': ix,
            'filename': name,
            'label': label,
            'age': age,
            'gender': gender,
            'project': data['project'] if 'project' in data.keys() else 0,
        }

    def collate_fn(self, batch_sample):

        batch_seq, batch_label = [], []
        batch_filenames, batch_indices = [], []
        batch_abu = []

        batch_age, batch_gender = [], []
        batch_project = []
        batch_state = []

        for sample in batch_sample:
            batch_seq.append(' '.join(sample['seq']))
            batch_abu.append(sample['abundance'])
            if isinstance(sample['label'], str):
                batch_label.append(self.PAN_LABELS[sample['label']])
                if sample['label'] == 'Healthy' or sample['label'] == 'healthy':
                    batch_state.append(0)
                else:
                    batch_state.append(1)
            else:
                batch_label.append(sample['label'])
                batch_state.append(sample['label'])

            batch_filenames.append(sample['filename'])
            batch_indices.append(sample['idx'])

            batch_age.append(sample['age'])
            batch_gender.append(sample['gender'])
            batch_project.append(sample['project'])

        seq_tokens = self.tokenizer.batch_encode_plus(
            batch_seq, add_special_tokens=True, padding=True, return_tensors='pt')
        length = seq_tokens['input_ids'].size(1)
        abundance, mask = self.batch_padding(
            abundance=batch_abu, mask=None, length=length)

        batch_label = torch.tensor(batch_label)
        batch_age = torch.tensor(batch_age, dtype=torch.float32)
        batch_age[torch.isnan(batch_age)] = -1
        batch_gender = torch.tensor(batch_gender)
        batch_state = torch.tensor(batch_state)

        return {
            'input_ids': seq_tokens['input_ids'],
            'attention_mask': mask,
            'padding_mask': seq_tokens['attention_mask'],
            'bin_abundance': abundance,
            'batch_label': batch_label,
            'batch_filenames': batch_filenames,
            'batch_indices': batch_indices,
            'batch_age': batch_age,
            'batch_gender': batch_gender,
            'batch_seq': batch_seq,
            'batch_state': batch_state,
        }

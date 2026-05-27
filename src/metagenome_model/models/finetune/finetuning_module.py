#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project : MetaIndex
# @File    : finetuning_module.py
# @Author  : zhangchao
# @Date    : 2024/12/24 10:24
# @Email   : zhangchao5@genomics.cn
from __future__ import annotations

from dataclasses import dataclass
import os
import torch
import torch.nn as nn
from transformers.utils import ModelOutput


@dataclass
class MetaGenomePEFTModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    logits: torch.FloatTensor = None
    state_logits: torch.FloatTensor = None
    domain_logits: torch.FloatTensor = None
    sample_emb: torch.Tensor = None
    fusion_emb: torch.Tensor = None
    taxa_emb: torch.Tensor = None
    fusion_emb1: torch.Tensor = None
    proj_emb_q: torch.Tensor | None = None
    proj_emb_k: torch.Tensor | None = None
    emb_k_label: torch.Tensor | None = None


@dataclass
class HeaderOutput(ModelOutput):
    logits: torch.FloatTensor = None
    domain_logits: torch.FloatTensor | None = None
    state_logits: torch.FloatTensor | None = None
    pool_emb: torch.Tensor | None = None
    fusion_emb: torch.Tensor | None = None
    emb: torch.Tensor | None = None


class Header(nn.Module):
    """Base class for phenotype prediction heads."""

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, **kwargs):
        raise NotImplementedError

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Load classifier weights from classifier.bin."""
        HEADER_WEIGHT_NAME = 'classifier.bin'
        ckpt = torch.load(os.path.join(pretrained_model_name_or_path, HEADER_WEIGHT_NAME), map_location='cpu')
        model = cls(**kwargs)
        state_dict = model.state_dict()

        matched_dict = {}
        unmatched_ckpt2model = {}
        unmatched_model2ckpt = {}
        for key, val in ckpt.items():
            if key in state_dict.keys():
                matched_dict[key] = val
            else:
                unmatched_ckpt2model[key] = val

        for key, val in state_dict.items():
            if key not in matched_dict.keys():
                unmatched_model2ckpt[key] = val

        state_dict.update(matched_dict)

        model.load_state_dict(state_dict)
        for param in model.parameters():
            param.requires_grad = False
        model.eval()

        if len(unmatched_ckpt2model) != 0:
            print(f'Some of the weights in the checkpoint are not used for model initialization! '
                  f'\n {list(unmatched_ckpt2model.keys())}')
        if len(unmatched_model2ckpt) != 0:
            print(
                f'The weights for some parameters in the model are missing from the checkpoint, they will be randomly initialized! '
                f'\n {list(unmatched_ckpt2model.keys())}')
        return model


class LinearHeader(Header):
    """Use sample, age, and gender embeddings for phenotype prediction."""

    def __init__(self, input_dims, output_dims, dropout_rate):
        super().__init__()
        self.gender_embed = nn.Embedding(3, input_dims)
        self.age_embed = nn.Sequential(
            nn.Linear(1, input_dims),
            nn.ReLU(),
            nn.Linear(input_dims, input_dims),
            nn.BatchNorm1d(input_dims),
            nn.Dropout(p=dropout_rate)
        )

        self.fusion = nn.ModuleList(
            [nn.Linear(input_dims * 3, input_dims),
             nn.BatchNorm1d(input_dims),
             nn.ReLU(),
             nn.Dropout(p=dropout_rate),
             nn.Linear(input_dims, input_dims),
             ]
        )

        self.net = nn.ModuleList([
            nn.Linear(input_dims, input_dims // 8),
            nn.Linear(input_dims // 8, input_dims // 8),
            nn.BatchNorm1d(input_dims // 8),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(input_dims // 8, output_dims),
        ])

        self.state_ly = nn.Sequential(
            nn.Linear(input_dims, input_dims // 8),
            nn.BatchNorm1d(input_dims // 8),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(input_dims // 8, 1)
        )

    def forward(self, sample_embeds, age, gender, abu_features, domain_idx=None, **kwargs):
        """Input: sample features. Output: disease logits and state logits."""
        age_embeds = self.age_embed(age[..., None])
        gender_embeds = self.gender_embed(gender)
        combined_features = torch.cat([sample_embeds, age_embeds, gender_embeds], dim=-1)

        fusion_emb = combined_features
        for ly in self.fusion:
            fusion_emb = ly(fusion_emb)

        head_input = fusion_emb + sample_embeds
        state_logits = self.state_ly(head_input)

        disease_logits = head_input
        for ly in self.net:
            disease_logits = ly(disease_logits)

        return HeaderOutput(
            logits=disease_logits,
            state_logits=state_logits,
            emb=head_input,
            fusion_emb=fusion_emb
        )

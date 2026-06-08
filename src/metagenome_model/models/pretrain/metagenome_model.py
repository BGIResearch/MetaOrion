#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project : metagenome
# @File    : metagenome_model.py
# @Author  : zhangchao
# @Date    : 2024/10/23 11:09 
# @Email   : zhangchao5@genomics.cn
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from src.metagenome_model.config.configuration_model import MetaOrionConfig
from src.metagenome_model.models.pretrain.metagenome_module import (
    MetaOrionBackbone,
    MetaOrionModelOutput,
    MetaOrionPreTrainedModel,
)


class MetaOrionEncoder(MetaOrionPreTrainedModel):
    """Encode taxa tokens and abundance values into token and sample embeddings."""

    def __init__(self, config: MetaOrionConfig, **kwargs):
        super().__init__(config)
        self.embed_abundances = nn.Sequential(
            nn.Linear(1, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            LlamaRMSNorm(config.hidden_size),
            nn.Dropout(p=0.1)
        )

        self.sequence_model = MetaOrionBackbone(config)
        self.fusion = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.attn_pooling = AttentionPooling(config.hidden_size, config.hidden_size)

    def forward(self, input_ids, attention_mask, padding_mask, abundance):
        """Run token embedding, abundance embedding, feature fusion, and sample pooling."""
        tokens_embed, abundance_embed, hidden_states = self._encode_sequence(
            input_ids=input_ids,
            attention_mask=attention_mask,
            padding_mask=padding_mask,
            abundance=abundance
        )
        fusion_emb = self._fuse_token_features(
            hidden_states=hidden_states,
            abundance_embed=abundance_embed
        )
        attn_weight, sample_emb = self.attn_pooling(fusion_emb, padding_mask)

        return MetaOrionModelOutput(
            fusion_emb=fusion_emb,
            sample_emb=sample_emb,
            token_emb=tokens_embed,
            last_hidden_state=hidden_states,
            abundance=abundance,
            attn_weight=attn_weight
        )

    def _encode_sequence(self, input_ids, attention_mask, padding_mask, abundance):
        """Input: tokens and abundance. Output: token, abundance, and sequence embeddings."""
        tokens_embed = self.sequence_model.embed_tokens(input_ids)
        abundance_embed = self.embed_abundances(abundance[..., None])
        hidden_states = self.sequence_model(
            inputs_embeds=tokens_embed + abundance_embed,
            attention_mask=attention_mask,
            padding_mask=padding_mask
        ).last_hidden_state
        return tokens_embed, abundance_embed, hidden_states

    def _fuse_token_features(self, hidden_states, abundance_embed):
        """Input: sequence and abundance embeddings. Output: fused token embeddings."""
        return self.fusion(torch.cat((hidden_states, abundance_embed), -1))


class AttentionPooling(nn.Module):
    """Pool token embeddings into one sample embedding with learned attention weights."""

    def __init__(self, input_dims, hidden_dims):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.Tanh(),
            nn.Linear(hidden_dims, 1)
        )

    def forward(self, x, attn_mask=None):
        """Input: token embeddings and mask. Output: attention weights and pooled embedding."""
        batch_size, seq_len, _ = x.size()
        if attn_mask is None:
            attn_mask = torch.ones((batch_size, seq_len), device=x.device, dtype=torch.float)
        else:
            attn_mask = attn_mask.float()

        token_embeds = x
        attn_logits = self.net(x)
        attn_weight = attn_logits + (1 - attn_mask[..., None]) * torch.finfo(attn_logits.dtype).min
        attn_weight = F.softmax(attn_weight, dim=1)
        pooled_embeds = torch.sum(attn_weight * token_embeds, dim=1)
        return attn_weight.squeeze(-1), pooled_embeds


class MetaOrionAbundanceHead(MetaOrionPreTrainedModel):
    """Predict abundance from token and sample embeddings."""

    keep_model_prefix = True

    def __init__(self, config: MetaOrionConfig, **kwargs):
        super().__init__(config)
        self.model = MetaOrionEncoder(config)

        self.token2abu_header = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size, bias=False),
            nn.LeakyReLU(),
            nn.Linear(config.hidden_size, 1, bias=False)
        )

        self.token2query_header = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size, bias=False),
            nn.Sigmoid(),
            nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        )

    def forward(self, input_ids, attention_mask, padding_mask, abundance):
        """Return token-level and sample-level abundance logits."""
        encoder_output = self.model(input_ids, attention_mask, padding_mask, abundance)

        # Token-level abundance prediction.
        token_logits = self.token2abu_header(encoder_output.fusion_emb).squeeze(2)

        # Sample-level abundance prediction by matching token queries to sample embedding.
        token_queries = self.token2query_header(encoder_output.token_emb)
        sample_logits = torch.bmm(token_queries, encoder_output.sample_emb[..., None]).squeeze(2)

        return MetaOrionModelOutput(
            token_logits=token_logits,
            sample_logits=sample_logits,
            fusion_emb=encoder_output.fusion_emb,
            sample_emb=encoder_output.sample_emb,
            attn_weight=encoder_output.attn_weight
        )



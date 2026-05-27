#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project : metagenome
# @File    : metagenome_model.py
# @Author  : zhangchao
# @Date    : 2024/10/23 11:09 
# @Email   : zhangchao5@genomics.cn
from __future__ import annotations

import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import PeftModel, LoraConfig, get_peft_model
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from src.metagenome_model.config.configuration_model import MetaGenomeConfig
from src.metagenome_model.models.pretrain.gcn_layer import CustomGCNLayer
from src.metagenome_model.models.pretrain.metagenome_module import MetaGenomePreTrainedModel, MetaGenomeModelOutput, MetaGenomeModel


class MeatGenomeForSEQEmbeddingModelWithGraph(MetaGenomePreTrainedModel):
    """Encode taxa tokens and abundance values into token and sample embeddings."""

    def __init__(self, config: MetaGenomeConfig, **kwargs):
        super().__init__(config)
        self.embed_abundances = nn.Sequential(
            nn.Linear(1, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            LlamaRMSNorm(config.hidden_size),
            nn.Dropout(p=0.1)
        )

        self.sequence_model = MetaGenomeModel(config)
        if config.use_graph:
            self.graph_model = CustomGCNLayer(config.hidden_size)
        else:
            self.register_parameter('graph_model', None)

        self.fusion = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.attn_pooling = AttentionPooling(config.hidden_size, config.hidden_size)

    def forward(self, input_ids, attention_mask, padding_mask, abundance, adjacency, sample):
        """Run token embedding, abundance embedding, optional graph fusion, and sample pooling."""
        tokens_embed, abundance_embed, hidden_states = self._encode_sequence(
            input_ids=input_ids,
            attention_mask=attention_mask,
            padding_mask=padding_mask,
            abundance=abundance
        )
        fusion_emb = self._fuse_graph_states(
            hidden_states=hidden_states,
            abundance_embed=abundance_embed,
            adjacency=adjacency
        )
        attn_weight, sample_emb = self.attn_pooling(fusion_emb, padding_mask)

        return MetaGenomeModelOutput(
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

    def _fuse_graph_states(self, hidden_states, abundance_embed, adjacency):
        """Input: sequence states and graph. Output: fused token embeddings."""
        # Fuse sequence states with graph states when graph is enabled.
        if self.graph_model is not None:
            graph_states = self.graph_model(hidden_states, adjacency)
            hidden_states = hidden_states + graph_states
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


class MeatGenomeForSEQEmbeddingModelWithGraphForAbundance(MetaGenomePreTrainedModel):
    """Predict abundance from token and sample embeddings."""

    def __init__(self, config: MetaGenomeConfig, **kwargs):
        super().__init__(config)
        self.model = MeatGenomeForSEQEmbeddingModelWithGraph(config)

        # self.token2ids_header = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

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

    def forward(self, input_ids, attention_mask, padding_mask, abundance, adjacency, sample):
        """Return token-level and sample-level abundance logits."""
        encoder_output = self.model(input_ids, attention_mask, padding_mask, abundance, adjacency, sample)

        # Historical token id head. It is not used in the current flow.
        # ids_logits = self.token2ids_header(encoder_output.last_hidden_state)

        # Token-level abundance prediction.
        token_logits = self.token2abu_header(encoder_output.fusion_emb).squeeze(2)

        # Sample-level abundance prediction by matching token queries to sample embedding.
        token_queries = self.token2query_header(encoder_output.token_emb)
        sample_logits = torch.bmm(token_queries, encoder_output.sample_emb[..., None]).squeeze(2)

        return MetaGenomeModelOutput(
            token_logits=token_logits,
            sample_logits=sample_logits,
            # ids_logits=ids_logits,
            fusion_emb=encoder_output.fusion_emb,
            sample_emb=encoder_output.sample_emb,
            attn_weight=encoder_output.attn_weight
        )


class MetaGenomeCTRModel(nn.Module):
    """Contrastive wrapper for sample embeddings."""

    def __init__(self, model_name_or_path, adapter_name='ctr', is_inference=False):
        super().__init__()
        self.model = MeatGenomeForSEQEmbeddingModelWithGraph.from_pretrained(
            pretrained_model_name_or_path=model_name_or_path
        )
        if is_inference:
            self.model = PeftModel.from_pretrained(
                model=self.model,
                model_id=os.path.join(model_name_or_path, adapter_name),
                device_map='cpu'
            )
            self.register_parameter('proj_q', None)
            self.register_parameter('proj_k', None)
        else:
            lora_config = LoraConfig(
                r=8,
                target_modules=[
                    'q_proj',
                    'k_proj',
                    'v_proj',
                    'o_proj',
                ],
                lora_alpha=8,
                lora_dropout=0.1,
                inference_mode=False
            )
            self.model = get_peft_model(
                model=self.model, peft_config=lora_config, adapter_name=adapter_name
            )

            # self.logit = nn.Linear(self.model.config.hidden_size, 1)

            self.proj_q = nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size // 4, bias=False)
            self.proj_k = nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size // 4, bias=False)
            for param_q, param_k in zip(self.proj_q.parameters(), self.proj_k.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False

    @torch.no_grad()
    def _momentum_update_proj_k(self, alpha=0.5):
        for param_q, param_k in zip(self.proj_q.parameters(), self.proj_k.parameters()):
            param_k.data = param_k.data * alpha + param_q.data * (1.0 - alpha)

    def forward(self, input_ids, attention_mask, padding_mask, abundance, adjacency, moment_alpha=1.):
        feat = self.model(
            input_ids=input_ids, attention_mask=attention_mask, padding_mask=padding_mask,
            abundance=abundance, adjacency=adjacency
        )
        # token_logits = self.logit(feat.fusion_emb)

        proj_emb_k, proj_emb_q = None, None
        if self.training:
            with torch.no_grad():
                self._momentum_update_proj_k(alpha=moment_alpha)
                proj_emb_k = self.proj_k(feat.sample_emb)
            proj_emb_q = self.proj_q(feat.sample_emb)

        return MetaGenomeModelOutput(
            sample_emb=feat.sample_emb,
            proj_emb_q=proj_emb_q,
            proj_emb_k=proj_emb_k,
            # token_logits=token_logits,
        )




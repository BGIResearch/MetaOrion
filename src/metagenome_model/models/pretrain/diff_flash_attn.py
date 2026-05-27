#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project : metagenome
# @File    : diff_flash_attn.py
# @Author  : zhangchao
# @Date    : 2024/12/23 9:45 
# @Email   : zhangchao5@genomics.cn
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Cache
from transformers.models.llama.modeling_llama import LlamaFlashAttention2, LlamaRMSNorm, LlamaRotaryEmbedding, \
    apply_rotary_pos_emb

from src.metagenome_model.models.pretrain.diff_attn import lambda_init_fn


class DiffFlashAttention(LlamaFlashAttention2):
    def __init__(self, config, layer_idx: Optional[int] = None, **kwargs):
        super().__init__(config=config, layer_idx=layer_idx)
        self.config = config
        self.layer_idx = layer_idx
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = False

        self.attn_type = config.attn_type if hasattr(config, 'attn_type') else 'diff_flash_attn'

        if (self.head_dim * self.num_heads) != self.hidden_size and self.head_dim % 2 == 0:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
                f" head_dim must be divisible by 2 (got `head_dim`: {self.head_dim})"
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)

        self.lambda_init = lambda_init_fn(self.layer_idx)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim // 2, dtype=torch.float32).normal_(mean=0., std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim // 2, dtype=torch.float32).normal_(mean=0., std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim // 2, dtype=torch.float32).normal_(mean=0., std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim // 2, dtype=torch.float32).normal_(mean=0., std=0.1))

        self.attn_layernorm = LlamaRMSNorm(hidden_size=self.head_dim)

        self.init_rope()

    def init_rope(self):
        self.rotary_emb = LlamaRotaryEmbedding(
            self.head_dim // 2,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads * 2, self.head_dim // 2).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads * 2, self.head_dim // 2).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        past_key_value = getattr(self, 'past_key_value', past_key_value)
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE modules; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        query_states = query_states.transpose(1, 2).view(bsz, q_len, self.num_heads, 2, self.head_dim // 2)
        key_states = key_states.transpose(1, 2).view(bsz, q_len, self.num_key_value_heads, 2, self.head_dim // 2)
        value_states = value_states.transpose(1, 2).view(bsz, q_len, self.num_key_value_heads, 2, self.head_dim // 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if self.attn_type == 'diff_flash_attn':
                target_dtype = torch.get_autocast_gpu_dtype()
            else:
                target_dtype = self.q_proj.weight.dtype

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        query_states1, query_states2 = query_states[:, :, :, 0], query_states[:, :, :, 1]
        key_states1, key_states2 = key_states[:, :, :, 0], key_states[:, :, :, 1]
        value_states1, value_states2 = value_states[:, :, :, 0], value_states[:, :, :, 1]

        attn11 = self._flash_attention_forward(
            query_states1, key_states1, value_states1, attention_mask,
            query_length=q_len, dropout=dropout_rate, softmax_scale=None
        )
        attn12 = self._flash_attention_forward(
            query_states1, key_states1, value_states2, attention_mask,
            query_length=q_len, dropout=dropout_rate, softmax_scale=None
        )
        attn1 = torch.cat([attn11, attn12], dim=-1)

        attn21 = self._flash_attention_forward(
            query_states2, key_states2, value_states1, attention_mask,
            query_length=q_len, dropout=dropout_rate, softmax_scale=None
        )
        attn22 = self._flash_attention_forward(
            query_states2, key_states2, value_states2, attention_mask,
            query_length=q_len, dropout=dropout_rate, softmax_scale=None
        )
        attn2 = torch.cat([attn21, attn22], dim=-1)

        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(query_states)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(query_states)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        attn_output = attn1 - lambda_full * attn2

        attn_output = self.attn_layernorm(attn_output)
        attn_output = attn_output * (1 - self.lambda_init)
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value

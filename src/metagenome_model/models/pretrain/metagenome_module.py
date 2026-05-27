#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project : MetaIndex 
# @File    : metagenome_module.py
# @Author  : zhangchao
# @Date    : 2024/12/23 13:54 
# @Email   : zhangchao5@genomics.cn
from __future__ import annotations

from typing import Optional, Tuple, Union, List, overload

import os
import torch
import torch.nn as nn
from dataclasses import dataclass

from transformers import PreTrainedModel, StaticCache, DynamicCache, Cache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaFlashAttention2, LlamaMLP, LlamaRMSNorm
from transformers.utils import ModelOutput

from src.metagenome_model.config.configuration_model import MetaGenomeConfig
from src.metagenome_model.models.pretrain.diff_attn import DiffAttention
from src.metagenome_model.models.pretrain.diff_flash_attn import DiffFlashAttention


@dataclass
class MetaGenomeModelOutput(ModelOutput):
    sample_logits: torch.FloatTensor = None
    token_logits: torch.FloatTensor = None
    ids_logits: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None
    proj_emb_q: torch.Tensor | None = None
    proj_emb_k: torch.Tensor | None = None
    fusion_emb: torch.Tensor | None = None
    sample_emb: torch.FloatTensor = None
    token_emb: torch.FloatTensor = None
    abundance: torch.FloatTensor = None
    attn_weight: torch.FloatTensor = None

MetaGenome_ATTN_CLASS = {
    'llama_attn': LlamaAttention,
    'diff_attn': DiffAttention,
    'diff_flash_attn': DiffFlashAttention,
    'flash_attention_2': LlamaFlashAttention2
}


# copied from transformers.modules.modules.modeling_llama.LlamaDecoderLayer
class MetaGenomeLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = MetaGenome_ATTN_CLASS[config.attn_type](config=config, layer_idx=layer_idx)
        self.self_attn.is_causal = config.is_causal
        self.mlp = LlamaMLP(config)
        # self.mlp = MetaGenomeMoE(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs,
    ):
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


# copied from transformers.modules.modules.modeling_llama.LlamaPreTrainedModel
class MetaGenomePreTrainedModel(PreTrainedModel):
    config_class = MetaGenomeConfig
    base_model_prefix = "modules"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MetaGenomeLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _setup_cache(self, cache_cls, max_batch_size, max_cache_len: Optional[int] = None):
        if self.config.attn_type in ["diff_flash_attn", "flash_attention_2"] and cache_cls == StaticCache:
            raise ValueError(
                "`static` cache implementation is not compatible with `attn_implementation==flash_attn` "
                "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
            )

        for layer in self.model.layers:
            device = layer.input_layernorm.weight.device
            if hasattr(self.config, "_pre_quantization_dtype"):
                dtype = self.config._pre_quantization_dtype
            else:
                dtype = layer.self_attn.o_proj.weight.dtype
            layer.self_attn.past_key_value = cache_cls(
                self.config, max_batch_size, max_cache_len, device=device, dtype=dtype
            )

    def _reset_cache(self):
        for layer in self.model.layers:
            layer.self_attn.past_key_value = None

    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
            **kwargs,
    ):
        MODEL_WEIGHT_NAME = 'pytorch_model.bin'
        ckpt = torch.load(os.path.join(pretrained_model_name_or_path, MODEL_WEIGHT_NAME), map_location='cpu')

        config = MetaGenomeConfig.from_pretrained(pretrained_model_name_or_path)
        model = cls(config, **kwargs)
        if model.__class__.__name__ != 'MeatGenomeForSEQEmbeddingModelWithGraphForAbundance':
            # remove `model.` prefix for each key
            ckpt = {key[6:] if key.startswith('model.') else key: val for key, val in ckpt.items()}
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
        model.eval()

        if len(unmatched_ckpt2model) != 0:
            print(f'Some of the weights in the checkpoint are not used for model initialization! '
                  f'\n {list(unmatched_ckpt2model.keys())}')
        if len(unmatched_model2ckpt) != 0:
            print(
                f'The weights for some parameters in the model are missing from the checkpoint, they will be randomly initialized! '
                f'\n {list(unmatched_ckpt2model.keys())}')
        return model


# copied from transformers.modules.modules.modeling_llama.LlamaModel
class MetaGenomeModel(MetaGenomePreTrainedModel):
    def __init__(self, config: MetaGenomeConfig):
        super().__init__(config)
        self.attn_type = config.attn_type
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [MetaGenomeLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            padding_mask: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        past_seen_tokens = 0
        if use_cache:  # kept for BC (cache positions)
            if not isinstance(past_key_values, StaticCache):
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                past_seen_tokens = past_key_values.get_seq_length()

        if cache_position is None:
            if isinstance(past_key_values, StaticCache):
                raise ValueError("cache_position is a required argument when using StaticCache.")
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask=attention_mask, padding_mask=padding_mask, input_tensor=inputs_embeds)

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache() if isinstance(next_decoder_cache, Cache) else next_decoder_cache
            )
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(self, attention_mask, padding_mask, input_tensor):
        dtype, device = input_tensor.dtype, input_tensor.device

        min_type = torch.finfo(dtype).min
        batch_size, sequence_length, _ = input_tensor.size()

        causal_mask = torch.zeros((batch_size, 1, sequence_length, sequence_length), dtype=dtype, device=device)
        causal_mask = causal_mask.eq(0.) * attention_mask[:, None, None, :].eq(1.)
        causal_mask = torch.ones_like(causal_mask) * (
                1. - torch.eye(sequence_length, dtype=dtype, device=device).expand(
            causal_mask.size())) * causal_mask

        causal_mask = (causal_mask + (1 - padding_mask[:, None, None, :])) * min_type
        causal_mask[causal_mask == float('-inf')] = min_type

        return causal_mask

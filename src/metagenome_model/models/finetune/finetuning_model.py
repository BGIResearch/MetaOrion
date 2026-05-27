import os.path

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, PeftModel

from src.metagenome_model.config.configuration_model import MetaGenomeConfig
from src.metagenome_model.models.pretrain.metagenome_model import MeatGenomeForSEQEmbeddingModelWithGraph
from src.metagenome_model.models.finetune.finetuning_module import *
from src.metagenome_model.basic.utils import *


class MetaGenomeForPhenotype(nn.Module):
    def __init__(
            self,
            model_name_or_path,
            dropout_rate,
            split,
            header_type='linear',
            adapter_name='lora_linear',
            is_inference=False,
            **kwargs
    ):
        super().__init__()
        self.model = MeatGenomeForSEQEmbeddingModelWithGraph.from_pretrained(
            pretrained_model_name_or_path=model_name_or_path
            )
        if is_inference:
            # self.model = PeftModel.from_pretrained(
            #     model=self.model,
            #     model_id=os.path.join(model_name_or_path, adapter_name),
            #     device_map='cpu'
            # )
            self.header = LinearHeader.from_pretrained(
                model_name_or_path, input_dims=self.model.config.hidden_size, output_dims=15,
                dropout_rate=dropout_rate
            )
        else:
            # lora微调
            # lora_config = LoraConfig(
            #     r=4,
            #     target_modules=[
            #         'q_proj',
            #         'k_proj',
            #         'v_proj',
            #         'o_proj',
            #     ],
            #     lora_alpha=8,
            #     lora_dropout=0.1,
            #     inference_mode=False,
            #     # modules_to_save=['attn_pooling']
            # )
            #
            # self.model = get_peft_model(
            #     model=self.model, peft_config=lora_config, adapter_name=adapter_name
            # )
            # 微调attn_pooling
            # for name, param in self.model.named_parameters():
            #     if 'attn_pooling' in name:
            #         param.requires_grad = True
            #     else:
            #         param.requires_grad = False
            # 冻结预训练
            # for name, param in self.model.named_parameters():
            #     param.requires_grad = False
            self.header = LinearHeader(input_dims=self.model.config.hidden_size, output_dims=15, dropout_rate=dropout_rate)


    @torch.no_grad()
    def _momentum_update_proj_k(self, alpha=0.5):
        for param_q, param_k in zip(self.proj_q.parameters(), self.proj_k.parameters()):
            param_k.data = param_k.data * alpha + param_q.data * (1.0 - alpha)

    def forward(self, input_ids, attention_mask, padding_mask, abundance, age, gender, domain_idx, sample=None,
                adjacency=None,
                moment_alpha=0.5, **kwargs):
        feat = self.model(
            input_ids=input_ids, attention_mask=attention_mask, padding_mask=padding_mask,
            abundance=abundance, adjacency=adjacency, sample=sample
        )
        out = self.header(sample_embeds=feat.sample_emb, age=age, gender=gender, abu_features=feat.abundance,
                          domain_idx=domain_idx)

        return MetaGenomePEFTModelOutput(
            logits=out.logits,
            state_logits=out.state_logits,
            taxa_emb=feat.fusion_emb,
            domain_logits=out.domain_logits,
            sample_emb=feat.sample_emb,
            fusion_emb1=out.emb,
            fusion_emb=out.fusion_emb
        )

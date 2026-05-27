import torch
import torch.nn as nn

from src.metagenome_model.models.finetune.finetuning_module import LinearHeader, MetaGenomePEFTModelOutput
from src.metagenome_model.models.pretrain.metagenome_model import MeatGenomeForSEQEmbeddingModelWithGraph


class MetaGenomeForPhenotype(nn.Module):
    """Phenotype prediction model built on the pretrained MetaGenome encoder."""

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
            # Historical PEFT load path. It is not used in the current flow.
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
            # Historical LoRA fine-tuning path. It is kept for reference.
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
            # Historical option: fine-tune attention pooling only.
            # for name, param in self.model.named_parameters():
            #     if 'attn_pooling' in name:
            #         param.requires_grad = True
            #     else:
            #         param.requires_grad = False
            # Historical option: freeze pretrained encoder.
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
        """Encode sample features and predict phenotype labels."""
        encoder_output = self.model(
            input_ids=input_ids, attention_mask=attention_mask, padding_mask=padding_mask,
            abundance=abundance, adjacency=adjacency, sample=sample
        )
        header_output = self.header(
            sample_embeds=encoder_output.sample_emb,
            age=age,
            gender=gender,
            abu_features=encoder_output.abundance,
            domain_idx=domain_idx
        )

        return MetaGenomePEFTModelOutput(
            logits=header_output.logits,
            state_logits=header_output.state_logits,
            taxa_emb=encoder_output.fusion_emb,
            domain_logits=header_output.domain_logits,
            sample_emb=encoder_output.sample_emb,
            fusion_emb1=header_output.emb,
            fusion_emb=header_output.fusion_emb
        )

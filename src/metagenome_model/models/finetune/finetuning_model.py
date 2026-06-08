import torch.nn as nn

from src.metagenome_model.models.finetune.finetuning_module import MetaOrionPhenotypeHead, MetaOrionPhenotypeOutput
from src.metagenome_model.models.pretrain.metagenome_model import MetaOrionEncoder


class MetaOrionForPhenotype(nn.Module):
    """Phenotype prediction model built on the pretrained MetaOrion encoder."""

    def __init__(
            self,
            model_name_or_path,
            dropout_rate,
            is_inference=False,
    ):
        super().__init__()
        self.model = MetaOrionEncoder.from_pretrained(
            pretrained_model_name_or_path=model_name_or_path
            )
        if is_inference:
            self.header = MetaOrionPhenotypeHead.from_pretrained(
                model_name_or_path, input_dims=self.model.config.hidden_size, output_dims=15,
                dropout_rate=dropout_rate
            )
        else:
            self.header = MetaOrionPhenotypeHead(
                input_dims=self.model.config.hidden_size, output_dims=15, dropout_rate=dropout_rate
            )

    def forward(self, input_ids, attention_mask, padding_mask, abundance, age, gender):
        """Encode sample features and predict phenotype labels."""
        encoder_output = self.model(
            input_ids=input_ids, attention_mask=attention_mask, padding_mask=padding_mask,
            abundance=abundance
        )
        header_output = self.header(
            sample_embeds=encoder_output.sample_emb,
            age=age,
            gender=gender
        )

        return MetaOrionPhenotypeOutput(
            logits=header_output.logits,
            state_logits=header_output.state_logits,
            taxa_emb=encoder_output.fusion_emb,
            domain_logits=header_output.domain_logits,
            sample_emb=encoder_output.sample_emb,
            fusion_emb=header_output.emb,
        )

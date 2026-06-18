# @Date    : 2026/6/8 16:54
# @Email   : zhangkexin2@genomics.cn
import os
import pickle

import torch
from captum.attr import IntegratedGradients
from captum.attr._models.base import configure_interpretable_embedding_layer, remove_interpretable_embedding_layer
from tqdm import tqdm

from src.metaorion.basic.datasets import MetaOrionPhenotypeDataset
from src.metaorion.basic.kernel import Kernel
from src.metaorion.models.finetune.modeling import MetaOrionForPhenotype


class MetaOrionPhenotypeAttribute(Kernel):
    """Run Integrated Gradients attribution for phenotype prediction."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataloader = self.register_dataloader(
            custom_dataset=MetaOrionPhenotypeDataset,
            is_inference=True
        )
        self.tokenizer = self.dataloader.dataset.tokenizer
        self.model = MetaOrionForPhenotype(
            model_name_or_path=self.model_name_or_path,
            dropout_rate=self.dropout_rate,
            is_inference=True
        )
        self.tax_weight_path = self.register_dir(os.path.join(self.output_home, 'tax_weight_tmp'))

        if self.accelerator.is_main_process:
            self.accelerator.print(self.model)
            self.print_trainable_parameters()

    def forward_func(self, input_ids, attention_mask, padding_mask, bin_abundance, batch_age, batch_gender):
        """Input: token embeddings and sample features. Output: state logits."""
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            padding_mask=padding_mask,
            abundance=bin_abundance,
            age=batch_age,
            gender=batch_gender,
        )
        return output.state_logits.flatten()

    def inference(self, **kwargs):
        self.model, self.dataloader = self.accelerator.prepare(self.model, self.dataloader)
        self.model.eval()
        self.model.zero_grad()

        batch_iters = tqdm(self.dataloader) if self.accelerator.is_main_process else self.dataloader
        interpretable_emb = configure_interpretable_embedding_layer(
            self._prepared_model().model.sequence_model,
            'embed_tokens'
        )
        integrated_gradients = IntegratedGradients(self.forward_func, multiply_by_inputs=True)

        try:
            for sample in batch_iters:
                input_emb = interpretable_emb.indices_to_embeddings(sample['input_ids'])
                baseline = self._build_baseline(interpretable_emb, input_emb)
                attributions, _ = integrated_gradients.attribute(
                    inputs=input_emb,
                    additional_forward_args=(
                        sample['attention_mask'],
                        sample['padding_mask'],
                        sample['bin_abundance'],
                        sample['batch_age'],
                        sample['batch_gender']
                    ),
                    target=None,
                    return_convergence_delta=True,
                    baselines=baseline,
                    n_steps=self.attribution_steps
                )

                if self.accelerator.is_main_process:
                    self._save_batch_attributions(sample=sample, attributions=attributions)
        finally:
            remove_interpretable_embedding_layer(
                self._prepared_model().model.sequence_model,
                interpretable_emb
            )

        self.accelerator.end_training()

    def _prepared_model(self):
        """Input: prepared model. Output: underlying model without DDP wrapper."""
        return self.model.module if hasattr(self.model, 'module') else self.model

    def _build_baseline(self, interpretable_emb, input_emb):
        """Input: input embeddings. Output: baseline token embeddings."""
        baseline_ids = torch.full(
            input_emb.shape[:2],
            fill_value=1,
            dtype=torch.long,
            device=input_emb.device
        )
        return interpretable_emb.indices_to_embeddings(baseline_ids)

    def _save_batch_attributions(self, sample, attributions):
        """Input: batch attribution tensor. Output: one tax weight pkl per sample."""
        token_weights = attributions.sum(dim=-1)
        for sample_idx in range(sample['input_ids'].shape[0]):
            valid_indices = torch.nonzero(sample['padding_mask'][sample_idx], as_tuple=False).flatten()
            if valid_indices.numel() <= 1:
                continue

            token_ids = sample['input_ids'][sample_idx][valid_indices].detach().cpu().tolist()
            taxa = [self._decode_token(token_id) for token_id in token_ids]
            weights = token_weights[sample_idx][valid_indices].detach().cpu().numpy()
            sorted_pairs = sorted(zip(taxa, weights), key=lambda pair: pair[1], reverse=True)
            sorted_data = {
                'taxa': [pair[0] for pair in sorted_pairs],
                'weight': [pair[1] for pair in sorted_pairs]
            }

            sample_name = sample['batch_filenames'][sample_idx].split('.json')[0]
            with open(os.path.join(self.tax_weight_path, sample_name + '.pkl'), 'wb') as fp:
                pickle.dump(sorted_data, fp)

    def _decode_token(self, token_id):
        """Input: token id. Output: readable taxa token."""
        token = self.tokenizer.convert_ids_to_tokens(int(token_id))
        separator = getattr(self.tokenizer, 'word_separator', '')
        if separator and token.startswith(separator):
            return token[len(separator):]
        if token.startswith('\u2581'):
            return token[1:]
        return token

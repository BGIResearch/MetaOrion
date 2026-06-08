# @Date    : 2026/6/8 16:54
# @Email   : zhangkexin2@genomics.cn
import os.path
import torch
import pickle
from tqdm import tqdm
from captum.attr import IntegratedGradients
from captum.attr._models.base import configure_interpretable_embedding_layer
from src.metaorion.basic.utils import *
from src.metaorion.basic.kernel import Kernel
from src.metaorion.basic.datasets import MetaOrionPhenotypeDataset
from src.metaorion.models.finetune.modeling import MetaOrionForPhenotype


class MetaGenomeForPhenotypeInfer(Kernel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataloader = self.register_dataloader(custom_dataset=MetaOrionPhenotypeDataset,
                                                   is_inference=True)
        self.model = MetaOrionForPhenotype(model_name_or_path=self.model_name_or_path, dropout_rate=self.dropout_rate,
                                            is_inference=True)
        self.figs_path = self.register_dir(os.path.join(self.output_home, 'figs'))
        self.probs_path = self.register_dir(os.path.join(self.output_home, 'probs'))
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

        if self.accelerator.is_main_process:
            self.accelerator.print(self.model)
            self.print_trainable_parameters()

    def forward_func(self, input_ids, attention_mask, padding_mask, bin_abundance, batch_age, batch_gender,
                     domain_idx):
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            padding_mask=padding_mask,
            abundance=bin_abundance,
            age=batch_age,
            gender=batch_gender,
            domain_idx=domain_idx,
        )
        return output.state_logits

    @torch.no_grad()
    def inference(self, **kwargs):
        self.model, self.dataloader = self.accelerator.prepare(self.model, self.dataloader)
        self.model.eval()
        self.model.zero_grad()

        batch_iters = tqdm(self.dataloader) if self.accelerator.is_main_process else self.dataloader

        tax_dict = pickle.load(open(
            '/bgi-seq-model-2/codes/zhangkexin/meta_index/output/llama/v4/tokenizer_vocab_dict.pkl',
            'rb'))
        tax_dict = {v: k for k, v in tax_dict.items()}

        interpretable_emb = configure_interpretable_embedding_layer(self.model.module.model.sequence_model,
                                                                    'embed_tokens')

        lig = IntegratedGradients(self.forward_func, multiply_by_inputs=True)

        for idx, sample in enumerate(batch_iters):
            # feature attribute
            input_emb = interpretable_emb.indices_to_embeddings(sample['input_ids'])
            baseline = interpretable_emb.indices_to_embeddings(torch.tensor(1, device='cuda')).expand(
                input_emb.shape[1], -1).unsqueeze(0).expand(input_emb.shape[0], -1, -1)
            attributions, delta = lig.attribute(
                inputs=input_emb,
                additional_forward_args=(
                    sample['attention_mask'], sample['padding_mask'], sample['bin_abundance'],
                    sample['batch_age'], sample['batch_gender'], sample['domain_idx']),
                target=None,
                return_convergence_delta=True,
                baselines=baseline,
                n_steps=100
            )

            # save microbe attribute weights
            if self.accelerator.is_main_process:

                for i in range(sample['input_ids'].shape[0]):
                    if sample['input_ids'].shape[1] == 1:
                        continue
                    name = sample['batch_filenames'][i].split('.json')[0]
                    nopad_indices = torch.nonzero(sample['padding_mask'][i]).squeeze()
                    tax_list = [tax_dict[x].split('▁')[-1] for x in
                                sample['input_ids'][i][nopad_indices].detach().cpu().numpy()]

                    weight_list = attributions.sum(dim=-1).squeeze(0)[nopad_indices].detach().cpu().numpy()
                    data = {'taxa': tax_list, 'weight': weight_list}
                    sorted_pairs = sorted(zip(data['taxa'], data['weight']), key=lambda pair: pair[1], reverse=True)
                    sorted_data = {
                        'taxa': [pair[0] for pair in sorted_pairs],
                        'weight': [pair[1] for pair in sorted_pairs]
                    }

                    if not os.path.exists(os.path.join(self.output_home, 'tax_weight')):
                        os.makedirs(os.path.join(self.output_home, 'tax_weight'))

                    pickle.dump(sorted_data,
                                open(os.path.join(self.output_home, 'tax_weight', name + '.pkl'), 'wb'))

        self.accelerator.end_training()

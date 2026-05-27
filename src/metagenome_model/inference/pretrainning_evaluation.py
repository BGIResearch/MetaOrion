# @Date    : 2026/5/26 16:52 
# @Email   : zhangkexin2@genomics.cn
import os.path
import pickle

import torch
import numpy as np
from tqdm import tqdm

from src.metagenome_model.basic.kernel import Kernel
from src.metagenome_model.models.pretrain.metagenome_model import MeatGenomeForSEQEmbeddingModelWithGraph, \
    MeatGenomeForSEQEmbeddingModelWithGraphForAbundance


class MetaGenomeSEQInference(Kernel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataloader = self.register_dataloader(is_inference=True)
        self.model = MeatGenomeForSEQEmbeddingModelWithGraphForAbundance.from_pretrained(
            # MeatGenomeForSEQEmbeddingModelWithGraph
            pretrained_model_name_or_path=self.model_name_or_path
        )

        if self.accelerator.is_main_process:
            self.accelerator.print(self.model)
            self.print_trainable_parameters()

    @torch.no_grad()
    def inference(self, **kwargs):
        # self.register_wandb()

        self.model, self.dataloader = self.accelerator.prepare(self.model, self.dataloader)
        self.model.eval()
        batch_iters = tqdm(self.dataloader) if self.accelerator.is_main_process else self.dataloader
        s_pad_distance = []
        s_attn_distance = []
        t_pad_distance = []
        t_attn_distance = []
        seq_len_list = []
        for idx, sample in enumerate(batch_iters):

            feat = self.model(
                input_ids=sample['input_ids'],
                attention_mask=sample['attention_mask'],
                padding_mask=sample['padding_mask'],
                # ignore `<pad>` token's abundance
                abundance=sample['bin_abundance'] * (1 - sample['attention_mask']) - sample['attention_mask'],
                adjacency=sample['adjacency'],
                sample=sample
            )

            seq_len = sample['padding_mask'].sum(1)
            s_logits = torch.where(feat.sample_logits < 0, torch.zeros_like(feat.sample_logits), feat.sample_logits)
            for sample_id in range(sample['input_ids'].shape[0]):
                taxa_emb = feat.fusion_emb[sample_id][:seq_len[sample_id].item()].cpu().numpy()
                taxa = sample['batch_seqs'][sample_id].split(' ')
                taxa_weight = feat.attn_weight[sample_id][:seq_len[sample_id].item()].cpu().numpy()
                bin_abu = sample['bin_abundance'][sample_id][:seq_len[sample_id].item()].cpu().numpy()
                predict_abu = s_logits[sample_id][:seq_len[sample_id].item()].cpu().numpy()
                with open(os.path.join(self.output_home, sample['batch_filenames'][sample_id].replace('json', 'pkl')),
                          'wb') as fp:
                    pickle.dump({
                        'taxa': taxa, 'embedding': taxa_emb, 'taxa_weight': taxa_weight,
                        'binning_abundance': bin_abu, 'predict_abundance': predict_abu,
                        'sample_embedding': feat.sample_emb[sample_id].cpu().numpy(),
                        'label': sample['batch_label'].cpu()[sample_id].item(),
                    }, fp)

            # 计算预测丰度与真实值距离
            s_logits = torch.where(feat.sample_logits < 0, torch.zeros_like(feat.sample_logits), feat.sample_logits)
            s_pad_d = self.bray_curtis_distance(s_logits, sample['bin_abundance'], sample['padding_mask'])
            s_attn_d = self.bray_curtis_distance(s_logits, sample['bin_abundance'], sample['attention_mask'])

            # t_logits = torch.where(feat.token_logits < 0, torch.zeros_like(feat.token_logits), feat.token_logits)
            # t_pad_d = self.bray_curtis_distance(t_logits, sample['bin_abundance'], sample['padding_mask'])
            # t_attn_d = self.bray_curtis_distance(t_logits, sample['bin_abundance'], sample['attention_mask'])

            s_pad_distance.append(self.accelerator.gather_for_metrics(s_pad_d).detach().cpu())
            s_attn_distance.append(self.accelerator.gather_for_metrics(s_attn_d).detach().cpu())
            # t_pad_distance.append(self.accelerator.gather_for_metrics(t_pad_d).detach().cpu())
            # t_attn_distance.append(self.accelerator.gather_for_metrics(t_attn_d).detach().cpu())
            seq_len_list.append(self.accelerator.gather_for_metrics(sample['padding_mask'].sum(1)).detach().cpu())

        seq_len = torch.hstack(seq_len_list).numpy()
        s_pad_sim = torch.hstack(s_pad_distance).numpy()
        s_attn_sim = torch.hstack(s_attn_distance).numpy()

        # t_pad_sim = torch.hstack(t_pad_distance).numpy()
        # t_attn_sim = torch.hstack(t_attn_distance).numpy()

        np.save(os.path.join(self.output_home + 'sample.pad.sim.npy'), s_pad_sim)
        np.save(os.path.join(self.output_home + 'sample.attn.sim.npy'), s_attn_sim)
        np.save(os.path.join(self.output_home + 'seq.len.npy'), seq_len)

    def train(self, **kwargs):
        pass

    def bray_curtis_distance(self, x: torch.Tensor, y: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        masked_x = x * attn_mask
        masked_y = y * attn_mask

        valid_lengths = attn_mask.sum(dim=1).float()

        numerator = torch.sum(torch.abs(masked_x - masked_y), dim=1)
        denominator = torch.sum(torch.abs(masked_x) + torch.abs(masked_y), dim=1)

        epsilon = 1e-8
        distance = numerator / (denominator + epsilon)

        all_masked = (valid_lengths == 0)
        distance = torch.where(all_masked, torch.zeros_like(distance), distance)

        return 1 - distance

#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project : MetaIndex
# @File    : finetuning_trainer.py
# @Author  : zhangchao
# @Date    : 2024/12/24 13:26
# @Email   : zhangchao5@genomics.cn
import os
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm


from src.metagenome_model.basic.utils import EMA, GHMC_Loss, get_loss_weights
from src.metagenome_model.basic.kernel import Kernel
from src.metagenome_model.basic.metagenome_dataset import MetaGenomeSortSEQLengthForFinetuneDataset
from src.metagenome_model.models.finetune.finetuning_model import MetaGenomeForPhenotype

BATCH_CHECKPOINT_STEP = 100


class MetaGenomeForPhenotypeTrainer(Kernel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.train_dataloader, self.val_dataloader = self.register_dataloader(
            custom_dataset=MetaGenomeSortSEQLengthForFinetuneDataset)

        self.model = MetaGenomeForPhenotype(
            model_name_or_path=self.model_name_or_path,
            dropout_rate=self.dropout_rate,
        )
        self.register_optimizer(params=self.model.parameters(), weight_decay=self.weight_decay)

        self.batch_ckpt_home = self.register_dir(os.path.join(self.output_home, 'batch_ckpt'))
        self.best_ckpt_home = self.register_dir(os.path.join(self.output_home, 'best_ckpt'))

        if self.accelerator.is_main_process:
            # self.accelerator.print(self.model)
            self.print_trainable_parameters()



    def train(self, **kwargs):
        self.prepare()
        ema = EMA(self.accelerator.unwrap_model(self.model), decay=0.995)
        ghmloss = GHMC_Loss(bins=10, alpha=0.9)

        best_score = [-np.Inf, np.Inf]
        patience = 40
        patience_counter = 0
        multi_weights = torch.tensor(get_loss_weights(), device=self.accelerator.device, dtype=torch.float32)
        stop_training = torch.tensor(0).to(self.accelerator.device)

        for eph in range(self.max_epochs):

            batch_iters = tqdm(self.train_dataloader) if self.accelerator.is_main_process else self.train_dataloader
            epoch_loss = []
            for idx, sample in enumerate(batch_iters):
                self.model.train()

                with self.accelerator.accumulate(self.model):
                    with self.accelerator.autocast():
                        output = self.model(
                            input_ids=sample['input_ids'],
                            attention_mask=sample['attention_mask'],
                            padding_mask=sample['padding_mask'],
                            abundance=sample['bin_abundance'],
                            age=sample['batch_age'],
                            gender=sample['batch_gender'])
                        # pan-disease
                        mask = sample['batch_label'] != 0
                        filtered_logits = output.logits[mask]
                        filtered_labels = sample['batch_label'][mask]

                        cls_loss = F.cross_entropy(filtered_logits, filtered_labels - 1
                                                   , weight=multi_weights)
                        state_ghmloss = ghmloss(output.state_logits, sample['batch_state'].float())
                        loss = state_ghmloss + cls_loss

                        batch_acc1 = (output.state_logits.flatten().sigmoid() >= 0.5).long().eq(
                            sample['batch_state']).sum() / sample['batch_state'].size(0)
                        batch_acc2 = filtered_logits.softmax(-1).argmax(-1).eq(
                            filtered_labels - 1).sum() / filtered_labels.size(0)

                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    ema.update()
                    self.optimizer.zero_grad()

                epoch_loss.append(loss.item())
                self.accelerator.wait_for_everyone()
                if self.accelerator.is_main_process:
                    batch_iters.set_postfix({
                        'EPH': f'{eph + 1:03d}', 'Loss': f'{loss.item():.4f}',
                        'Pan-Acc': f'{batch_acc1.item():.5f}',
                        'Acc': f'{batch_acc2.item():.5f}',
                    })

                    if idx % BATCH_CHECKPOINT_STEP == 0:
                        self.save_ckpt(self.batch_ckpt_home)

            ema.apply_shadow()
            self.model.eval()
            Panacc_list = []
            acc_list = []
            val_preds, val_labels, val_probs = [], [], []

            with torch.no_grad():
                for val_idx, val_sample in enumerate(self.val_dataloader):
                    val_output = self.model(
                        input_ids=val_sample['input_ids'],
                        attention_mask=val_sample['attention_mask'],
                        padding_mask=val_sample['padding_mask'],
                        abundance=val_sample['bin_abundance'],
                        age=val_sample['batch_age'],
                        gender=val_sample['batch_gender'])

                    Panacc_list.append(self.accelerator.gather_for_metrics(
                        (val_output.state_logits.flatten().sigmoid() >= 0.5).long().eq(
                            val_sample['batch_state']).sum() / val_sample['batch_state'].size(0)))
                    acc_list.append(
                        torch.tensor([0 if (val_output.state_logits.flatten().sigmoid() >= 0.5).long()[i] == 0
                                      else val_output.logits.softmax(-1).argmax(-1)[i].item() + 1
                                      for i in range(len(val_output.state_logits))],
                                     device=val_output.state_logits.device, dtype=torch.float32).eq(
                            val_sample['batch_label']).sum() / val_sample['batch_label'].size(0))

                    val_preds.append(self.accelerator.gather_for_metrics(
                        (val_output.state_logits.flatten().sigmoid() >= 0.5).long()))
                    val_labels.append(self.accelerator.gather_for_metrics(val_sample['batch_state']))

                ema.restore()

                self.accelerator.wait_for_everyone()
                if self.accelerator.is_main_process:
                    pan_acc = torch.mean(torch.hstack(Panacc_list)).item()
                    acc = torch.mean(torch.hstack(acc_list)).item()

                    print({'EPH': f'{eph + 1:03d}', 'Val Pan-Acc': f'{pan_acc:.5f}',
                           'Val Acc': f'{acc:.5f}'})


            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                current_acc = pan_acc

                if current_acc > best_score[0]:
                    ema.apply_shadow()
                    print('Saving {}-th epoch checkpoint and validate acc is {}'.format(eph + 1, current_acc))
                    self.save_ckpt(self.best_ckpt_home)
                    ema.restore()
                    best_score[0] = current_acc
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print("  -> Early stopping triggered.")
                    stop_training = torch.tensor(1).to(self.accelerator.device)
                    # torch.distributed.barrier()

            self.accelerator.wait_for_everyone()
            # self.accelerator.broadcast(stop_training, src=0)

            if stop_training.item() == 1:
                break
            else:
                self.manually_update_lr(epoch=eph, decay_gamma=self.decay_gamma, decay_step=self.decay_step)
                # self.scheduler.step()

        self.accelerator.wait_for_everyone()
        self.accelerator.end_training()
        torch.cuda.empty_cache()
        # if torch.distributed.is_initialized():
        #     torch.distributed.destroy_process_group()

    def save_ckpt(self,ckpt_home, **kwargs):
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.model.save_pretrained(save_directory=ckpt_home, safe_serialization=False)
        self.accelerator.save(
            unwrapped_model.header.state_dict(),
            os.path.join(ckpt_home, 'classifier.bin')
        )

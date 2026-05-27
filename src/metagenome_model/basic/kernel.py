#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project : metagenome
# @File    : basic.py
# @Author  : zhangchao
# @Date    : 2024/12/23 9:40 
# @Email   : zhangchao5@genomics.cn
from __future__ import annotations

import os
import torch
import socket
from abc import abstractmethod
from dataclasses import field
from datetime import timedelta, datetime
from pathlib import Path

from accelerate import DistributedDataParallelKwargs, InitProcessGroupKwargs, Accelerator
from torch.utils.data import SequentialSampler, DataLoader
from transformers import set_seed, PreTrainedModel, get_cosine_schedule_with_warmup
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

from src.metagenome_model.basic.metagenome_dataset import MetaGenomeSortSEQLengthDataset


class Kernel:
    # model parameters
    model: torch.nn.Module | PreTrainedModel = field(default=None, metadata={"help": "modules"})
    optimizer: torch.optim.Optimizer = field(default=None, metadata={"help": "optimizer"})
    train_dataloader: torch.utils.data.DataLoader = field(default=None, metadata={"help": "train dataloader"})
    val_dataloader: torch.utils.data.DataLoader = field(default=None, metadata={"help": "validate dataloader"})

    # trainer parameters
    seed: int = field(default=42, metadata={"help": "random seed"})
    train_data_path: str = field(default=None, metadata={'help': 'dataset path'})
    val_data_path: str = field(default=None, metadata={'help': 'validate dataset path'})
    model_name_or_path: str = field(default=None, metadata={'help': 'model name or path'})
    tokenizer_path: str = field(default=None, metadata={'help': 'tokenizer path'})
    split: str = field(default=None, metadata={'help': 'split'})
    mixed_precision: str = field(default='fp16', metadata={
        "help": "mixed precision type. option: ['fp16', 'fp8', 'bf16', 'no']"})
    accumulation_step: int = field(default=1, metadata={"help": "gradient accumulation steps"})
    accelerator: Accelerator = field(default=None, metadata={"help": "accelerator"})
    learning_rate: float = field(default=2e-6, metadata={"help": "learning rate"})
    dropout_rate: float = field(default=0.1, metadata={"help": "dropout rate"})
    weight_decay: float = field(default=1e-4, metadata={"help": "weight decay"})
    batch_size: int = field(default=1, metadata={"help": "batch size"})
    max_epochs: int = field(default=100, metadata={"help": "max epochs"})
    output_home: str = field(default='output', metadata={"help": "output home"})
    decay_gamma: float = field(default=0.995, metadata={'help': 'learning rate decay gamma'})
    decay_step: int = field(default=10, metadata={'help': 'learning rate decay step'})

    # tracker parameters
    tracker_name: str = field(default='wandb', metadata={"help": "logger name"})
    username: str = field(default=None, metadata={"help": "tracker username"})
    projectname: str = field(default=None, metadata={"help": "tracker project name"})
    group: str = field(default=None, metadata={"help": "tracker group"})
    job_type: str = field(default='training', metadata={"help": "tracker job type"})

    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)
        set_seed(self.seed)
        self.configure_ddp()
        self.register_dir(self.output_home)

    def register_dataloader(self, custom_dataset=MetaGenomeSortSEQLengthDataset, *, num_workers=0, drop_last=False,
                            is_inference=False):
        if is_inference:
            val_dataset = custom_dataset(data_path=self.val_data_path, model_name_or_path=self.model_name_or_path,
                                         split=self.split)
            val_dataloader = DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=val_dataset.collate_fn,
                drop_last=drop_last, num_workers=num_workers)
            return val_dataloader
        else:
            train_dataset = custom_dataset(data_path=self.train_data_path, model_name_or_path=self.model_name_or_path,
                                           split=self.split)
            val_dataset = custom_dataset(data_path=self.val_data_path, model_name_or_path=self.model_name_or_path,
                                         split=self.split)
            # sampler = SequentialSampler(dataset)
            # batch_sampler = MetaGenomeSortedBatchSampler(sampler, batch_size=self.batch_size, drop_last=drop_last)
            # dataloader = DataLoader(
            #     dataset, batch_sampler=batch_sampler, collate_fn=dataset.collate_fn, num_workers=num_workers
            # )
            train_dataloader = DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=train_dataset.collate_fn,
                drop_last=drop_last, num_workers=num_workers, persistent_workers=False)
            val_dataloader = DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=val_dataset.collate_fn,
                drop_last=drop_last, num_workers=num_workers, persistent_workers=False)
            return train_dataloader, val_dataloader

    def configure_ddp(self):
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
        process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=1800))
        self.accelerator = Accelerator(
            mixed_precision=self.mixed_precision,
            # log_with=self.tracker_name,
            gradient_accumulation_steps=self.accumulation_step,
            kwargs_handlers=[process_group_kwargs, ddp_kwargs]
        )

    @staticmethod
    def register_dir(dir_name: str):
        Path(dir_name).mkdir(parents=True, exist_ok=True)
        return dir_name

    def register_wandb(self):
        os.environ["WANDB_MODE"] = "offline"
        if self.job_type == 'training':
            config = {
                'init_learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'max_epochs': self.max_epochs,
            }
        else:
            config = None

        self.accelerator.init_trackers(
            project_name=self.projectname,
            config=config,
            init_kwargs={
                'wandb': {
                    'entity': self.username,
                    'notes': socket.gethostname(),
                    'name': f"MetaGenome_{datetime.now().strftime('%d_%m_%Y_%H')}",
                    'group': self.group,
                    'dir': self.output_home,
                    'job_type': self.job_type,
                    'reinit': True,
                }
            }
        )

    def register_optimizer(self, params, weight_decay):
        self.optimizer = torch.optim.AdamW(params, weight_decay=weight_decay, lr=self.learning_rate)

    def manually_update_lr(self, epoch, decay_gamma, decay_step):
        lr = self.learning_rate * (decay_gamma ** (epoch // decay_step))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def warmup_lr(self, train_loader, optimizer):
        steps_per_epoch = len(train_loader)
        total_steps = self.max_epochs * steps_per_epoch

        warmup_steps = int(0.1 * total_steps)
        self.scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

    def prepare(self):
        self.model = self.accelerator.prepare_model(self.model)
        self.optimizer = self.accelerator.prepare_optimizer(self.optimizer)
        self.train_dataloader = self.accelerator.prepare_data_loader(self.train_dataloader)
        self.val_dataloader = self.accelerator.prepare_data_loader(self.val_dataloader)

    @abstractmethod
    def train(self, **kwargs):
        """Training loop"""

    def save_ckpt(self, ckpt_home, **kwargs):
        """Save checkpoint"""
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(save_directory=ckpt_home, safe_serialization=False)
        self.accelerator.save(self.optimizer.state_dict(), os.path.join(ckpt_home, 'optimizer.bin'))

    def print_trainable_parameters(self, model=None):
        total = 0
        trainable = 0
        if model is None:
            parameter_dict = self.model.named_parameters()
        else:
            parameter_dict = model.named_parameters()
        for k, v in parameter_dict:
            total += v.numel()
            if v.requires_grad:
                trainable += v.numel()
        self.accelerator.print(
            f'Trainable Params:|| Trainable: {trainable} || All: {total} || Rate: {(trainable / total) * 100:.2f}%'
        )

    def reload_weights(self, model_name_or_path):
        ckpt_name = 'pytorch_model.bin'
        ckpt = torch.load(os.path.join(model_name_or_path, ckpt_name), map_location='cpu')
        state_dict = self.model.state_dict()

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
        self.model.load_state_dict(state_dict)

        for param in self.model.parameters():
            param.requires_grad = True

        if len(unmatched_ckpt2model) != 0:
            print(f'Some of the weights in the checkpoint are not used for model initialization! '
                  f'\n {list(unmatched_ckpt2model.keys())}')
        if len(unmatched_model2ckpt) != 0:
            print(
                f'The weights for some parameters in the model are missing from the checkpoint, they will be randomly initialized! '
                f'\n {list(unmatched_ckpt2model.keys())}')


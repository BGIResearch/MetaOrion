#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project : MetaOrion
# @File    : basic.py
# @Author  : zhangchao
# @Date    : 2024/12/23 9:40 
# @Email   : zhangchao5@genomics.cn
from __future__ import annotations

import os
import torch
from abc import abstractmethod
from dataclasses import field
from datetime import timedelta
from pathlib import Path

from accelerate import DistributedDataParallelKwargs, InitProcessGroupKwargs, Accelerator
from torch.utils.data import DataLoader
from transformers import set_seed, PreTrainedModel

from src.metaorion.basic.datasets import MetaOrionSortedSequenceDataset


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

    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)
        set_seed(self.seed)
        self.configure_ddp()
        self.register_dir(self.output_home)

    def register_dataloader(self, custom_dataset=MetaOrionSortedSequenceDataset, *, num_workers=0, drop_last=False,
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
            gradient_accumulation_steps=self.accumulation_step,
            kwargs_handlers=[process_group_kwargs, ddp_kwargs]
        )

    @staticmethod
    def register_dir(dir_name: str):
        Path(dir_name).mkdir(parents=True, exist_ok=True)
        return dir_name

    def register_optimizer(self, params, weight_decay):
        self.optimizer = torch.optim.AdamW(params, weight_decay=weight_decay, lr=self.learning_rate)

    def manually_update_lr(self, epoch, decay_gamma, decay_step):
        lr = self.learning_rate * (decay_gamma ** (epoch // decay_step))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def prepare(self):
        self.model = self.accelerator.prepare_model(self.model)
        self.optimizer = self.accelerator.prepare_optimizer(self.optimizer)
        self.train_dataloader = self.accelerator.prepare_data_loader(self.train_dataloader)
        self.val_dataloader = self.accelerator.prepare_data_loader(self.val_dataloader)

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


import torch
import time
import torch.nn as nn
import sys
import logging as lg
import numpy as np
import math
import torchvision

from torch.utils.data import DataLoader
from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomGrayscale

from src.learners.ce import CELearner
from src.utils import name_match
from src.models.resnet import SupConResNet

LR_MIN = 5e-4
LR_MAX = 5e-2

class GDUMBLearner(CELearner):
    def __init__(self, args):
        self.mem_epochs = 30
        self.mem_training_batch_size = 16
        
        super().__init__(args)
        self.buffer = name_match.buffers[self.params.buffer](
                max_size=self.params.mem_size,
                img_size=self.params.img_size,
                nb_ch=self.params.nb_channels,
                n_classes=self.params.n_classes,
                drop_method=self.params.drop_method
            )
        self.init_model()

    def init_model(self):
        self.model = SupConResNet(
            head='mlp',
            dim_in=self.params.dim_in,
            proj_dim=self.params.n_classes,
            input_channels=self.params.nb_channels,
            nf=self.params.nf
        ).to(self.device)
        self.optim = self.load_optim()
    
    def load_optim(self):
        optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=LR_MAX,
                momentum=0.9,
                weight_decay=1e-6
                )
        
        return optimizer

    def load_criterion(self):
        return nn.CrossEntropyLoss() 

    def train(self, dataloader, **kwargs):
        self.model = self.model.train()

        for j, batch in enumerate(dataloader):
            # Stream data
            batch_x, batch_y = batch[0], batch[1]
            self.stream_idx += len(batch_x)
            
            # Update buffer
            self.buffer.update(imgs=batch_x, labels=batch_y)
    
    def train_on_memory(self, **kwargs):
        if not self.params.custom_gdumb:
            self.init_model()
        self.model.train()
        mem_x, mem_y = self.buffer.get_all()

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optim, T_0=1, T_mult=2, eta_min=LR_MIN)

        for e in range(self.mem_epochs):
            if e <= 0: # Warm start of 1 epoch
                for param_group in self.optim.param_groups:
                    param_group['lr'] = LR_MAX * 0.1
            elif e == 1: # Then set to maxlr
                for param_group in self.optim.param_groups:
                    param_group['lr'] = LR_MAX

            else: # Aand go!
                scheduler.step()

            idx = np.random.permutation(len(mem_x)).tolist()
            mem_x = mem_x[idx]
            mem_y = mem_y[idx]
            self.model.train()
            batch_size = self.mem_training_batch_size
            
            for j in range(len(mem_y) // batch_size):
                _, logits = self.model(self.transform_train(mem_x[batch_size * j:batch_size * (j + 1)].to(self.device)))
                loss = self.criterion(logits, mem_y[batch_size * j:batch_size * (j + 1)].to(self.device))
                self.loss = loss.item()

                # Optim
                self.optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                self.optim.step()
            
    def evaluate(self, dataloaders, task_id, **kwargs):
        self.train_on_memory()
        return super().evaluate(dataloaders, task_id)

    def evaluate_offline(self, dataloaders, epoch):
        self.train_on_memory()
        return super().evaluate_offline(dataloaders, epoch)
import torch
import time
import torch.nn as nn
import sys
import logging as lg
import random as r
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torchvision
import torch.cuda.amp as amp
import random

from sklearn.metrics import accuracy_score
from copy import deepcopy

from src.learners.base import BaseLearner
from src.utils.losses import vMFLoss, AGDLoss
from src.utils import name_match
from src.utils.utils import get_device

device = get_device()

scaler = amp.GradScaler()

class FDLearner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.buffer = name_match.buffers[self.params.buffer](
                max_size=self.params.mem_size,
                img_size=self.params.img_size,
                nb_ch=self.params.nb_channels,
                n_classes=self.params.n_classes,
                drop_method=self.params.drop_method,
            )

        self.tf_seq = {}
        for i in range(self.params.n_augs):
            self.tf_seq[f"aug{i}"] = self.transform_train.to(device)

    def load_criterion(self):
        if self.params.fd_loss == 'vmf':
            return vMFLoss(
                var=self.params.var,
                mu=self.params.mu,
                proj_dim=self.params.proj_dim,
            )
        elif self.params.fd_loss == 'agd':
            return AGDLoss(
                var=self.params.var,
                mu=self.params.mu,
                proj_dim=self.params.proj_dim
            )

    def train(self, dataloader, **kwargs):
        self.model = self.model.train()
        if self.params.training_type == 'inc':
            self.train_inc(dataloader, **kwargs)
        elif self.params.training_type == "uni":
            self.train_uni(dataloader, **kwargs)
        elif self.params.training_type == "blurry":
            self.train_blurry(dataloader, **kwargs)

    # @profile
    def train_inc(self, dataloader, **kwargs):
        self.model.train()
        task_name = kwargs.get("task_name")
    
        for j, batch in enumerate(dataloader):
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                # Stream batch
                batch_x, batch_y = batch[0], batch[1]
                self.stream_idx += len(batch_x)
                
                for _ in range(self.params.mem_iters):
                    # Iteration over memory + stream
                    mem_x, mem_y = self.buffer.random_retrieve(n_imgs=self.params.mem_batch_size)
                    
                    if mem_x.size(0) > 0:
                        combined_x = torch.cat([mem_x, batch_x]).to(device)
                        combined_y = torch.cat([mem_y, batch_y]).to(device)

                        # Augment
                        augmentations = self.augment(combined_x=combined_x, mem_x=mem_x.to(device), batch_x=batch_x.to(device))
                        
                        # Inference
                        proj_list = []
                        for aug in augmentations:
                            _, p = self.model(aug, y=combined_y, proj_norm=True)
                            proj_list.append(p.unsqueeze(1))

                        projections = torch.cat(proj_list, dim=1)
                        
                        loss = self.criterion(
                            features=projections,
                            labels=combined_y
                            )

                        # Loss
                        loss = loss.mean()

                        # Backprop
                        self.loss = loss.item()
                        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=False):
                            scaler.scale(loss).backward()
                            scaler.step(self.optim)
                            scaler.update()
                        self.optim.zero_grad()
                        print(f"Loss {self.loss:.3f} batch {j}", end="\r")

                self.buffer.update(imgs=batch_x, labels=batch_y)
                if (j == (len(dataloader) - 1)) and (j > 0):
                    print(
                        f"Phase : {task_name}   batch {j}/{len(dataloader)}   Loss : {self.loss:.4f}    time : {time.time() - self.start:.4f}s",
                        end="\r"
                    )
                    self.save(model_name=f"ckpt_{task_name}.pth")
    
    def train_blurry(self, dataloader, **kwargs):
        self.model.train()
        for j, batch in enumerate(dataloader):
            # bincounts.append(batch[1].long().bincount(minlength=self.params.n_classes).cpu().tolist())
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                 # Stream batch
                batch_x, batch_y = batch[0], batch[1]
                self.stream_idx += len(batch_x)
                for _ in range(self.params.mem_iters):
                    # Iteration over memory + stream
                    mem_x, mem_y = self.buffer.random_retrieve(n_imgs=self.params.mem_batch_size)
                    if mem_x.size(0) > 0:
                        combined_x = torch.cat([mem_x, batch_x]).to(device)
                        combined_y = torch.cat([mem_y, batch_y]).to(device)

                        # Augment
                        augmentations = self.augment(combined_x=combined_x, mem_x=mem_x.to(device), batch_x=batch_x.to(device))
                        
                        # Inference
                        proj_list = []
                        for aug in augmentations:
                            _, p = self.model(aug, proj_norm=True)
                            proj_list.append(p.unsqueeze(1))

                        projections = torch.cat(proj_list, dim=1)
                        
                        loss = self.criterion(
                            features=projections,
                            labels=combined_y
                            )

                        # Loss
                        loss = loss.mean()

                        # Backprop
                        self.loss = loss.item()
                        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=False):
                            scaler.scale(loss).backward()
                            scaler.step(self.optim)
                            scaler.update()
                        self.optim.zero_grad()
                        print(f"Loss {self.loss:.3f}  batch {j}", end="\r")
                self.buffer.update(imgs=batch_x, labels=batch_y)

    def train_uni(self, dataloader, **kwargs):
        raise NotImplementedError
            
    def plot(self):
        self.writer.add_scalar("loss", self.loss, self.stream_idx)
        self.writer.add_scalar("n_added_so_far", self.buffer.n_added_so_far, self.stream_idx)
        percs = self.buffer.get_labels_distribution()
        for i in range(self.params.n_classes):
            self.writer.add_scalar(f"labels_distribution/{i}", percs[i], self.stream_idx)
    
    def augment(self, combined_x, mem_x, batch_x, **kwargs):
        with torch.no_grad():
            augmentations = []
            for key in self.tf_seq:
                augmentations.append(self.tf_seq[key](combined_x))
                
            augmentations.append(combined_x)
            return augmentations
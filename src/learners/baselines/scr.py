import torch
import time
import torch.nn as nn
import sys
import logging as lg
import pandas as pd
import numpy as np

from copy import deepcopy

from torch.utils.data import DataLoader

from src.learners.base import BaseLearner
from src.utils.losses import SupConLoss
from src.buffers.reservoir import Reservoir
from src.utils import name_match
from src.utils.utils import filter_labels
from src.utils.utils import get_device

device = get_device()

class SCRLearner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        # In SCR they use the images from memory for evaluation
        self.params.eval_mem = True
        self.params.supervised = True
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
        return SupConLoss(self.params.temperature)
    
    def train(self, dataloader, **kwargs):
        self.model = self.model.train()
        if self.params.training_type == 'inc' or self.params.training_type == "blurry":
            self.train_inc(dataloader, **kwargs)
        elif self.params.training_type == "uni":
            self.train_uni(dataloader, **kwargs)

    def train_uni(self, dataloader, **kwargs):
        for j, batch in enumerate(dataloader):
            # Stream batch
            batch_x, batch_y = batch[0].to(self.device), batch[1].to(self.device)
            self.stream_idx += len(batch_x)
            
            aug_1 = self.transform_train(batch_x)
            aug_2 = self.transform_train(batch_x)
            
            augs = [aug_1, aug_2]

            self.model.train()
            projs = []
            for a in augs:
                _, p = self.model(a, proj_norm=True)
                projs.append(p.unsqueeze(1))

            projections = torch.cat(projs, dim=1)
            
            # Loss
            loss = self.criterion(
                features=projections,
                labels=batch_y
                )
            loss = loss.mean()

            # Backprop
            self.loss = loss.item()
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            print(f"Loss {self.loss:.3f}  batch {j}", end="\r")

    def train_inc(self, dataloader, **kwargs):
        self.model = self.model.train()
        task_name = kwargs.get('task_name', None)
        task_id = kwargs.get('task_id', None)
        dataloaders = kwargs.get("dataloaders")

        for j, batch in enumerate(dataloader):
            # Stream data
            batch_x, batch_y = batch[0], batch[1]
            self.stream_idx += len(batch_x)

            for _ in range(self.params.mem_iters):
                mem_x, mem_y = self.buffer.random_retrieve(n_imgs=self.params.mem_batch_size)

                if mem_x.size(0) > 0:
                    # Combined batch
                    combined_x, combined_y = self.combine(batch_x, batch_y, mem_x, mem_y)  # (batch_size, nb_channel, img_size, img_size)

                    # Augment
                    augmentations = self.augment(combined_x=combined_x, mem_x=mem_x.to(device), batch_x=batch_x.to(device))

                    self.model.train()
                    projs = []
                    for a in augmentations:
                        _, p = self.model(a, proj_norm=True)
                        projs.append(p.unsqueeze(1))

                    projections = torch.cat(projs, dim=1)

                    # Loss
                    loss = self.criterion(features=projections, labels=combined_y if self.params.supervised else None)
                    loss = loss.mean()

                    self.loss = loss.item()
                    print(f"Loss {self.loss:.3f}  batch {j}", end="\r")
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()
                    
            # Update buffer
            self.buffer.update(imgs=batch_x, labels=batch_y)

            # Plot to tensorboard
            if self.params.tensorboard:
                self.plot()
            if (j == (len(dataloader) - 1)) and (j > 0):
                lg.info(
                    f"Phase : {task_name}   batch {j}/{len(dataloader)}   Loss : {self.loss:.4f}    time : {time.time() - self.start:.4f}s"
                )
    
    def plot(self):
        self.writer.add_scalar("loss", self.loss, self.stream_idx)
        self.writer.add_scalar("n_added_so_far", self.buffer.n_added_so_far, self.stream_idx)
        percs = self.buffer.get_labels_distribution()
        for i in range(self.params.n_classes):
            self.writer.add_scalar(f"labels_distribution/{i}", percs[i], self.stream_idx)

    def combine(self, batch_x, batch_y, mem_x, mem_y):
        mem_x, mem_y = mem_x.to(self.device), mem_y.to(self.device)
        batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
        combined_x = torch.cat([mem_x, batch_x])
        combined_y = torch.cat([mem_y, batch_y])
        return combined_x, combined_y

    def augment(self, combined_x, mem_x, batch_x, **kwargs):
        with torch.no_grad():
            augmentations = []
            for key in self.tf_seq:
                augmentations.append(self.tf_seq[key](combined_x))
            augmentations.append(combined_x)
            return augmentations
    
    def encode(self, dataloader, use_proj=False, nbatches=-1):
        """Compute representations - labels pairs for a certain number of batches of dataloader.
            Not really optimized.
        Args:
            dataloader (torch dataloader): dataloader to encode
            nbatches (int, optional): Number of batches to encode. Use -1 for all. Defaults to -1.
        Returns:
            representations - labels pairs
        """
        i = 0
        with torch.no_grad():
            for sample in dataloader:
                if nbatches != -1 and i >= nbatches:
                    break
                inputs = sample[0]
                labels = sample[1]
                
                inputs = inputs.to(self.device)
                if use_proj:
                    _, features = self.model(self.transform_test(inputs))
                else:
                    features, _ = self.model(self.transform_test(inputs))
                if i == 0:
                    all_labels = labels.cpu().numpy()
                    all_feat = features.cpu().numpy()
                else:
                    all_labels = np.hstack([all_labels, labels.cpu().numpy()])
                    all_feat = np.vstack([all_feat, features.cpu().numpy()])
                i += 1
        return all_feat, all_labels
    
    def get_mem_rep_labels(self, eval=True, use_proj=False):
        """Compute every representation -labels pairs from memory
        Args:
            eval (bool, optional): Whether to turn the mdoel in evaluation mode. Defaults to True.
        Returns:
            representation - labels pairs
        """
        if eval: self.model.eval()
        mem_imgs, mem_labels = self.buffer.get_all()
        batch_s = 10
        n_batch = len(mem_imgs) // batch_s
        all_reps = []
        for i in range(n_batch):
            mem_imgs_b = mem_imgs[i*batch_s:(i+1)*batch_s].to(self.device)
            mem_imgs_b = self.transform_test(mem_imgs_b)
            if use_proj:
                _, mem_representations_b = self.model(mem_imgs_b)
            else:
                mem_representations_b, _ = self.model(mem_imgs_b)
            all_reps.append(mem_representations_b)
        mem_representations = torch.cat(all_reps, dim=0)
        return mem_representations, mem_labels

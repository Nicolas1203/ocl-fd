import torch
import time
import torch.nn as nn
import sys
import logging as lg
import numpy as np

from torch.utils.data import DataLoader

from src.learners.base import BaseLearner
from src.utils.losses import SupConLoss
from src.buffers.reservoir import Reservoir
from src.utils import name_match

class LUMPLearner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.buffer = name_match.buffers[self.params.buffer](
                max_size=self.params.mem_size,
                img_size=self.params.img_size,
                nb_ch=self.params.nb_channels,
                n_classes=self.params.n_classes,
                drop_method=self.params.drop_method
            )

    def load_criterion(self):
        return SupConLoss(self.params.temperature) 

    def train(self, dataloader, task_name, **kwargs):
        self.model = self.model.train()

        for j, batch in enumerate(dataloader):
            # Stream data
            batch_x, batch_y = batch[0], batch[1]
            self.stream_idx += len(batch_x)
            
            for _ in range(self.params.mem_iters):
                mem_x, mem_y = self.buffer.random_retrieve(n_imgs=self.params.mem_batch_size)

                if mem_x.size(0) > 0:
                    # Augment
                    lam = np.random.beta(0.4, 0.4)
                    mixed_x = lam * batch_x.to(self.device) + (1 - lam) * mem_x[:batch_x.shape[0]].to(self.device)

                    self.model.train()

                    # Inference
                    f1, p1 = self.model(self.transform_train(mixed_x))
                    f2, p2 = self.model(self.transform_train(mixed_x))

                    # features = torch.cat([f1_mem.unsqueeze(1), f2_mem.unsqueeze(1)], dim=1)
                    projections = torch.cat([
                        p1.unsqueeze(1),
                        p2.unsqueeze(1),
                        ], 
                        dim=1
                        )
                    # Loss
                    loss = self.criterion(features=projections, labels=None)
                    loss = loss.mean()
                    self.loss = loss.item()
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()
                    
            # Update buffer
            self.buffer.update(imgs=batch_x, labels=batch_y, model=self.model)

            # Plot to tensorboard
            if self.params.tensorboard:
                self.plot()

            if (j == (len(dataloader) - 1)) and (j > 0):
                lg.info(
                    f"Phase : {task_name}   batch {j}/{len(dataloader)}   Loss : {self.loss:.4f}    time : {time.time() - self.start:.4f}s"
                )
                self.save(model_name=f"ckpt_{task_name}.pth")
    
    def plot(self):
        self.writer.add_scalar("loss", self.loss, self.stream_idx)
        self.writer.add_scalar("n_added_so_far", self.buffer.n_added_so_far, self.stream_idx)
        percs = self.buffer.get_labels_distribution()
        for i in range(self.params.n_classes):
            self.writer.add_scalar(f"labels_distribution/{i}", percs[i], self.stream_idx)

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
import os
import matplotlib.pyplot as plt
import wandb

from sklearn.metrics import accuracy_score, confusion_matrix
from copy import deepcopy

from src.learners.base import BaseLearner
from src.utils.losses import vMFLoss, AGDLoss
from src.utils import name_match
from src.utils.utils import get_device
from src.utils.metrics import forgetting_line 

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

        if self.params.eval_proj:
            self.results = []
            self.results_forgetting = []
        
        # Classes to infer
        self.classes_seen_so_far = torch.LongTensor(size=(0,)).to(device)

        
    def load_criterion(self):
        if self.params.fd_loss == 'vmf':
            return vMFLoss(
                var=self.params.var,
                mu=self.params.mu,
                proj_dim=self.params.proj_dim,
                norm_all=self.params.norm_all_classes
            )
        elif self.params.fd_loss == 'agd':
            return AGDLoss(
                var=self.params.var,
                mu=self.params.mu,
                proj_dim=self.params.proj_dim,
                norm_all=self.params.norm_all_classes
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
                
                # update classes seen
                present = batch_y.long().unique().to(device)
                self.classes_seen_so_far = torch.cat([self.classes_seen_so_far, present]).unique()
                
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
                # update classes seen
                present = batch_y.unique().to(device)
                self.classes_seen_so_far = torch.cat([self.classes_seen_so_far, present]).unique()
                
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
    
    def evaluate(self, dataloaders, task_id):
        if self.params.eval_proj:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                self.model.eval()
                accs = []

                all_preds = []
                all_targets = []
                for j in range(task_id + 1):
                    test_preds, test_targets = self.encode(dataloaders[f"test{j}"], use_proj=self.params.eval_proj)
                    acc = accuracy_score(test_preds, test_targets)
                    accs.append(acc)
                    if not self.params.no_wandb:
                        all_preds = np.concatenate([all_preds, test_preds])
                        all_targets = np.concatenate([all_targets, test_targets])
                        wandb.log({
                            f"acc_{j}": acc,
                            "task_id": task_id
                        })
                for _ in range(self.params.n_tasks - task_id - 1):
                    accs.append(np.nan)
                self.results.append(accs)
                
                line = forgetting_line(pd.DataFrame(self.results), task_id=task_id, n_tasks=self.params.n_tasks)
                line = line[0].to_numpy().tolist()
                self.results_forgetting.append(line)

                self.print_results(task_id)
                
                # Make confusion matrix
                if not self.params.no_wandb:
                    # re-index to have classes in task order
                    all_targets = [self.params.labels_order.index(int(i)) for i in all_targets]
                    all_preds = [self.params.labels_order.index(int(i)) for i in all_preds]
                    n_im_pt = self.params.n_classes // self.params.n_tasks
                    cm = confusion_matrix(all_targets, all_preds)
                    cm_log = np.log(1 + cm)
                    fig = plt.matshow(cm_log)
                    wandb.log({
                            "cm_raw": cm,
                            "cm": fig,
                            "task_id": task_id
                        })

                return np.nanmean(self.results[-1]), np.nanmean(self.results_forgetting[-1])
        else:
            return super().evaluate(dataloaders, task_id)
    
    def save_results(self):
        if self.params.eval_proj:
            if self.params.run_id is not None:
                results_dir = os.path.join(self.params.results_root, self.params.tag, f"run{self.params.run_id}")
            else:
                results_dir = os.path.join(self.params.results_root, self.params.tag, f"run{self.params.seed}")
            print(f"Saving accuracy results in : {results_dir}")
            if not os.path.exists(results_dir):
                os.makedirs(results_dir, exist_ok=True)
            
            df_avg = pd.DataFrame()
            cols = [f'task {i}' for i in range(self.params.n_tasks)]
            # Loop over classifiers results
            # Each classifier has a value for every task. NaN if future task
            df_clf = pd.DataFrame(self.results, columns=cols)
            # Average accuracy over all tasks with not NaN value
            df_avg = df_clf.mean(axis=1)
            df_clf.to_csv(os.path.join(results_dir, 'acc.csv'), index=False)
            df_avg.to_csv(os.path.join(results_dir, 'avg.csv'), index=False)
            
            df_avg = pd.DataFrame()
            print(f"Saving forgetting results in : {results_dir}")
            cols = [f'task {i}' for i in range(self.params.n_tasks)]
            df_clf = pd.DataFrame(self.results_forgetting, columns=cols)
            df_avg = df_clf.mean(axis=1)
            df_clf.to_csv(os.path.join(results_dir, 'forgetting.csv'), index=False)
            df_avg.to_csv(os.path.join(results_dir, 'avg_forgetting.csv'), index=False)

            self.save_parameters()
        else:
            super().save_results()
    
    def print_results(self, task_id):
        if self.params.eval_proj:
            n_dashes = 20
            pad_size = 8
            print('-' * n_dashes + f"TASK {task_id + 1} / {self.params.n_tasks}" + '-' * n_dashes)
            
            print('-' * n_dashes + "ACCURACY" + '-' * n_dashes)        
            for line in self.results:
                print('Acc'.ljust(pad_size) + ' '.join(f'{value:.4f}'.ljust(pad_size) for value in line), f"{np.nanmean(line):.4f}")
        else:
            super().print_results(task_id)
    
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
                    logits = self.model.logits(self.transform_test(inputs))
                    preds = logits[:,:self.classes_seen_so_far.long().max()].argmax(dim=1)

                    if i == 0:
                        all_labels = labels.cpu().numpy()
                        all_feat = preds.cpu().numpy()
                    else:
                        all_labels = np.hstack([all_labels, labels.cpu().numpy()])
                        all_feat = np.hstack([all_feat, preds.cpu().numpy()])
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
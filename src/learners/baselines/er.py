import torch
import time
import torch.nn as nn
import random as r
import numpy as np
import os
import pandas as pd

from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score

from src.learners.ce import CELearner
from src.utils.losses import SupConLoss
from src.buffers.reservoir import Reservoir
from src.models.resnet import ResNet18
from src.utils.metrics import forgetting_line   
from src.utils.utils import get_device

device = get_device()

class ERLearner(CELearner):
    def __init__(self, args):
        super().__init__(args)
        self.results = []
        self.results_forgetting = []
        self.buffer = Reservoir(
            max_size=self.params.mem_size,
            img_size=self.params.img_size,
            nb_ch=self.params.nb_channels,
            n_classes=self.params.n_classes,
            drop_method=self.params.drop_method,
        )
    
    def load_model(self, **kwargs):
        return ResNet18(
            dim_in=self.params.dim_in,
            nclasses=self.params.n_classes,
            nf=self.params.nf
        ).to(device)
    
    def train(self, dataloader, **kwargs):
        task_name  = kwargs.get('task_name', 'unknown task')
        self.model = self.model.train()

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
                    combined_x = self.transform_train(combined_x)

                    # Inference
                    logits = self.model.logits(combined_x)

                    # Loss
                    loss = self.criterion(logits, combined_y.long())
                    self.loss = loss.item()
                    print(f"Loss {self.loss:.3f}  batch {j}", end="\r")
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()
            
            # Update reservoir buffer
            self.buffer.update(imgs=batch_x, labels=batch_y, model=self.model)

            if (j == (len(dataloader) - 1)) and (j > 0):
                print(
                    f"Task : {task_name}   batch {j}/{len(dataloader)}   Loss : {loss.item():.4f}    time : {time.time() - self.start:.4f}s"
                )

    def evaluate_offline(self, dataloaders, epoch):
        with torch.no_grad():
            self.model.eval()

            test_preds, test_targets = self.encode(dataloaders['test'])
            acc = accuracy_score(test_preds, test_targets)
            self.results.append(acc)
            
        print(f"ACCURACY {self.results[-1]}")
        return self.results[-1]
    
    def evaluate(self, dataloaders, task_id):
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            self.model.eval()
            accs = []

            for j in range(task_id + 1):
                test_preds, test_targets = self.encode(dataloaders[f"test{j}"])
                acc = accuracy_score(test_preds, test_targets)
                accs.append(acc)
            for _ in range(self.params.n_tasks - task_id - 1):
                accs.append(np.nan)
            self.results.append(accs)
            
            line = forgetting_line(pd.DataFrame(self.results), task_id=task_id, n_tasks=self.params.n_tasks)
            line = line[0].to_numpy().tolist()
            self.results_forgetting.append(line)

            self.print_results(task_id)

            return np.nanmean(self.results[-1]), np.nanmean(self.results_forgetting[-1])
    
    def print_results(self, task_id):
        n_dashes = 20
        pad_size = 8
        print('-' * n_dashes + f"TASK {task_id + 1} / {self.params.n_tasks}" + '-' * n_dashes)
        
        print('-' * n_dashes + "ACCURACY" + '-' * n_dashes)        
        for line in self.results:
            print('Acc'.ljust(pad_size) + ' '.join(f'{value:.4f}'.ljust(pad_size) for value in line), f"{np.nanmean(line):.4f}")
    
    def encode(self, dataloader, nbatches=-1):
        i = 0
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            for sample in dataloader:
                if nbatches != -1 and i >= nbatches:
                    break
                inputs = sample[0]
                labels = sample[1]
                
                inputs = inputs.to(device)
                logits = self.model.logits(self.transform_test(inputs))
                preds = logits.argmax(dim=1)

                if i == 0:
                    all_labels = labels.cpu().numpy()
                    all_feat = preds.cpu().numpy()
                    i += 1
                else:
                    all_labels = np.hstack([all_labels, labels.cpu().numpy()])
                    all_feat = np.hstack([all_feat, preds.cpu().numpy()])
        return all_feat, all_labels
    
    def save_results(self):
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
    
    def save_results_offline(self):
        if self.params.run_id is not None:
            results_dir = os.path.join(self.params.results_root, self.params.tag, f"run{self.params.run_id}")
        else:
            results_dir = os.path.join(self.params.results_root, self.params.tag)

        print(f"Saving accuracy results in : {results_dir}")
        if not os.path.exists(results_dir):
            os.makedirs(results_dir, exist_ok=True)

        pd.DataFrame(self.results).to_csv(os.path.join(results_dir, 'acc.csv'), index=False)

        self.save_parameters()

    def combine(self, batch_x, batch_y, mem_x, mem_y):
        mem_x, mem_y = mem_x.to(self.device), mem_y.to(self.device)
        batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
        combined_x = torch.cat([mem_x, batch_x])
        combined_y = torch.cat([mem_y, batch_y])
        return combined_x, combined_y

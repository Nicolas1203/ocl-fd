"""
Code adapted from https://github.com/DigiTurk84/class-incremental-polytope
"""
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
from src.models.fixed_resnet import FixedResNet18_cifar
from src.utils.metrics import forgetting_line   
from src.utils.utils import get_device

device = get_device()

class PFCLearner(CELearner):
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
        if self.params.drop_fc:
            self.init_results()
        
    def load_model(self, **kwargs):
        out_dim = self.params.n_classes
        fixed_classifier_feat_dim = out_dim - 1
        fixed_weights = torch.from_numpy(self.dsimplex(num_classes=out_dim).transpose())
        
        model = FixedResNet18_cifar(
            out_dim=out_dim,
            fixed_classifier_feat_dim=fixed_classifier_feat_dim
            ).to(device)

        model.last.weight.requires_grad = False        
        model.last.weight.copy_(fixed_weights)
        
        return model
    
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
                    combined_x = self.augment(combined_x)
                    combined_y = torch.cat([combined_y.long() for _ in range(self.params.n_augs+1)]) if self.params.n_augs > 1 else combined_y
                    print(combined_y.shape)
                    # Inference
                    logits = self.model(combined_x)

                    # Loss
                    loss = self.criterion(logits, combined_y.long())
                    self.loss = loss.item()
                    print(f"Loss {self.loss:.3f}  batch {j}", end="\r")
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()
            
            # Update reservoir buffer
            self.buffer.update(imgs=batch_x, labels=batch_y, model=self.model)
            # print(self.model.last.weight)
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
        if not self.params.drop_fc:
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
        else:
            return super().evaluate(dataloaders, task_id)
    
    def print_results(self, task_id):
        if not self.params.drop_fc:
            n_dashes = 20
            pad_size = 8
            print('-' * n_dashes + f"TASK {task_id + 1} / {self.params.n_tasks}" + '-' * n_dashes)
            
            print('-' * n_dashes + "ACCURACY" + '-' * n_dashes)        
            for line in self.results:
                print('Acc'.ljust(pad_size) + ' '.join(f'{value:.4f}'.ljust(pad_size) for value in line), f"{np.nanmean(line):.4f}")
        else:
            super().print_results(task_id)
    
    def encode(self, dataloader, nbatches=-1):
        if self.params.drop_fc:
            i = 0
            with torch.no_grad():
                for sample in dataloader:
                    if nbatches != -1 and i >= nbatches:
                        break
                    inputs = sample[0]
                    labels = sample[1]
                    
                    inputs = inputs.to(self.device)
                    features = self.model.features(self.transform_test(inputs))
                    
                    if i == 0:
                        all_labels = labels.cpu().numpy()
                        all_feat = features.cpu().numpy()
                    else:
                        all_labels = np.hstack([all_labels, labels.cpu().numpy()])
                        all_feat = np.vstack([all_feat, features.cpu().numpy()])
                    i += 1
            return all_feat, all_labels
        else:
            i = 0
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                for sample in dataloader:
                    if nbatches != -1 and i >= nbatches:
                        break
                    inputs = sample[0]
                    labels = sample[1]
                    
                    inputs = inputs.to(device)
                    logits = self.model(self.transform_test(inputs))
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

    def get_mem_rep_labels(self, eval=True, **kwargs):
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
            mem_representations_b = self.model.features(mem_imgs_b)
            all_reps.append(mem_representations_b)
        mem_representations = torch.cat(all_reps, dim=0)
        return mem_representations, mem_labels
    
    def augment(self, combined_x, **kwargs):
        with torch.no_grad():
            augmentations = []
            for _ in range(self.params.n_augs):
                augmentations.append(self.transform_train(combined_x))
            # If its 1, we train as usual.
            # If above 1, we add raw data for fair comparison with FD-AGD, and SCR.
            if self.params.n_augs > 1:
                augmentations.append(combined_x)
            return torch.cat(augmentations)
    
    def dsimplex(self, num_classes=10):
        def simplex_coordinates2(m):
            # add the credit
            import numpy as np

            x = np.zeros([m, m + 1])
            for j in range(0, m):
                x[j, j] = 1.0

            a = (1.0 - np.sqrt(float(1 + m))) / float(m)

            for i in range(0, m):
                x[i, m] = a

            #  Adjust coordinates so the centroid is at zero.
            c = np.zeros(m)
            for i in range(0, m):
                s = 0.0
                for j in range(0, m + 1):
                    s = s + x[i, j]
                c[i] = s / float(m + 1)

            for j in range(0, m + 1):
                for i in range(0, m):
                    x[i, j] = x[i, j] - c[i]

            #  Scale so each column has norm 1. UNIT NORMALIZED
            s = 0.0
            for i in range(0, m):
                s = s + x[i, 0] ** 2
            s = np.sqrt(s)

            for j in range(0, m + 1):
                for i in range(0, m):
                    x[i, j] = x[i, j] / s

            return x

        feat_dim = num_classes - 1
        ds = simplex_coordinates2(feat_dim)
        return ds

from __future__ import print_function
import torch
import torch.nn.functional as F

from torch import nn
from copy import deepcopy

from src.utils.utils import get_device, AG_SawSeriesPT

device = get_device()
eps = 1e-7

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all'):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode

    # @profile
    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -1 * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size)

        return loss


class vMFLoss(nn.Module):
    def __init__(self, var=1, **kwargs):
        super().__init__()
        self.var = var
        self.dim = kwargs.get('proj_dim', 128)
        self.mu = kwargs.get('mu', 1)
        self.init_class_seen()
        self.init_means()

    def init_class_seen(self):
        self.class_seen = torch.LongTensor(size=(0,)).to(device)
    
    def init_means(self):
        self.means = None

    def update_class_seen(self, labels):
        new_classes = labels.unique()
        self.class_seen = torch.cat([self.class_seen, new_classes]).unique()

    def update_means(self, labels):
        curr = deepcopy(self.class_seen)
        self.update_class_seen(labels)
        if len(self.class_seen) > len(curr):
            self.means = torch.eye(max(self.class_seen) + 1, self.dim).to(device) 
                
    def get_means(self):
        return self.means
    
    # @profile
    def forward(self, features, labels, **kwargs):
        nviews = features.shape[1]
        labels = labels.contiguous().view(-1).short()

        # fixing the means manually
        self.update_means(labels)
        
        uniq = [i for i in range(int(max(self.class_seen) + 1))]

        mask = torch.eq(torch.tensor(uniq).unsqueeze(1).to(device),labels.unsqueeze(0))
        mask = mask.repeat(1, nviews).float().to(device)

        features_flat = torch.cat(torch.unbind(features, dim=1), dim=0)
        
        mc = self.get_means().to(device)
        
        mask_mean = labels.unique().long()

        zi_mc = (features_flat.unsqueeze(0) - mc.unsqueeze(1)) # z_i - mc
        norm_zi_mc = - (zi_mc ** 2).sum(dim=2) / (2*self.var)    # -||z_i - m_c||^2 / 2sigma
        logits_exps = torch.exp(norm_zi_mc).to(device)
        
        norms_exp = logits_exps[mask_mean, :].sum(0)

        # Compute final loss
        loss = - ((norm_zi_mc - torch.log(norms_exp + 1e-8)) * mask)[mask_mean, :]
        loss = loss.sum()

        return loss


class AGDLoss(nn.Module):
    def __init__(self, var=1, **kwargs):
        super().__init__()
        self.var = var
        self.dim = kwargs.get('proj_dim', 128)
        self.mu = kwargs.get('mu', 1)
        self.init_class_seen()

    def init_class_seen(self):
        self.class_seen = torch.LongTensor(size=(0,)).to(device)
    
    def update_class_seen(self, labels):
        new_classes = labels.unique()
        self.class_seen = torch.cat([self.class_seen, new_classes]).unique()

    # @profile
    def forward(self, features, labels=None, **kwargs):
        nviews = features.shape[1]
        labels = labels.contiguous().view(-1).short()

        self.update_class_seen(labels)
        
        uniq = [i for i in range(int(max(self.class_seen) + 1))]

        mask = torch.eq(torch.tensor(uniq).unsqueeze(1).to(device),labels.unsqueeze(0))
        mask = mask.repeat(1, nviews).long().to(device)

        features_expand = (torch.cat(torch.unbind(features, dim=1), dim=0)).expand(mask.shape[0], mask.shape[1], features.shape[-1])
        maskoh = F.one_hot(torch.ones_like(mask) * torch.arange(0, mask.shape[0]).to(device).view(-1, 1), features_expand.shape[-1])
        features_p = (features_expand * maskoh).sum(-1)

        densities = AG_SawSeriesPT(
            y=features_p.double(),
            sigma2=torch.tensor([self.var], dtype=torch.float64).to(device),
            d=torch.tensor([self.dim], dtype=torch.float64).to(device),
            N=torch.arange(0,40)
            ).to(device)

        mask_mean = labels.unique().long()
        norms_densities = densities[mask_mean, :].sum(0, keepdim=True)

        # Compute final loss
        loss = - (torch.log(densities / norms_densities) * mask)[mask_mean, :].sum()
        
        return loss


class BYOLLoss(nn.Module):
    """
    Implements BYOL (https://arxiv.org/abs/2006.07733)
    """
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction='sum')

    def forward(self, pred1, ema_pred1, pred2, ema_pred2):
        pred1 = F.normalize(pred1, dim=1)
        ema_pred1 = F.normalize(ema_pred1, dim=1)
        pred2 = F.normalize(pred2, dim=1)
        ema_pred2 = F.normalize(ema_pred2, dim=1)
        mse_loss = (self.mse_loss(pred1, ema_pred2) + self.mse_loss(pred2, ema_pred1)) / 2
        return mse_loss

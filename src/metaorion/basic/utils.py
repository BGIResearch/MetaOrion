import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


class GHM_Loss(nn.Module):
    def __init__(self, bins=10, alpha=0.5):
        '''
        bins: split to n bins
        alpha: hyper-parameter 0.75/0.9
        '''
        super(GHM_Loss, self).__init__()
        self._bins = bins
        self._alpha = alpha
        self._last_bin_count = None

    def _g2bin(self, g):
        return torch.floor(g * (self._bins - 0.0001)).long()

    def _custom_loss(self, x, target, weight):
        raise NotImplementedError

    def _custom_loss_grad(self, x, target):
        raise NotImplementedError

    def forward(self, x, target):
        x = x.view(-1)
        target = target.view(-1)

        g = torch.abs(self._custom_loss_grad(x, target)).detach()

        bin_idx = self._g2bin(g)

        bin_count = torch.zeros((self._bins),device=x.device)
        for i in range(self._bins):
            bin_count[i] = (bin_idx == i).sum().item()

        N = x.numel()
        # N = (x.size(0) * x.size(1))

        if self._last_bin_count is None:
            self._last_bin_count = bin_count
        else:
            bin_count = self._alpha * self._last_bin_count + (1 - self._alpha) * bin_count
            self._last_bin_count = bin_count

        nonempty_bins = (bin_count > 0).sum().item()

        gd = bin_count * nonempty_bins
        gd = torch.clamp(gd, min=0.0001)
        beta = N / gd

        return self._custom_loss(x, target, beta[bin_idx])


class GHMC_Loss(GHM_Loss):
    '''
        GHM_Loss for classification
    '''

    def __init__(self, bins, alpha):
        super(GHMC_Loss, self).__init__(bins, alpha)

    def _custom_loss(self, x, target, weight):
        return F.binary_cross_entropy_with_logits(x.flatten(), target, weight=weight)

    def _custom_loss_grad(self, x, target):
        return torch.sigmoid(x).detach() - target




class EMA:
    def __init__(self, model, decay=0.995):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

def get_loss_weights():
    counts = [1990, 961, 930, 374, 296, 290, 282, 277, 265, 231, 214, 206, 177, 159, 127]
    label_info = {idx: count for idx, count in enumerate(counts)}
    class_counts = np.array(list(label_info.values()))
    # Inverse
    class_weights = [1.0 / c for c in class_counts]
    weight_norm = class_weights / np.sum(class_weights) * len(class_counts)
    return weight_norm

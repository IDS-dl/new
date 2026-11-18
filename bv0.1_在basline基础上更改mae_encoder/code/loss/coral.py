import torch
from torch.nn import functional as F


def coral_targets(y):
    if torch.any((y<1)|(y>9)):
        raise ValueError('ordinal labels must be 1..9')
    return (y[...,None]>torch.arange(1,9,device=y.device)).float()


def coral_loss(logits, y):
    return F.binary_cross_entropy_with_logits(logits,coral_targets(y),reduction='none').sum(-1).mean()

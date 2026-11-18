import torch
from torch import nn
from torch.nn import functional as F


def coral_targets(y):
    if torch.any((y<1)|(y>9)):
        raise ValueError('ordinal labels must be 1..9')
    return (y[...,None]>torch.arange(1,9,device=y.device)).float()


class CoralHead(nn.Module):
    """Shared rank weight with strictly increasing learned cutpoints.

    A constrained CORAL variant: logits = score - ordered thresholds.
    Positive softplus increments guarantee rank consistency at every step.
    """
    def __init__(self, hidden):
        super().__init__()
        self.score=nn.Linear(hidden,1,bias=False)
        self.first=nn.Parameter(torch.tensor(-3.5))
        self.increments=nn.Parameter(torch.full((7,), .54132485))

    def thresholds(self):
        return torch.cat([self.first[None],self.first+torch.cumsum(F.softplus(self.increments)+1e-6,0)])

    def forward(self,x):
        return self.score(x)-self.thresholds()


def coral_loss(logits, y):
    return F.binary_cross_entropy_with_logits(logits,coral_targets(y),reduction='none').sum(-1).mean()


def coral_predict(logits):
    p=logits.sigmoid()
    return 1+(p>.5).sum(-1), 1+p.sum(-1)

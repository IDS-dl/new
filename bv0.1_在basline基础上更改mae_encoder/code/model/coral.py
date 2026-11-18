import torch
from torch import nn
from torch.nn import functional as F


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

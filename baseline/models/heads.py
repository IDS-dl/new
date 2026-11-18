from torch import nn
from .coral import CoralHead


def make_head(task, hidden):
    if task=='coral':
        return CoralHead(hidden)
    if task=='regression':
        return nn.Sequential(nn.Linear(hidden,1),nn.Sigmoid())
    if task not in ('binary','softmax'):
        raise ValueError(task)
    return nn.Linear(hidden,2 if task=='binary' else 9)

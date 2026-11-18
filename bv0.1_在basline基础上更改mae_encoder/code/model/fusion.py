from torch import nn


def fusion_mlp(input_dim, hidden, dropout):
    return nn.Sequential(nn.Linear(input_dim,hidden),nn.LayerNorm(hidden),nn.GELU(),nn.Dropout(dropout))

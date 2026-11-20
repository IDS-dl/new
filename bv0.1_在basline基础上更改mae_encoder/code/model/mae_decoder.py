import torch
from torch import nn
from .mae_encoder import RATES, position, transformer


def restore_tokens(visible, mask, mask_token):
    full = mask_token.expand(*mask.shape, -1).clone()
    full[~mask] = visible.flatten(0, 1)
    return full


class MAEDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dim = cfg['decoder_dim']
        self.project = nn.Linear(cfg['encoder_dim'], dim)
        self.mask_token = nn.Parameter(torch.randn(dim) * .02)
        self.modality_embedding = nn.Parameter(torch.randn(2, dim) * .02)
        self.transformer = transformer(cfg, 'decoder')
        self.heads = nn.ModuleDict({name: nn.Linear(dim, width) for name, width in RATES.items()})

    def forward(self, encoded, masks):
        lengths = [int((~masks[name][0]).sum()) for name in RATES]
        parts = self.project(encoded).split(lengths, dim=1)
        full = []
        for i, (name, visible) in enumerate(zip(RATES, parts)):
            restored = restore_tokens(visible, masks[name], self.mask_token)
            full.append(restored + position(restored) + self.modality_embedding[i])
        decoded = self.transformer(torch.cat(full, dim=1)).chunk(2, dim=1)
        return {name: self.heads[name](tokens) for name, tokens in zip(RATES, decoded)}

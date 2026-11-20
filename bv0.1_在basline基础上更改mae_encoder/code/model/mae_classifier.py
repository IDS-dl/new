import torch
from torch import nn
from .mae_encoder import MAEEncoder
from .fusion import fusion_mlp
from .coral import CoralHead


class MAEClassifier(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = MAEEncoder(cfg)
        dim = cfg['encoder_dim']
        self.fusion = fusion_mlp(2 * dim, dim, cfg['dropout'])
        self.heads = nn.ModuleDict({key: CoralHead(dim) for key in ('valence', 'arousal')})
        self.frozen = cfg['freeze_encoder']
        self.encoder.requires_grad_(not self.frozen)

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.encoder.eval()
        return self

    def forward(self, ppg, eda):
        parts = self.encoder.encode_full(ppg, eda).chunk(2, dim=1)
        fused = self.fusion(torch.cat([part.mean(1) for part in parts], dim=-1))
        return {name + '_logits': head(fused) for name, head in self.heads.items()}

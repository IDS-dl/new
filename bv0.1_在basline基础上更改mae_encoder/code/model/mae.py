from torch import nn
from .mae_encoder import MAEEncoder, random_masks, patchify
from .mae_decoder import MAEDecoder


class WaveformMAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = MAEEncoder(cfg)
        self.decoder = MAEDecoder(cfg)
        self.ratio = cfg['mask_ratio']
        self.strategy = cfg['mask_type']

    def forward(self, ppg, eda, generator=None):
        batch, count, _ = patchify(ppg, 64).shape
        masks = random_masks(batch, count, self.ratio, ppg.device, generator, self.strategy)
        encoded = self.encoder.encode_visible(ppg, eda, masks)
        return self.decoder(encoded, masks), masks

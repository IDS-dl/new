"""One-second waveform patches and the shared visible-token encoder."""
import torch
from torch import nn

RATES = {'ppg': 64, 'eda': 4}


def patchify(wave, width):
    if wave.ndim != 3 or wave.size(1) != 1 or not wave.size(2) or wave.size(2) % width:
        raise ValueError('Expected [B,1,T] with a whole number of one-second patches')
    return wave[:, 0].reshape(wave.size(0), -1, width)


def unpatchify(patches):
    return patches.flatten(1).unsqueeze(1)


def random_masks(batch, count, ratio, device, generator=None, mask_type='random'):
    if mask_type != 'random':
        raise NotImplementedError(f'Unsupported mask strategy: {mask_type}')
    if count < 2 or not 0 < ratio < 1:
        raise ValueError('Masking requires at least two patches and 0 < mask_ratio < 1')
    hidden = min(count - 1, max(1, round(count * ratio)))
    return {name: torch.rand(batch, count, device=device, generator=generator).argsort(1) < hidden
            for name in RATES}


def position(tokens):
    n, dim = tokens.shape[1:]
    t = torch.arange(n, device=tokens.device)[:, None]
    freq = 10000 ** (-torch.arange(0, dim, 2, device=tokens.device) / dim)
    return torch.stack(((t * freq).sin(), (t * freq).cos()), -1).flatten(1)[:, :dim].to(tokens.dtype)


def transformer(cfg, prefix):
    dim = cfg[prefix + '_dim']
    layer = nn.TransformerEncoderLayer(dim, cfg[prefix + '_heads'],
        int(dim * cfg[prefix + '_mlp_ratio']), cfg[prefix + '_dropout'],
        activation='gelu', batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, cfg[prefix + '_depth'], nn.LayerNorm(dim),
                                 enable_nested_tensor=False)


class MAEEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dim = cfg['encoder_dim']
        self.embedding = nn.ModuleDict({name: nn.Sequential(nn.Linear(size, dim),
            nn.LayerNorm(dim), nn.GELU()) for name, size in RATES.items()})
        self.modality_embedding = nn.Parameter(torch.randn(2, dim) * .02)
        self.transformer = transformer(cfg, 'encoder')

    def encode_visible(self, ppg, eda, masks):
        parts = []
        for i, (name, wave) in enumerate(zip(RATES, (ppg, eda))):
            tokens = self.embedding[name](patchify(wave, RATES[name]))
            if i == 0:
                shape = tokens.shape[:2]
            elif tokens.shape[:2] != shape:
                raise ValueError('PPG and EDA must have aligned windows')
            tokens = tokens + position(tokens) + self.modality_embedding[i]
            if masks is not None:
                mask = masks[name]
                if mask.dtype != torch.bool or mask.shape != shape or not torch.all(mask.sum(1) == mask[0].sum()):
                    raise ValueError('Expected a boolean mask with equal visible counts per batch')
                tokens = tokens[~mask].reshape(shape[0], -1, tokens.size(-1))
            parts.append(tokens)
        return self.transformer(torch.cat(parts, dim=1))

    def encode_full(self, ppg, eda):
        return self.encode_visible(ppg, eda, None)

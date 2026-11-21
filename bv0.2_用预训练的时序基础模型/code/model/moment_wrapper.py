"""Native-rate segmentation and explicit MOMENT observation masks."""
import torch

RATES = {'ppg': 64, 'eda': 4}


def segment(waves):
    length = waves.shape[-1]
    if not length or length % 8:
        raise ValueError('Expected complete eight-point patches')
    size = ((length + 511) // 512) * 512
    valid = (torch.arange(size, device=waves.device) < length).expand(waves.shape[0], -1)
    return torch.nn.functional.pad(waves, (0, size-length)).reshape(-1, 1, 512), valid.reshape(-1, 512)


def stitch(segments, batch, length):
    return segments.reshape(batch, -1)[:, :length]


def observation_mask(valid, ratio, generator):
    if not 0 < ratio < 1:
        raise ValueError('mask ratio must be between zero and one')
    patches = valid.reshape(-1, 64, 8).all(-1)
    count = patches.sum(-1)
    if (count < 2).any():
        raise ValueError('Each segment requires at least two valid patches')
    hidden = (count.float() * ratio).round().long().clamp_min(1).minimum(count-1)
    scores = torch.rand(patches.shape, generator=generator, device=valid.device)
    ranks = scores.masked_fill(~patches, 2).argsort(1).argsort(1)
    return (patches & (ranks >= hidden[:, None])).repeat_interleave(8, dim=1)


def reconstruct(model, signal, valid, observed):
    # Explicitly remove hidden and padded values before the official forward.
    visible = signal.masked_fill(~observed[:, None], 0)
    return model(x_enc=visible, input_mask=valid.long(),
                 mask=observed.long()).reconstruction

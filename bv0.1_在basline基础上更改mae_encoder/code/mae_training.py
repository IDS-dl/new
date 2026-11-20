"""Fold-local MAE training, transfer validation and reconstruction diagnostics."""
from collections import defaultdict
from pathlib import Path

import torch
from loss.mae import reconstruction_loss
from model.mae import WaveformMAE
from model import model_from_config


def build_stage(cfg, fold, identity, device):
    if cfg['model'] == 'waveform_mae':
        model = WaveformMAE(cfg).to(device)
    else:
        model = model_from_config(cfg).to(device)
    if cfg['model'] == 'mae_classifier':
        if cfg.get('pretrained_root'):
            path = Path(cfg['pretrained_root']) / 'weights' / f'fold_{fold}' / 'best_mae.pt'
            saved = torch.load(path, map_location='cpu', weights_only=False)
            if saved.get('fold_id') != fold or saved.get('identity') != identity:
                raise ValueError(f'Fold, data or preprocessing mismatch in {path}')
            for key in cfg:
                if key.startswith('encoder_') and key != 'encoder_lr' and saved['config'].get(key) != cfg[key]:
                    raise ValueError(f'Encoder configuration mismatch for {key}')
            model.encoder.load_state_dict(saved['encoder'], strict=True)
        params = [
            {'params': model.encoder.parameters(), 'lr': cfg['encoder_lr']},
            {'params': list(model.fusion.parameters()) + list(model.heads.parameters()), 'lr': cfg['head_lr']},
        ]
    else:
        params = model.parameters()
    optimizer = torch.optim.AdamW(params, lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    return model, optimizer


def reconstruction_epoch(model, loader, device, cfg, optimizer=None, generator=None):
    model.train(optimizer is not None)
    totals = defaultdict(float)
    count = 0
    with torch.set_grad_enabled(optimizer is not None):
        for batch in loader:
            waves = {name: batch[name].to(device) for name in ('ppg', 'eda')}
            output, masks = model(**waves, generator=generator)
            losses = reconstruction_loss(output, waves, masks, cfg)
            if not torch.isfinite(losses['loss']):
                raise FloatingPointError('Nonfinite MAE loss')
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                losses['loss'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
            size = len(waves['ppg'])
            count += size
            for key, value in losses.items():
                totals[key] += value.item() * size
    if not count:
        raise ValueError('Empty MAE split')
    return {key: total / count for key, total in totals.items()}


@torch.no_grad()
def reconstruction_plot(model, loader, device, seed, path):
    import matplotlib.pyplot as plt
    model.eval()
    batch = next(iter(loader))
    waves = {name: batch[name][:1].to(device) for name in ('ppg', 'eda')}
    output, masks = model(**waves, generator=torch.Generator(device=device).manual_seed(seed))
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    for ax, (name, rate) in zip(axes, [('ppg', 64), ('eda', 4)]):
        target = waves[name].flatten().cpu()
        time = torch.arange(len(target)) / rate
        ax.plot(time, target, label='target', lw=1)
        masked_samples = masks[name][0].repeat_interleave(rate).cpu()
        completed = torch.where(masked_samples, output[name].flatten().cpu(), target)
        ax.plot(time, completed, label='observed + MAE fill', lw=.8)
        for index in masks[name][0].nonzero().flatten().tolist():
            ax.axvspan(index, index + 1, alpha=.18, color='orange')
        ax.set_ylabel(name.upper())
        ax.legend()
    axes[-1].set_xlabel('Seconds (orange = masked, white = visible)')
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

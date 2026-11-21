"""Compare longer independent full fine-tuning with the original Stage 1 run."""
import argparse
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from evaluate_moment import predict_batches
from metrics.reconstruction import measures
from model.moment_loader import load_moment, REVISION
from train_moment import data, train_one


def run(reference, output, epochs):
    old = reference / 'results'
    official = reference / 'weights/official'
    cfg = yaml.safe_load((old / 'config.yaml').read_text())
    device = torch.device(cfg['resolved_device'])
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is required to reproduce the original Fold 0 masks')
    torch.set_num_threads(cfg['cpu_threads'])
    output.mkdir(parents=True, exist_ok=False)
    results = output / 'results'
    results.mkdir()
    (output / 'README.md').write_text(
        f'# Independent MOMENT, {epochs} epochs\n\n'
        'Fold 0 EmoWear. Official initialization, full fine-tuning, fixed epoch budget.\n\n'
        'See results/config.yaml and results/comparison.csv.\n\n'
        'TensorBoard command `tensorboard --logdir results/tensorboard`.\n')
    cfg.update(full_epochs=epochs, full_patience=None)
    (results / 'config.yaml').write_text(yaml.safe_dump(cfg))
    datasets, _ = data(cfg, results)
    for name in ('splits.json', 'normalizer.json', 'audit/sources.json'):
        if (results / name).read_bytes() != (old / name).read_bytes():
            raise RuntimeError(f'Original experiment differs in {name}')
    old_metrics = pd.read_csv(old / 'reconstruction_metrics.csv')
    output_rows = []
    for ratio in cfg['ratios']:
        for signal in ('ppg', 'eda'):
            name = f'tune_{ratio:g}_full_{signal}'
            torch.manual_seed(cfg['seed'])
            model = load_moment(official, 'full').to(device)
            best = train_one(model, datasets, cfg, ratio, signal, 'full',
                             results / name, device, epochs, None)
            del model
            prior = old_metrics.loc[(old_metrics.ratio == ratio) & (old_metrics.signal == signal)
                                    & (old_metrics.method == 'full independent')].iloc[0]
            paths = [reference/'weights'/name/f'best_moment_{signal}.pt',
                     best, best.with_name(f'last_moment_{signal}.pt')]
            with np.load(old/f'masks_{signal}_{ratio:g}.npz') as mask_archive:
                hidden = mask_archive['hidden']
                np.testing.assert_array_equal(mask_archive['subject'], datasets[2].meta.subject_id.to_numpy(dtype=str))
                np.testing.assert_array_equal(mask_archive['trial'], datasets[2].meta.trial_id.to_numpy(dtype=str))
            model = load_moment(official,'zero').to(device)
            for label, path in zip(('original_best', 'extended_best', 'extended_last'), paths):
                checkpoint = torch.load(path, map_location='cpu', weights_only=True)
                if (checkpoint['revision'], checkpoint['ratio'], checkpoint['method'], checkpoint['mode']) != (REVISION, ratio, signal, 'full'):
                    raise ValueError(f'Checkpoint identity mismatch: {path}')
                model.load_state_dict(checkpoint['model'], strict=True)
                target, observed, pred = predict_batches(model, datasets[2], (signal,), cfg, ratio, device)[signal]
                np.testing.assert_array_equal(hidden, ~observed)
                metrics = measures(target[hidden], pred[hidden])
                if label == 'original_best':
                    np.testing.assert_allclose(metrics['rmse'], prior.rmse, rtol=1e-5)
                if label == 'extended_last' and checkpoint['epoch'] != epochs:
                    raise RuntimeError(f'Last checkpoint did not reach epoch {epochs}')
                output_rows.append(dict(ratio=ratio, signal=signal, checkpoint=label,
                    epoch=checkpoint['epoch'], val_mse=checkpoint['val'],
                    rmse_improvement_pct=100*(1-metrics['rmse']/prior.rmse), **metrics))
            del model
            pd.DataFrame(output_rows).to_csv(results/'comparison.csv', index=False)
    fig, axes = plt.subplots(len(cfg['ratios']), 2, figsize=(11, 7), squeeze=False, layout='constrained')
    for i, ratio in enumerate(cfg['ratios']):
        for j, signal in enumerate(('ppg', 'eda')):
            name = f'tune_{ratio:g}_full_{signal}'
            df = pd.read_csv(results/name/'loss_history.csv')
            info = json.loads((results/name/'training.json').read_text())
            np.testing.assert_array_equal(df.epoch, np.arange(1,epochs+1))
            ax = axes[i,j]
            ax.plot(df.epoch, df.train, label='Train')
            ax.plot(df.epoch, df.val, label='Validation')
            ax.axvline(info['best_epoch'], color='grey', ls=':', label='Best validation')
            ax.set(title=f'{signal.upper()} · mask {ratio:g}', xlabel='Epoch', ylabel='Masked MSE', yscale='log')
            ax.legend()
    fig.savefig(results/'loss_curves.png', dpi=140)
    plt.close(fig)
    print(pd.DataFrame(output_rows).to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    run(args.reference, args.output, args.epochs)

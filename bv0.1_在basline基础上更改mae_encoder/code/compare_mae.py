"""Compare three complete, matched supervised runs."""
import argparse
import json
from pathlib import Path
import pandas as pd
from experiments import compare


def compare_mae(roots, destination):
    results = [Path(p) / 'results' for p in roots]
    configs = [json.loads((p/'run.json').read_text())['config'] for p in results]
    original, scratch, pretrained = configs
    if (original['model'], scratch['model'], pretrained['model']) != ('dual_conformer', 'mae_classifier', 'mae_classifier'):
        raise ValueError('Expected E0, E1, E2 in this order')
    if scratch.get('pretrained_root') or not pretrained.get('pretrained_root'):
        raise ValueError('Expected random initialization for E1 and pretrained initialization for E2')
    ignore = {'pretrained_root', 'output_dir'}
    if any(scratch[k] != pretrained.get(k) for k in scratch.keys() - ignore):
        raise ValueError('E1/E2 supervised configurations differ')
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    compare(results, destination)
    tables = []
    for label, folder in zip(('E0', 'E1', 'E2'), results):
        table = pd.read_csv(folder/'metrics_summary.csv', index_col=0)
        tables.append(table.rename_axis('metric').reset_index().assign(experiment=label))
    pd.concat(tables).to_csv(destination/'mae_comparison.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('roots', nargs=3)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    compare_mae(args.roots, args.output_dir)

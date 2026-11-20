"""Recompute fold metrics and figures exclusively from saved test predictions."""
from pathlib import Path
from metrics.reporting import summarize
from train import load_config_args

if __name__=='__main__':
    c=load_config_args()
    root = Path(c['output_dir'])
    if c.get('experiment_layout'):
        root = root / 'results'
    summarize(root,c['task'],c['num_folds'])

"""Recompute fold metrics and figures exclusively from saved test predictions."""
from metrics.reporting import summarize
from train import load_config_args

if __name__=='__main__':
    c=load_config_args()
    summarize(c['output_dir'],c['task'],c['num_folds'])

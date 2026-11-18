"""Run comparable ablations with an explicit, fixed training budget."""
import argparse
import json
from pathlib import Path
import pandas as pd
import yaml
from train import run_training
from audit import save_json

EXPERIMENTS=[(task,modality) for task in ('binary','coral') for modality in ('ppg_only','eda_only','ppg_eda')]+[('regression','ppg_eda'),('softmax','ppg_eda')]


def compare(roots,output):
    rows=[];reference=None
    for root in roots:
        root=Path(root)
        info=json.loads((root/'run.json').read_text())
        splits=json.loads((root/'splits.json').read_text())
        samples=pd.read_csv(root/'audit'/'samples.csv')
        cfg=info['config']
        # Validate identity, not merely equal subject counts or split indices.
        identity=dict(splits=splits,samples=samples[['subject_id','trial_id','window_start','window_end','valence_raw','arousal_raw']].to_json(),
                      budget={k:cfg[k] for k in ('seed','epochs','patience','batch_size','learning_rate','weight_decay','hidden_dim','depth','heads','dropout','ppg_normalization','eda_normalization','preprocess','quality','exclude_neutral','window_seconds','window_policy','ppg_sampling_rate','eda_sampling_rate')})
        if reference is None:
            reference=identity
        elif reference!=identity:
            raise ValueError(f'{root}: sample identity, splits or training budgets differ')
        ev=json.loads((root/'evaluation.json').read_text())
        if not ev['complete_cross_validation']:
            raise ValueError(f'{root}: incomplete cross-validation')
        table=pd.read_csv(root/'metrics_summary.csv',index_col=0)
        baseline=pd.read_csv(root/'baseline_metrics_summary.csv',index_col=0)
        for dim in ('valence','arousal'):
            names=['accuracy','macro_f1'] if cfg['task']=='binary' else ['vs_raw.mae','vs_raw.rmse','vs_ordinal.mae','vs_ordinal.rmse','vs_raw.spearman','vs_raw.within_1','expected_vs_raw.mae']
            for metric in names:
                k=dim+'.'+metric
                rows.append(dict(task=cfg['task'],modalities=cfg['modalities'],dimension=dim,metric=metric,
                                 mean=table.loc[k,'mean'],std=table.loc[k,'std'],baseline_mean=baseline.loc[k,'mean'],
                                 pilot=cfg['pilot'],synthetic=info['synthetic'],epochs=cfg['epochs']))
    out=Path(output);out.mkdir(exist_ok=True,parents=True)
    frame=pd.DataFrame(rows);frame.to_csv(out/'comparison.csv',index=False)
    save_json(out/'comparison.json',rows)
    return frame


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',required=True)
    p.add_argument('--output-root',required=True)
    p.add_argument('--epochs',type=int,default=60)
    p.add_argument('--pilot',action='store_true')
    p.add_argument('--only',nargs='+',choices=[t+'_'+m for t,m in EXPERIMENTS])
    a=p.parse_args()
    config_dir = Path(__file__).resolve().parent / 'configs'
    roots=[]
    for task,mode in EXPERIMENTS:
        name=task+'_'+mode
        if a.only and name not in a.only:
            continue
        cfg=yaml.safe_load((config_dir / f'emowear_{task}.yaml').read_text())
        root=Path(a.output_root)/name
        cfg.update(data_root=a.data_root,output_dir=str(root),modalities=mode,epochs=a.epochs,pilot=a.pilot)
        if root.exists():
            raise FileExistsError(f'{root}: choose a fresh output-root')
        run_training(cfg);roots.append(root)
    compare(roots,Path(a.output_root)/'comparison')

if __name__=='__main__':
    main()

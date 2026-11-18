"""Export timestamp-checked trial windows for checkpoint inference."""
import argparse
from pathlib import Path
import numpy as np
import yaml
from .dataset import build_windows
from .audit import save_json

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True)
    p.add_argument('--data-root',required=True)
    p.add_argument('--subject-id',required=True)
    p.add_argument('--trial-id',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    cfg=yaml.safe_load(Path(a.config).read_text())
    arrays,meta,_,_,_=build_windows(a.data_root,cfg)
    selection=meta.index[(meta.subject_id==a.subject_id)&(meta.trial_id==a.trial_id)].tolist()
    if len(selection)!=1:
        raise ValueError('Expected exactly one accepted subject/trial')
    i=selection[0]
    np.savez(a.output,ppg=arrays['ppg_raw'][i],eda=arrays['eda_raw'][i])
    save_json(str(a.output)+'.json',meta.iloc[i].to_dict())

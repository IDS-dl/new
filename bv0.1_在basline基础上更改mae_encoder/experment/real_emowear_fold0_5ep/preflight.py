from pathlib import Path
import json
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from audit import run
from dataloder.splits import grouped_folds

out = Path(__file__).resolve().parent / 'results'
data = Path('../../datasets/EmoWear/EmoWear_development').resolve()
configs = [yaml.safe_load(Path(f'configs/emowear_{name}.yaml').read_text()) for name in ('mae','mae_coral','e0')]
cfg = configs[0]
for key in ('window_policy','window_seconds','quality','preprocess','ppg_normalization','eda_normalization','seed','num_folds','val_fraction','exclude_neutral','modalities','ppg_sampling_rate','eda_sampling_rate'):
    assert all(c[key] == cfg[key] for c in configs), key
for key in ('epochs','patience','batch_size','min_delta','weight_decay'):
    assert configs[1][key] == configs[2][key], key
arrays, meta, audit = run(data, cfg, out/'audit')
assert not audit['synthetic'] and len(meta) and not meta.duplicated(['subject_id','trial_id']).any()
assert all(arrays[n].shape == (len(meta), rate*30) and np.isfinite(arrays[n]).all() for n,rate in [('ppg',64),('eda',4)])
locked = {p.name for p in (data.parent/'EmoWear_locked_v2').iterdir() if p.is_dir()}
assert not locked.intersection(meta.subject_id)
folds = []
for fold, split in enumerate(grouped_folds(meta,cfg['num_folds'],cfg['seed'],cfg['val_fraction'])):
    groups = [set(meta.iloc[idx].subject_id) for idx in split]
    assert not any(groups[i]&groups[j] for i,j in [(0,1),(1,2),(0,2)])
    for role, idx, subjects in zip(('train','validation','test'), split, groups):
        folds.append(dict(fold=fold, role=role, subjects=len(subjects), windows=len(idx), ids=' '.join(sorted(subjects))))
pd.DataFrame(folds).to_csv(out/'folds.csv',index=False)
rows=[]
fig, axes=plt.subplots(2,2,figsize=(10,6))
for i,dim in enumerate(('valence','arousal')):
    for j,kind in enumerate(('raw','ord')):
        y=meta[f'{dim}_{kind}']
        rows.extend(dict(dimension=dim,scale=kind,value=value,count=n) for value,n in y.value_counts().sort_index().items())
        axes[i,j].hist(y,bins=np.arange(.5,10) if j else 32)
        axes[i,j].set_title(f'{dim} ({kind})')
fig.tight_layout()
fig.savefig(out/'labels.png')
pd.DataFrame(rows).to_csv(out/'labels.csv',index=False)
records=json.loads((out/'audit/quality.json').read_text())
qrows=[]
for name,rate in [('ppg',64),('eda',4)]:
    q=[r[name+'_quality'] for r in records if name+'_quality' in r]
    qrows.append(dict(modality=name,windows_inspected=len(q),no_quality_record=len(records)-len(q),short_count=sum(x['n']<30*rate for x in q),nonfinite_samples=sum(x['nonfinite'] for x in q),gap_windows=sum(x['timestamp_gaps']>0 for x in q),long_gap_windows=sum((x['max_gap_seconds'] or 0)>cfg['quality']['max_gap_seconds']+1.1/rate for x in q),flat=sum('flat' in x['flags'] for x in q),abrupt_jump=sum('abrupt_jump' in x['flags'] for x in q),negative=sum('negative_conductance' in x['flags'] for x in q)))
pd.DataFrame(qrows).to_csv(out/'signal_quality.csv',index=False)
print(len(records),len(meta),meta.subject_id.nunique(),audit['rejected_reasons'],flush=True)

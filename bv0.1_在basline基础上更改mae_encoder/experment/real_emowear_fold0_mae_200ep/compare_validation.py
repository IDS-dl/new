"""Run from code/ with PYTHONPATH=. after training."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from dataloder.dataset import build_windows, EmoWearDataset
from dataloder.preprocess import FoldNormalizer
from model.mae import WaveformMAE
from model.mae_encoder import patchify
from mae_training import reconstruction_plot

root = Path(__file__).resolve().parent
old = root.parent / 'real_emowear_fold0_5ep' / 'A'
histories = [pd.read_csv(folder/'results/fold_0/history.csv') for folder in (old,root)]
short, long = histories
columns = short.columns.drop('seconds')
np.testing.assert_allclose(short[columns], long.iloc[:5][columns], rtol=1e-5, atol=1e-6)
checkpoints = [torch.load(folder/'weights/fold_0/best_mae.pt',map_location='cpu',weights_only=False)
               for folder in (old,root)]
a,b = checkpoints
assert a['identity'] == b['identity']
assert {k for k in a['config'] if a['config'][k] != b['config'][k]} == {'epochs','output_dir'}
assert b['epoch'] == int(long.loc[long.val_loss.idxmin(),'epoch'])
assert len(long)==200 or len(long)-b['epoch']==b['config']['patience']
events = EventAccumulator(str(root/'results/tensorboard/fold_0/seed_42')).Reload()
for phase in ('train','val'):
    np.testing.assert_allclose([e.value for e in events.Scalars('loss/'+phase)],long[phase+'_loss'],rtol=1e-6)
cfg = b['config']
torch.set_num_threads(cfg['cpu_threads'])
arrays, meta, *_ = build_windows(cfg['data_root'],cfg)
val = EmoWearDataset(arrays,meta,b['identity']['split']['val']['indices'],FoldNormalizer.from_state(b['normalizer']))
rows=[]
for ck in checkpoints:
    model = WaveformMAE(cfg).eval()
    model.load_state_dict(ck['model'])
    generator = torch.Generator().manual_seed(cfg['seed'])
    with torch.no_grad():
        for batch in DataLoader(val,batch_size=cfg['batch_size']):
            output,masks = model(batch['ppg'],batch['eda'],generator)
            for name,rate in [('ppg',64),('eda',4)]:
                for p,t,m in zip(output[name],patchify(batch[name],rate),masks[name]):
                    fill=t[~m].mean()
                    x,y=p[m],t[m]
                    rows.append(dict(epoch=ck['epoch'],signal=name,
                        huber=F.huber_loss(x,y).item(),visible_mean=F.huber_loss(fill.expand_as(y),y).item(),
                        std_ratio=(x.std()/y.std().clamp_min(1e-8)).item()))
    if ck is b:
        for i in val.meta.drop_duplicates('subject_id').index[:3]:
            reconstruction_plot(model,[{k:v.unsqueeze(0) for k,v in val[i].items()}],'cpu',42+i,
                                root/'results'/f'reconstruction_val_{i}.png')
summary=pd.DataFrame(rows).groupby(['epoch','signal']).agg(
    masked_huber=('huber','mean'),visible_mean_huber=('visible_mean','mean'),
    median_std_ratio=('std_ratio','median'),windows=('huber','size'))
for ck,h in zip(checkpoints,histories):
    for signal in ('ppg','eda'):
        np.testing.assert_allclose(summary.loc[(ck['epoch'],signal),'masked_huber'],
            h.loc[h.epoch==ck['epoch'],'val_'+signal].item(),rtol=1e-5)
summary.to_csv(root/'results/comparison.csv')
print(summary.to_string())
print(long.loc[[0,4,long.val_loss.idxmin(),long.index[-1]]].to_string(index=False))

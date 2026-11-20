"""Run from code/ with PYTHONPATH=. after all three pilots complete."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from dataloder.dataset import build_windows, EmoWearDataset
from dataloder.preprocess import FoldNormalizer
from model.mae import WaveformMAE
from model.mae_encoder import patchify
from model import model_from_config
from loss.coral import coral_loss
from mae_training import reconstruction_plot, build_stage

root=Path(__file__).resolve().parent
out=root/'results'
torch.set_num_threads(4)
ck={}
for stage in 'ABC':
    path=root/stage
    ck[stage]=torch.load(next((path/'weights/fold_0').glob('best*.pt')),map_location='cpu',weights_only=False)
    history=pd.read_csv(path/'results/fold_0/history.csv')
    assert len(history)==5 and np.isfinite(history.to_numpy()).all()
    event=EventAccumulator(str(path/'results/tensorboard/fold_0/seed_42')).Reload()
    for phase in ('train','val'):
        values=event.Scalars('loss/'+phase)
        np.testing.assert_allclose([v.value for v in values],history[phase+'_loss'],rtol=1e-6)
    assert ck[stage]['identity']==ck['A']['identity']
cfg=ck['A']['config']
config_b,config_c=ck['B']['config'],ck['C']['config']
assert {k for k in config_b if config_b[k]!=config_c[k]} == {'pretrained_root','output_dir'}
loaded,_=build_stage(config_c,0,ck['A']['identity'],'cpu')
for k,v in loaded.encoder.state_dict().items():
    torch.testing.assert_close(v,ck['A']['encoder'][k],atol=0,rtol=0)
arrays, meta,*_=build_windows(cfg['data_root'],cfg)
split=ck['A']['identity']['split']
norm=FoldNormalizer(cfg['ppg_normalization'],cfg['eda_normalization']).fit(arrays,split['train']['indices'])
assert norm.state_dict()==ck['A']['normalizer']
val=EmoWearDataset(arrays,meta,split['val']['indices'],norm)
mae=WaveformMAE(cfg)
mae.load_state_dict(ck['A']['model'])
mae.eval()
for i in val.meta.drop_duplicates('subject_id').index[:3]:
    reconstruction_plot(mae,[{k:v.unsqueeze(0) for k,v in val[i].items()}],'cpu',42+i,out/f'reconstruction_val_{i}.png')
rows=[]
generator=torch.Generator().manual_seed(42)
with torch.no_grad():
    for batch in DataLoader(val,batch_size=cfg['batch_size']):
        pred,masks=mae(batch['ppg'],batch['eda'],generator=generator)
        for name,rate in [('ppg',64),('eda',4)]:
            target=patchify(batch[name],rate)
            for p,t,m in zip(pred[name],target,masks[name]):
                visible_level=t[~m].mean()
                p,t=p[m],t[m]
                rows.append(dict(modality=name,recon_std=p.std().item(),true_std=t.std().item(),
                    centered_rms_ratio=(p.std() / t.std().clamp_min(1e-8)).item(),
                    visible_mean_huber=F.huber_loss(visible_level.expand_as(t),t).item(),
                    model_huber=F.huber_loss(p,t).item(),zero_huber=F.huber_loss(torch.zeros_like(t),t).item()))
wave=pd.DataFrame(rows)
wave.to_csv(out/'masked_waveforms.csv',index=False)
print(wave.groupby('modality').median(),wave.groupby('modality')[['model_huber','zero_huber']].mean())
train=EmoWearDataset(arrays,meta,split['train']['indices'],norm)
batch=next(iter(DataLoader(train,batch_size=32,shuffle=True,generator=torch.Generator().manual_seed(42))))
fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True)
scores=[]
gradients=[]
for i,stage in enumerate(('B','C')):
    model=model_from_config(ck[stage]['config'])
    model.load_state_dict(ck[stage]['model'])
    model.eval()
    preds=pd.read_csv(root/stage/'results/fold_0/predictions.csv')
    stats=json.loads((root/stage/'results/fold_0/metrics.json').read_text())
    for j,dim in enumerate(('valence','arousal')):
        model.zero_grad()
        logits=model(batch['ppg'],batch['eda'])[dim+'_logits']
        coral_loss(logits,batch['ordinal'][:,j]).backward()
        def grad(module):
            return sum(p.grad.square().sum().item() for p in module.parameters() if p.grad is not None)**.5
        gradients.append(dict(stage=stage,dimension=dim,head=grad(model.heads[dim]),encoder=grad(model.encoder)))
        levels=np.arange(1,10)
        predicted=preds['pred_'+dim]
        for col,shift,label in [(dim+'_ord',-.2,'truth'),('pred_'+dim,.2,'prediction')]:
            heights=preds[col].value_counts().reindex(levels,fill_value=0)
            axes[i,j].bar(levels+shift,heights,width=.4,label=label)
        axes[i,j].set(title=f'{stage}: {dim}',xticks=levels,ylabel='Test windows')
        axes[i,j].legend()
        for view in ('vs_raw','vs_ordinal','expected_vs_raw'):
            scores.append(dict(stage=stage,dimension=dim,view=view,**stats[dim][view],distinct=predicted.nunique(),fraction_4to6=predicted.between(4,6).mean(),counts=str(predicted.value_counts().sort_index().to_dict())))
fig.tight_layout()
fig.savefig(out/'predicted_distributions.png')
pd.DataFrame(gradients).to_csv(out/'gradient_norms.csv',index=False)
pd.DataFrame(scores).to_csv(out/'scores.csv',index=False)
print(pd.DataFrame(gradients))

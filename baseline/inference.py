"""Inference from one saved trial, with the checkpoint's preprocessing rules."""
import argparse
from pathlib import Path
import numpy as np
import torch
from .preprocess import FoldNormalizer,filter_wave
from .train import model_from_config,decode
from .audit import save_json


def load_checkpoint(path,device='cpu'):
    c=torch.load(path,map_location=device,weights_only=False)
    model=model_from_config(c['config']).to(device)
    model.load_state_dict(c['model']);model.eval()
    return model,FoldNormalizer.from_state(c['normalizer']),c


@torch.no_grad()
def infer(path,ppg,eda,filtered=False,device='cpu'):
    model,norm,c=load_checkpoint(path,device)
    waves={}
    for name,x,fs in [('ppg',ppg,64),('eda',eda,4)]:
        x=np.asarray(x,np.float32).reshape(-1)
        n=fs*c['config']['window_seconds']
        if x.size!=n or not np.isfinite(x).all():
            raise ValueError(f'{name} requires {n} finite samples, aligned to the same trial')
        if not filtered:
            x=filter_wave(x,name,c['config'].get('preprocess',{}))
        waves[name]=torch.from_numpy(norm.transform(name,x)[None,None]).to(device)
    out=decode(model(**waves),c['config']['task'])
    return {k:(int(v[0]) if c['config']['task'] in ('binary','coral','softmax') and k in ('pred_valence','pred_arousal') else float(v[0])) for k,v in out.items()}

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--sample',required=True,help='npz with timestamp-aligned raw ppg and eda arrays')
    p.add_argument('--filtered',action='store_true')
    p.add_argument('--output',default='prediction.json')
    a=p.parse_args()
    with np.load(a.sample) as s:
        pred=infer(a.checkpoint,s['ppg'],s['eda'],a.filtered)
    save_json(a.output,pred);print(pred)

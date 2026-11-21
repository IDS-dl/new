"""Audit a completed experiment and archive deterministic validation masks."""
import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import yaml

from model.moment_wrapper import RATES, segment, stitch, observation_mask
from dataloder.preprocess import filter_wave


def verify(experiment):
    root=experiment/'results'
    cfg=yaml.safe_load((root/'config.yaml').read_text())
    torch.set_num_threads(cfg["cpu_threads"])
    split=json.loads((root/'splits.json').read_text())
    counts={}
    source=experiment/'weights/official/model.safetensors'
    official=load_file(str(source)) if source.exists() else None
    changes={}
    for folder in sorted(root.glob('smoke_*'))+sorted(root.glob('tune_*')):
        summary=json.loads((folder/'training.json').read_text())
        history=pd.read_csv(folder/'loss_history.csv')
        assert np.isfinite(history.to_numpy()).all()
        assert int(history.loc[history.val.idxmin(),'epoch'])==summary['best_epoch']
        assert len(history)==summary['epochs']
        paths=list((experiment/'weights'/folder.name).glob('best*.pt'))
        assert len(paths)==1
        ckpt=torch.load(paths[0],map_location='cpu',weights_only=True)
        assert ckpt['epoch']==summary['best_epoch'] and ckpt['val']==summary['best_val']
        if official is not None:
            changed=[key for key,tensor in ckpt['model'].items() if not torch.equal(tensor,official[key])]
            assert any(key.startswith('head.') for key in changed)
            if ckpt['mode']=='head':
                assert all(key.startswith('head.') for key in changed)
            else:
                assert any(key.startswith('encoder.block.') for key in changed)
                assert any(key.startswith('patch_embedding.') for key in changed)
            changes[folder.name]=changed
        logs=root/'tensorboard'/folder.name/f"fold_{cfg['fold']}"/f"seed_{cfg['seed']}"
        tb=EventAccumulator(str(logs),size_guidance={'scalars':0}).Reload()
        for phase in ('train','val'):
            np.testing.assert_allclose([e.value for e in tb.Scalars('loss/'+phase)],history[phase],rtol=1e-6)
        assert all(e.value==0 for e in tb.Scalars('anomaly/nonfinite_gradient'))
        counts[folder.name]=summary
    assert len(counts)==3+6*len(cfg['ratios'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for ratio in cfg['ratios']:
        fig, axes=plt.subplots(2,3,figsize=(12,6),layout='constrained')
        for i,mode in enumerate(('full','head')):
            for j,method in enumerate(('shared','ppg','eda')):
                name=f'tune_{ratio:g}_{mode}_{method}'
                history=pd.read_csv(root/name/'loss_history.csv')
                ax=axes[i,j]
                for phase in ('train','val'):
                    ax.plot(history.epoch,history[phase],label=phase)
                ax.axvline(counts[name]['best_epoch'],ls=':',color='grey',label='best val')
                ax.set(xlabel='Epoch',ylabel='Masked MSE',title=f'{mode} / {method}',yscale='log')
                ax.legend()
        fig.savefig(root/f'loss_mask{ratio:g}.png',dpi=130)
        plt.close(fig)
    device=torch.device(cfg['resolved_device'])
    samples=pd.read_csv(root/'audit/samples.csv').iloc[split['val']['indices']]
    for ratio in cfg['ratios']:
        for offset,(name,rate) in enumerate(RATES.items()):
            rng=torch.Generator(device=device).manual_seed(cfg['seed']+100+offset)
            masks=[]
            length=cfg['window_seconds']*rate
            for start in range(0,len(samples),cfg['batch_size']):
                size=min(cfg['batch_size'],len(samples)-start)
                _,valid=segment(torch.zeros(size,1,length,device=device))
                observed=observation_mask(valid,ratio,rng)
                masks.append(~stitch(observed,size,length).cpu().numpy())
            np.savez_compressed(root/f'validation_masks_{name}_{ratio:g}.npz',
                                hidden=np.concatenate(masks),subject=samples.subject_id.to_numpy(dtype=str),
                                trial=samples.trial_id.to_numpy(dtype=str))
    # Change one raw hidden patch and measure its effect on prefiltered visible points.
    leakage={}
    for name,rate in RATES.items():
        t=np.arange(30*rate)/rate
        original=(2+np.sin(2*np.pi*.8*t)).astype(np.float32)
        changed=original.copy()
        hidden=slice(14*rate,14*rate+8)
        changed[hidden]+=1
        visible=np.ones(len(t),dtype=bool)
        visible[hidden]=False
        difference=filter_wave(changed,name,cfg['preprocess'])-filter_wave(original,name,cfg['preprocess'])
        leakage[name]=float(np.max(np.abs(difference[visible])))
    (root/'verified_runs.json').write_text(json.dumps(
        dict(runs=counts,changed_checkpoint_tensors=changes,synthetic_raw_perturbation_max_visible_delta=leakage),indent=2)+'\n')
    print(f'Verified {len(counts)} histories, TensorBoard runs and checkpoints. Filter leakage {leakage}')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('experiment',type=Path)
    verify(parser.parse_args().experiment)

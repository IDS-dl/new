"""Final test evaluation with identical masks for all methods."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import yaml

from audit import save_json
from metrics.reconstruction import fill_baselines, measures
from model.moment_loader import load_moment, REVISION
from model.moment_wrapper import RATES, reconstruct, stitch
from train_moment import data, generator, modality_batch


@torch.no_grad()
def predict_batches(model, dataset, names, cfg, ratio, device, phase='test'):
    model.eval()
    rng = {name:generator(device,cfg['seed']+(100 if phase=='val' else 200)+i) for i,name in enumerate(RATES)}
    results = {name:[] for name in names}
    for batch in DataLoader(dataset, batch_size=cfg['batch_size']):
        for name in names:
            waves = batch[name].to(device)
            x,valid,mask = modality_batch(waves,ratio,rng[name])
            y = reconstruct(model,x,valid,mask)
            assert torch.isfinite(y[valid[:,None]]).all()
            size,_,length = waves.shape
            results[name].append([stitch(t,size,length).cpu().numpy() for t in (x,mask,y)])
    return {n:tuple(np.concatenate(parts) for parts in zip(*batches)) for n,batches in results.items()}


def evaluate(experiment, source, dataset, normalizer, cfg, device, root=None):
    from visualize_reconstruction import plot_examples
    root=experiment/'results' if root is None else root
    rows, boundaries, shapes=[],[],[]
    for ratio in cfg['ratios']:
        model=load_moment(source,'zero').to(device)
        signals=predict_batches(model,dataset,tuple(RATES),cfg,ratio,device)
        methods={n:{'zero-shot':entry[2]} for n,entry in signals.items()}
        del model
        for mode in ('full','head'):
            for method,names in [('shared',tuple(RATES)),('ppg',('ppg',)),('eda',('eda',))]:
                path=experiment/'weights'/f'tune_{ratio:g}_{mode}_{method}'/f'best_moment_{method}.pt'
                ckpt=torch.load(path,map_location='cpu',weights_only=True)
                assert (ckpt['ratio'],ckpt['mode'],ckpt['method'],ckpt['revision'])==(ratio,mode,method,REVISION)
                model=load_moment(source,'zero').to(device)
                model.load_state_dict(ckpt['model'],strict=True)
                for name,(target,observed,pred) in predict_batches(model,dataset,names,cfg,ratio,device).items():
                    np.testing.assert_array_equal(observed,signals[name][1])
                    methods[name][f'{mode} '+('shared' if method=='shared' else 'independent')]=pred
                del model,ckpt
        for name,(target,observed,_) in signals.items():
            mean,linear=zip(*(fill_baselines(x,o) for x,o in zip(target,observed)))
            methods[name].update(mean=np.asarray(mean),linear=np.asarray(linear))
            np.savez_compressed(root/f'masks_{name}_{ratio:g}.npz', hidden=~observed,
                                subject=dataset.meta.subject_id.to_numpy(dtype=str),
                                trial=dataset.meta.trial_id.to_numpy(dtype=str))
            hidden=~observed
            center,scale=normalizer.stats[name]
            def inverse(x):
                x=x.astype(np.float64)*scale+center
                return np.expm1(x) if name=='eda' and normalizer.eda=='log_robust' else x
            for label,pred in methods[name].items():
                metric=measures(target[hidden],pred[hidden])
                metric['physical_rmse']=measures(inverse(target[hidden]),inverse(pred[hidden]))['rmse']
                rows.append(dict(ratio=ratio,actual_ratio=hidden.mean(),signal=name,method=label,**metric))
                ratios, correlations = [], []
                for x,y in zip(pred[hidden].reshape(len(pred),-1),target[hidden].reshape(len(target),-1)):
                    target_std = np.std(y, dtype=np.float64)
                    pred_std = np.std(x, dtype=np.float64)
                    if target_std > 0:
                        ratios.append(pred_std / target_std)
                        if pred_std > 0:
                            correlations.append(np.corrcoef(x,y)[0,1])
                shapes.append(dict(ratio=ratio,signal=name,method=label,
                    median_std_ratio=np.median(ratios),median_within_trial_pearson=np.median(correlations) if correlations else None))
                if name=='ppg':
                    filled=np.where(observed,target,pred)
                    points=np.arange(512,filled.shape[1],512)
                    affected=hidden[:,points] | hidden[:,points-1]
                    delta=(filled[:,points]-filled[:,points-1])[affected]
                    true=(target[:,points]-target[:,points-1])[affected]
                    boundaries.append(dict(ratio=ratio,method=label,count=len(delta),mean_abs_jump=np.abs(delta).mean(),
                                           target_mean_abs_jump=np.abs(true).mean(),step_error_mae=np.abs(delta-true).mean()))
            plot_examples(root/'reconstruction_examples',name,ratio,dataset.meta,target,observed,methods[name])
    table=pd.DataFrame(rows)
    table.to_csv(root/'reconstruction_metrics.csv',index=False)
    table.pivot(index=['ratio','method'],columns='signal',values='rmse').to_csv(root/'comparison.csv')
    pd.DataFrame(boundaries).to_csv(root/'ppg_boundaries.csv',index=False)
    pd.DataFrame(shapes).to_csv(root/'waveform_shape.csv',index=False)
    return table


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('experiment',type=Path)
    parser.add_argument('--official',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    cfg=yaml.safe_load((args.experiment/'results/config.yaml').read_text())
    torch.set_num_threads(cfg['cpu_threads'])
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'README.md').write_text(f'# MOMENT test evaluation\n\nCheckpoints from {args.experiment}.\n')
    root=args.output/'results'
    root.mkdir()
    datasets,norm=data(cfg,root)
    for name in ('splits.json','normalizer.json'):
        assert (root/name).read_bytes()==(args.experiment/'results'/name).read_bytes(), name
    assert (root/'audit/sources.json').read_bytes()==(args.experiment/'results/audit/sources.json').read_bytes()
    evaluate(args.experiment,args.official,datasets[2],norm,cfg,torch.device(cfg['resolved_device']),root=root)

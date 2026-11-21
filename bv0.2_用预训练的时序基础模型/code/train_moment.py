"""Run official MOMENT Stage 1 on a subject-wise EmoWear fold."""
import argparse
import hashlib
import time
from pathlib import Path
from importlib.metadata import version

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from audit import run as audit, save_json
from dataloder.dataset import EmoWearDataset
from dataloder.preprocess import FoldNormalizer
from dataloder.splits import grouped_folds
from loss.reconstruction import masked_mse
from model.moment_loader import REPO, REVISION, load_moment
from model.moment_wrapper import RATES, segment, observation_mask, reconstruct

METHODS = {'shared': tuple(RATES), 'ppg': ('ppg',), 'eda': ('eda',)}


def generator(device, seed):
    return torch.Generator(device=device).manual_seed(seed)


def modality_batch(waves, ratio, rng):
    signal, valid = segment(waves)
    return signal, valid, observation_mask(valid, ratio, rng)


def train_one(model, datasets, cfg, ratio, method, mode, output, device, epochs, patience):
    output.mkdir()
    checkpoint = output.parent.parent/'weights'/output.name/f'best_moment_{method}.pt'
    checkpoint.parent.mkdir(parents=True)
    train, val = datasets[:2]
    loaders = [DataLoader(ds, batch_size=cfg['batch_size'], shuffle=i==0,
                         generator=generator('cpu', cfg['seed'])) for i,ds in enumerate((train,val))]
    groups = [
        dict(params=[p for n,p in model.named_parameters() if n.startswith('head.') and p.requires_grad],
             lr=cfg['head_lr']),
        dict(params=[p for n,p in model.named_parameters() if not n.startswith('head.') and p.requires_grad],
             lr=cfg['backbone_lr']),
    ]
    optimizer = torch.optim.AdamW([g for g in groups if g['params']], weight_decay=cfg['weight_decay'])
    names = METHODS[method]
    rng = {name:generator(device,cfg['seed']+i) for i,name in enumerate(RATES)}
    best, history = float('inf'), []
    started = time.perf_counter()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    log = output.parent/'tensorboard'/output.name/f"fold_{cfg['fold']}"/f"seed_{cfg['seed']}"
    with SummaryWriter(log_dir=str(log)) as writer:
        for epoch in range(1,epochs+1):
            row = dict(epoch=epoch)
            for phase,loader in zip(('train','val'),loaders):
                training = phase=='train'
                model.train(training)
                if mode=='head':
                    model.encoder.eval()
                    model.patch_embedding.eval()
                masks = rng if training else {n:generator(device,cfg['seed']+100+i) for i,n in enumerate(RATES)}
                totals = {name:0. for name in names}
                with torch.set_grad_enabled(training):
                    for batch in loader:
                        if training:
                            optimizer.zero_grad(set_to_none=True)
                        for name in names:
                            inputs = modality_batch(batch[name].to(device),ratio,masks[name])
                            with torch.autocast('cpu',enabled=cfg['cpu_bfloat16'] and device.type=='cpu',dtype=torch.bfloat16):
                                pred = reconstruct(model,*inputs)
                            loss = masked_mse(pred,*inputs)
                            if not torch.isfinite(loss):
                                writer.add_scalar('anomaly/nonfinite_loss',1,epoch)
                                raise FloatingPointError(f'{name} non-finite {phase} loss')
                            totals[name] += float(loss.detach()) * len(batch[name])
                            if training:
                                (loss/len(names)).backward()
                        if training:
                            grad = torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'])
                            if not torch.isfinite(grad):
                                writer.add_scalar('anomaly/nonfinite_gradient',1,epoch)
                                raise FloatingPointError('Non-finite gradient')
                            optimizer.step()
                for name,value in totals.items():
                    row[f'{phase}_{name}'] = value/len(loader.dataset)
                    writer.add_scalar(f'loss/{name}' if training else f'loss/{name}_val',
                                      row[f'{phase}_{name}'],epoch)
                row[phase] = sum(totals.values())/(len(loader.dataset)*len(names))
                writer.add_scalar(f'loss/{phase}',row[phase],epoch)
            for i,g in enumerate(optimizer.param_groups):
                writer.add_scalar(f'lr/group_{i}',g['lr'],epoch)
            writer.add_scalar('anomaly/nonfinite_gradient',0,epoch)
            writer.flush()
            history.append(row)
            pd.DataFrame(history).to_csv(output/'loss_history.csv',index=False)
            if row['val'] < best:
                best, best_epoch = row['val'], epoch
                torch.save(dict(model=model.state_dict(),epoch=epoch,val=best,revision=REVISION,
                                ratio=ratio,mode=mode,method=method),checkpoint)
            print(output.name, epoch, f"train={row['train']:.7g} val={row['val']:.7g}", flush=True)
            if patience is not None and epoch-best_epoch >= patience:
                break
    if patience is None:
        torch.save(dict(model=model.state_dict(),epoch=epoch,val=row['val'],revision=REVISION,
                        ratio=ratio,mode=mode,method=method),
                   checkpoint.with_name(f'last_moment_{method}.pt'))
    save_json(output/'training.json',dict(epochs=epoch,best_epoch=best_epoch,best_val=best,
        seconds=time.perf_counter()-started,peak_cuda_bytes=torch.cuda.max_memory_allocated(device)
        if device.type=='cuda' else None))
    return checkpoint


def data(cfg, root):
    arrays, meta, _ = audit(cfg['data_root'],cfg,root/'audit')
    splits = list(grouped_folds(meta,cfg['num_folds'],cfg['seed'],cfg['val_fraction']))
    indices = splits[cfg['fold']]
    groups = [set(meta.iloc[ix].subject_id) for ix in indices]
    assert not any(groups[i] & groups[j] for i,j in ((0,1),(0,2),(1,2)))
    normalizer = FoldNormalizer(cfg['ppg_normalization'],cfg['eda_normalization']).fit(arrays,indices[0])
    save_json(root/'normalizer.json',normalizer.state_dict())
    save_json(root/'splits.json',{name:dict(subjects=sorted(g),indices=ix.tolist(),windows=len(ix))
                                 for name,g,ix in zip(('train','val','test'),groups,indices)})
    return [EmoWearDataset(arrays,meta,ix,normalizer) for ix in indices], normalizer


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/moment_stage1.yaml')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--official',type=Path)
    parser.add_argument('--full-epochs',type=int)
    parser.add_argument('--head-epochs',type=int)
    parser.add_argument('--ratios',type=float,nargs='+',default=[.3,.6])
    parser.add_argument('--threads',type=int)
    args=parser.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text())
    for key in ('full_epochs','head_epochs'):
        value=getattr(args,key)
        if value is not None:
            if value<1:
                parser.error(f'{key} must be positive')
            cfg[key]=value
    if args.threads:
        cfg['cpu_threads']=args.threads
    root=args.output/'results'
    args.output.mkdir(parents=True,exist_ok=False)
    root.mkdir()
    (args.output/'README.md').write_text('# MOMENT Stage 1\n\nFold 0 EmoWear reconstruction. See results/config.yaml.\n\nTensorBoard: `tensorboard --logdir results/tensorboard`.\n')
    cfg['ratios']=args.ratios
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu') if cfg['device']=='auto' else torch.device(cfg['device'])
    cfg['resolved_device']=str(device)
    (root/'config.yaml').write_text(yaml.safe_dump(cfg))
    torch.set_num_threads(cfg['cpu_threads'])
    datasets, normalizer=data(cfg,root)
    model=load_moment(args.official,'zero')
    report=dict(repo=REPO,revision=REVISION,official_config=vars(model.config),
                packages={p:version(p) for p in ('momentfm','torch','transformers','huggingface-hub')},
                strict=True,missing=[],unexpected=[],shape_mismatches=[],
                parameters=sum(p.numel() for p in model.parameters()))
    if args.official:
        report['sha256']=hashlib.sha256((args.official/'model.safetensors').read_bytes()).hexdigest()
    for mode in ('full','head','zero'):
        report[mode]={name:p.numel() for name,p in model.named_parameters()
                      if mode=='full' or (mode=='head' and name.startswith('head.'))}
    save_json(root/'model_loading_report.json',report)
    from evaluate_moment import evaluate, predict_batches
    for ratio in args.ratios:
        predict_batches(model.to(device),datasets[1],tuple(RATES),cfg,ratio,device,phase='val')
    del model
    for stage,ratio,mode in [('smoke',args.ratios[0],'full')]+[
            ('tune',ratio,mode) for ratio in args.ratios for mode in ('full','head')]:
        for method in METHODS:
            torch.manual_seed(cfg['seed'])
            model=load_moment(args.official,mode).to(device)
            name=f'{stage}_{ratio:g}_{mode}_{method}'
            train_one(model,datasets,cfg,ratio,method,mode,root/name,device,
                      1 if stage=='smoke' else cfg[f'{mode}_epochs'],cfg[f'{mode}_patience'])
            del model
    evaluate(args.output,args.official,datasets[2],normalizer,cfg,device)
    from verify_experiment import verify
    verify(args.output)


if __name__=='__main__':
    main()

import argparse
from importlib.metadata import version
import platform
import random
import shutil
import subprocess
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml
from audit import run as audit_run, save_json, distributions
from dataloder.dataset import make_loaders
from dataloder.preprocess import FoldNormalizer
from dataloder.splits import grouped_folds
from loss.tasks import loss_fn
from metrics import metrics
from metrics.decoding import decode
from metrics.reporting import baseline_predictions, plot_history, summarize
from model import model_from_config


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True,warn_only=True)


def epoch_pass(model,loader,device,task,opt=None,regression_loss='huber'):
    model.train(opt is not None)
    total,count=0.,0
    with torch.set_grad_enabled(opt is not None):
        for batch in loader:
            batch={k:v.to(device) for k,v in batch.items()}
            if opt:
                opt.zero_grad(set_to_none=True)
            out=model(batch['ppg'],batch['eda'])
            loss=loss_fn(out,batch,task,regression_loss)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss')
            if opt:
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            n=len(batch['ppg']);total+=loss.item()*n;count+=n
    return total/count


@torch.no_grad()
def predict(model,loader,device,task,meta):
    model.eval()
    chunks=[]
    for batch in loader:
        out=model(batch['ppg'].to(device),batch['eda'].to(device))
        chunks.append(pd.DataFrame(decode(out,task)))
    return pd.concat([meta.reset_index(drop=True),pd.concat(chunks,ignore_index=True)],axis=1)


def run_training(cfg):
    if cfg.get('ppg_sampling_rate',64)!=64 or cfg.get('eda_sampling_rate',4)!=4:
        raise ValueError('E4 native rates must remain 64/4 Hz')
    seed_all(cfg['seed'])
    torch.set_num_threads(cfg.get('cpu_threads',4))
    root=Path(cfg['output_dir'])
    if (root/'run.json').exists():
        raise FileExistsError(f'{root} contains a run; choose a new output_dir to preserve provenance')
    root.mkdir(parents=True,exist_ok=True)
    arrays,meta,audit=audit_run(cfg['data_root'],cfg,root/'audit')
    if meta.empty:
        raise ValueError('No accepted windows; see audit')
    code_root = Path(__file__).resolve().parent
    git_options = dict(cwd=code_root, text=True, stderr=subprocess.DEVNULL)
    try:
        sha=subprocess.check_output(['git','rev-parse','HEAD'], **git_options).strip()
        dirty=bool(subprocess.check_output(['git','status','--porcelain'], **git_options).strip())
        diff=subprocess.check_output(['git','diff'], **git_options)
    except (FileNotFoundError, subprocess.CalledProcessError):
        sha,dirty,diff='unknown',True,''
    (root/'tracked_changes.patch').write_text(diff)
    # Snapshot the scripts, modules, and configs for reproducing the run.
    snapshot = root / 'source'
    snapshot.mkdir(exist_ok=True)
    for source_file in code_root.glob('*.py'):
        shutil.copy2(source_file, snapshot / source_file.name)
    for folder in ('model', 'dataloder', 'loss', 'metrics', 'configs'):
        shutil.copytree(
            code_root / folder,
            snapshot / folder,
            ignore=shutil.ignore_patterns('__pycache__'),
            dirs_exist_ok=True,
        )
    (root/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    requested=cfg.get('device','auto')
    device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if requested=='auto' else requested)
    save_json(root/'run.json',dict(seed=cfg['seed'],git_commit=sha,dirty=dirty,device=str(device),
              synthetic=audit['synthetic'],python=platform.python_version(),torch=torch.__version__,
              numpy=np.__version__,pandas=pd.__version__,config=cfg,
              packages={name:version(name) for name in ('torch','numpy','pandas','scipy','scikit-learn','PyYAML','matplotlib')},
              interpretation='pilot only' if cfg.get('pilot',False) else 'configured experiment'))
    folds=list(grouped_folds(meta,cfg['num_folds'],cfg['seed'],cfg['val_fraction']))
    split_records=[]
    for fold,(tr,va,te) in enumerate(folds):
        split_records.append(dict(fold_id=fold,**{name:dict(subjects=sorted(meta.iloc[idx].subject_id.unique()),
                       indices=idx.tolist(),trials=len(meta.iloc[idx][['subject_id','trial_id']].drop_duplicates()),
                       windows=len(idx),labels=distributions(meta.iloc[idx])) for name,idx in [('train',tr),('val',va),('test',te)]}))
    save_json(root/'splits.json',split_records)
    for fold,(tr,va,te) in enumerate(folds):
        if cfg.get('fold_ids') is not None and fold not in cfg['fold_ids']:
            continue
        seed_all(cfg['seed']+fold)
        folder=root/f'fold_{fold}';folder.mkdir(exist_ok=True)
        norm=FoldNormalizer(cfg['ppg_normalization'],cfg['eda_normalization']).fit(arrays,tr)
        save_json(folder/'normalizer.json',norm.state_dict())
        loaders = make_loaders(arrays, meta, (tr, va, te), norm, cfg['batch_size'], cfg['seed'] + fold)
        model=model_from_config(cfg).to(device)
        opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
        best=float('inf');bad=0;history=[];begin=time.perf_counter()
        if device.type=='cuda':
            torch.cuda.reset_peak_memory_stats(device)
        for epoch in range(cfg['epochs']):
            t=time.perf_counter()
            train_loss=epoch_pass(model,loaders[0],device,cfg['task'],opt,cfg['regression_loss'])
            val_loss=epoch_pass(model,loaders[1],device,cfg['task'],regression_loss=cfg['regression_loss'])
            history.append(dict(epoch=epoch+1,train_loss=train_loss,val_loss=val_loss,seconds=time.perf_counter()-t))
            if val_loss<best-cfg.get('min_delta',0.):
                best=val_loss;bad=0
                torch.save(dict(model=model.state_dict(),config=cfg,normalizer=norm.state_dict(),fold_id=fold,
                                epoch=epoch+1,val_loss=val_loss,git_commit=sha,synthetic=audit['synthetic']),folder/'best.pt')
            else:
                bad+=1
            if (epoch+1)%cfg.get('log_every',5)==0 or epoch==0:
                print(f"fold={fold} epoch={epoch+1} train={train_loss:.5f} val={val_loss:.5f}",flush=True)
            if bad>=cfg['patience']:
                break
        checkpoint=torch.load(folder/'best.pt',map_location=device,weights_only=False)
        model.load_state_dict(checkpoint['model'])
        frame=predict(model,loaders[2],device,cfg['task'],meta.iloc[te])
        frame['fold_id']=fold;frame.to_csv(folder/'predictions.csv',index=False)
        base=baseline_predictions(meta.iloc[tr],meta.iloc[te],cfg['task'])
        base['fold_id']=fold;base.to_csv(folder/'baseline_predictions.csv',index=False)
        save_json(folder/'metrics.json',metrics(frame,cfg['task']))
        save_json(folder/'performance.json',dict(seconds=time.perf_counter()-begin,
                  parameters=sum(p.numel() for p in model.parameters()),best_epoch=checkpoint['epoch'],
                  peak_cuda_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0))
        h=pd.DataFrame(history);h.to_csv(folder/'history.csv',index=False);plot_history(h,folder/'loss.png')
        print(f'fold={fold} saved test predictions ({len(frame)} windows)',flush=True)
    summarize(root,cfg['task'],cfg['num_folds'])
    return root


def load_config_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--data-root')
    parser.add_argument('--output-dir')
    parser.add_argument('--set',action='append',default=[],metavar='KEY=YAML_VALUE')
    a=parser.parse_args()
    cfg=yaml.safe_load(Path(a.config).read_text())
    for item in a.set:
        k,v=item.split('=',1)
        if k not in cfg:
            raise ValueError(f'Unknown config key {k}')
        cfg[k]=yaml.safe_load(v)
    if a.data_root:
        cfg['data_root']=a.data_root
    if a.output_dir:
        cfg['output_dir']=a.output_dir
    return cfg

if __name__=='__main__':
    run_training(load_config_args())

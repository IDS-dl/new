import argparse
import json
import os
import platform
import random
import subprocess
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import yaml
from .audit import run as audit_run, save_json, distributions
from .dataset import EmoWearDataset
from .labels import ordinal
from .preprocess import FoldNormalizer
from .splits import grouped_folds
from .models import EmoWearDualConformer
from .models.coral import coral_loss,coral_predict
from .metrics import metrics,flatten_metrics


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True,warn_only=True)


def model_from_config(c):
    return EmoWearDualConformer(**{k:c[k] for k in ('task','modalities','hidden_dim','depth','heads','dropout')})


def loss_fn(out,batch,task,regression_loss='huber'):
    loss=0
    for i,dim in enumerate(('valence','arousal')):
        logits=out[dim+'_logits']
        if task in ('binary','softmax'):
            y=batch['binary'][:,i] if task=='binary' else batch['ordinal'][:,i]-1
            loss=loss+F.cross_entropy(logits,y)
        elif task=='coral':
            loss=loss+coral_loss(logits,batch['ordinal'][:,i])
        else:
            fn={'huber':F.huber_loss,'mae':F.l1_loss,'mse':F.mse_loss}[regression_loss]
            loss=loss+fn(logits[:,0],batch['raw'][:,i])
    return loss


def decode(out,task):
    result={}
    for dim in ('valence','arousal'):
        x=out[dim+'_logits']
        if task=='coral':
            pred,expected=coral_predict(x)
            for k in range(8):
                result[f'{dim}_prob_gt_{k+1}']=x[:,k].sigmoid().cpu().numpy()
        elif task in ('binary','softmax'):
            prob=x.softmax(-1)
            pred=prob.argmax(-1)+(0 if task=='binary' else 1)
            levels=torch.arange(prob.shape[-1],device=x.device)+(0 if task=='binary' else 1)
            expected=(prob*levels).sum(-1)
            for k in range(prob.shape[-1]):
                result[f'{dim}_prob_{k if task=="binary" else k+1}']=prob[:,k].cpu().numpy()
        else:
            pred=expected=x[:,0]
        result['pred_'+dim]=pred.cpu().numpy()
        result['pred_'+dim+'_expected']=expected.cpu().numpy()
    return result


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


def baseline_predictions(train,test,task):
    out=test.copy()
    for dim in ('valence','arousal'):
        raw=train[dim+'_raw'].to_numpy()
        pred=int((raw>5).mean()>.5) if task=='binary' else float(np.median(raw))
        if task in ('coral','softmax'):
            pred=int(ordinal(pred))
        out['pred_'+dim]=pred
        out['pred_'+dim+'_expected']=float(raw.mean()) if task!='binary' else float((raw>5).mean())
    return out


def plot_history(history,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    f,ax=plt.subplots()
    ax.plot(history.epoch,history.train_loss,label='train')
    ax.plot(history.epoch,history.val_loss,label='validation')
    ax.set(xlabel='epoch',ylabel='loss');ax.legend();f.tight_layout();f.savefig(path);plt.close(f)


def summarize(root,task,expected_folds):
    run_file=Path(root)/'run.json'
    if run_file.exists():
        saved=json.loads(run_file.read_text())['config']
        if saved['task']!=task or saved['num_folds']!=expected_folds:
            raise ValueError('Evaluation config task/fold count differs from saved run')
    frames=[];metric_rows=[];baseline_rows=[]
    for folder in sorted(Path(root).glob('fold_*')):
        p=folder/'predictions.csv'
        if not p.exists():
            continue
        frame=pd.read_csv(p);frames.append(frame)
        save_json(folder/'metrics.json',metrics(frame,task))
        metric_rows.append({'fold_id':int(folder.name.split('_')[-1]),**flatten_metrics(metrics(frame,task))})
        baseline_rows.append({'fold_id':metric_rows[-1]['fold_id'],**flatten_metrics(metrics(pd.read_csv(folder/'baseline_predictions.csv'),task))})
    if not frames:
        raise ValueError('No test predictions available')
    pd.concat(frames,ignore_index=True).to_csv(Path(root)/'predictions.csv',index=False)
    for name,rows in [('metrics',metric_rows),('baseline_metrics',baseline_rows)]:
        table=pd.DataFrame(rows);table.to_csv(Path(root)/(name+'_per_fold.csv'),index=False)
        stats=table.drop(columns='fold_id').agg(['mean','std']).T
        stats.to_csv(Path(root)/(name+'_summary.csv'))
    save_json(Path(root)/'evaluation.json',dict(completed_folds=len(frames),expected_folds=expected_folds,
                 complete_cross_validation=len(frames)==expected_folds,task=task,
                 pooled=metrics(pd.concat(frames,ignore_index=True),task)))
    visualize_predictions(pd.concat(frames,ignore_index=True),task,Path(root))


def visualize_predictions(frame,task,root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for dim in ('valence','arousal'):
        fig,ax=plt.subplots()
        if task=='binary':
            matrix=metrics(frame,task)[dim]['confusion_matrix']
            ax.imshow(matrix,cmap='Blues')
            for i in range(2):
                for j in range(2):
                    ax.text(j,i,str(matrix[i][j]),ha='center',va='center')
            ax.set(xlabel='predicted',ylabel='true',xticks=[0,1],yticks=[0,1])
            wrong=frame[(frame['pred_'+dim].values!=(frame[dim+'_raw'].values>5))]
            wrong.to_csv(root/f'{dim}_errors.csv',index=False)
        else:
            ax.scatter(frame[dim+'_raw'],frame['pred_'+dim],alpha=.2,s=8)
            ax.plot([1,9],[1,9],'k--');ax.set(xlabel='raw self rating',ylabel='prediction',xlim=(.5,9.5),ylim=(.5,9.5))
        ax.set_title(dim);fig.tight_layout();fig.savefig(root/f'{dim}_predictions.png');plt.close(fig)


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
    try:
        sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip())
        diff=subprocess.check_output(['git','diff'],text=True)
    except subprocess.CalledProcessError:
        sha,dirty,diff='unknown',True,''
    (root/'tracked_changes.patch').write_text(diff)
    # Snapshot the package and configs for reproducing the run.
    import shutil
    package_dir = Path(__file__).resolve().parent
    snapshot = root / 'source' / package_dir.name
    snapshot.mkdir(parents=True, exist_ok=True)
    for source_file in package_dir.glob('*.py'):
        shutil.copy2(source_file, snapshot / source_file.name)
    for folder in ('models', 'configs'):
        shutil.copytree(
            package_dir / folder,
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
              packages={name:__import__('importlib.metadata',fromlist=['version']).version(name) for name in ('torch','numpy','pandas','scipy','scikit-learn','PyYAML','matplotlib','pytest')},
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
        datasets=[EmoWearDataset(arrays,meta,i,norm) for i in (tr,va,te)]
        generator=torch.Generator().manual_seed(cfg['seed']+fold)
        loaders=[DataLoader(d,batch_size=cfg['batch_size'],shuffle=i==0,num_workers=0,generator=generator) for i,d in enumerate(datasets)]
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

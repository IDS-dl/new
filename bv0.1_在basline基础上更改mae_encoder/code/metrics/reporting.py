import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit import save_json
from dataloder.labels import ordinal
from .scoring import flatten_metrics, metrics


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

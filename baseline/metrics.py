import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score,f1_score,classification_report,confusion_matrix


def errors(y,p):
    y,p=np.asarray(y,float),np.asarray(p,float)
    correlation=float(spearmanr(y,p).statistic) if np.ptp(y)>0 and np.ptp(p)>0 else None
    return dict(mae=float(np.abs(y-p).mean()),rmse=float(np.sqrt(np.mean((y-p)**2))),
                spearman=correlation,within_1=float(np.mean(np.abs(y-p)<=1)))


def metrics(frame,task):
    out={}
    for dim in ('valence','arousal'):
        pred=frame['pred_'+dim].to_numpy()
        if task=='binary':
            y=(frame[dim+'_raw'].to_numpy()>5).astype(int)
            out[dim]=dict(accuracy=accuracy_score(y,pred),macro_f1=f1_score(y,pred,labels=[0,1],average='macro',zero_division=0),
                          per_class=classification_report(y,pred,labels=[0,1],output_dict=True,zero_division=0),
                          confusion_matrix=confusion_matrix(y,pred,labels=[0,1]).tolist())
        else:
            out[dim]=dict(vs_raw=errors(frame[dim+'_raw'],pred),
                          vs_ordinal=errors(frame[dim+'_ord'],pred),
                          expected_vs_raw=errors(frame[dim+'_raw'],frame['pred_'+dim+'_expected']),
                          expected_vs_ordinal=errors(frame[dim+'_ord'],frame['pred_'+dim+'_expected']),
                          prediction_distribution={str(k):int(v) for k,v in zip(*np.unique(pred,return_counts=True))},
                          rounded_prediction_distribution={str(k):int(np.sum(np.clip(np.floor(pred+.5),1,9)==k)) for k in range(1,10)})
    return out


def flatten_metrics(obj,prefix=''):
    result={}
    for k,v in obj.items():
        name=f'{prefix}.{k}' if prefix else k
        if k in ('prediction_distribution', 'rounded_prediction_distribution'):
            continue
        if isinstance(v,dict):
            result.update(flatten_metrics(v,name))
        elif isinstance(v,(int,float)) or v is None:
            result[name]=v
    return result

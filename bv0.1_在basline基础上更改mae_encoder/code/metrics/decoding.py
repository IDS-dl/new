import torch


def coral_predict(logits):
    p=logits.sigmoid()
    return 1+(p>.5).sum(-1), 1+p.sum(-1)


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

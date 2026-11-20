import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from inference import infer
from dataloder.dataset import build_windows
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from mae_training import build_stage

def verify(root):
    root = Path(root)
    for stage in ('pre','e0','e1','e2'):
        for fold in range(5):
            folder = root/stage/'results'/'tensorboard'/f'fold_{fold}'/'seed_42'
            events = EventAccumulator(str(folder)).Reload()
            assert {'loss/train','loss/val'} <= set(events.Tags()['scalars'])
            assert events.Scalars('loss/train')[0].step == 1
    for stage in ('e0','e1','e2'):
        weights = root/stage/'weights/fold_0/best.pt'
        cfg=torch.load(weights, weights_only=False)['config']
        arrays,meta,*_=build_windows(root/'data', cfg)
        row=pd.read_csv(root/stage/'results/fold_0/predictions.csv').iloc[0]
        i=meta.index[(meta.subject_id==row.subject_id)&(meta.trial_id==row.trial_id)].item()
        pred=infer(weights,arrays['ppg_raw'][i],arrays['eda_raw'][i])
        for key,value in pred.items():
            np.testing.assert_allclose(value,row[key],atol=1e-5)
    try:
        build_stage(cfg,0,{'split':'incorrect'},'cpu')
    except ValueError as e:
        assert 'mismatch' in str(e)
    else:
        raise AssertionError('Mismatched split accepted')
    train_readme = root / "pre" / "README.md"
    assert train_readme.is_file()
    print('20 TensorBoard logs, E0/E1/E2 inference and mismatched split verified')


if __name__ == '__main__':
    verify(sys.argv[1])

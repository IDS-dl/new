import numpy as np
from sklearn.model_selection import GroupKFold, GroupShuffleSplit


def grouped_folds(meta, n_splits=5, seed=42, val_fraction=.2):
    groups=meta.subject_id.to_numpy()
    if len(np.unique(groups)) < n_splits:
        raise ValueError('Not enough subjects for GroupKFold')
    for trainval,test in GroupKFold(n_splits=n_splits).split(meta,groups=groups):
        train, val = next(GroupShuffleSplit(n_splits=1,test_size=val_fraction,random_state=seed).split(trainval,groups=groups[trainval]))
        train,val=trainval[train],trainval[val]
        sets=[set(groups[idx]) for idx in (train,val,test)]
        assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
        yield train,val,test

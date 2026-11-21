import numpy as np


def ordinal(raw):
    x = np.asarray(raw)
    if not np.all(np.isfinite(x) & (x >= 1) & (x <= 9)):
        raise ValueError('V/A must be finite and in [1,9]')
    return np.clip(np.floor(x + .5), 1, 9).astype(np.int64)


def binary(raw):
    ordinal(raw)  # validate, retain original precision for threshold
    return (np.asarray(raw) > 5).astype(np.int64)

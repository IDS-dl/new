import numpy as np


def measures(target, predicted):
    error = predicted.astype(np.float64) - target
    mse = np.mean(error ** 2)
    variance = np.var(target.astype(np.float64))
    return dict(mse=mse, mae=np.mean(np.abs(error)), rmse=np.sqrt(mse),
                pearson=np.corrcoef(target, predicted)[0, 1] if variance and np.std(predicted) else None,
                r2=1-mse/variance if variance else None)


def fill_baselines(signal, observed):
    indices = np.flatnonzero(observed)
    values = signal[indices]
    return np.full_like(signal, values.mean()), np.interp(np.arange(len(signal)), indices, values)

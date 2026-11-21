import numpy as np
from scipy.signal import butter, sosfiltfilt


def align_window(frame, start, end, fs, cfg):
    """Interpolate only short gaps, strictly inside the stimulus interval.

    Half-open grid; boundary extrapolation is limited to one native period.
    Native counts/coverage and max gap are checked before interpolation.
    """
    t, x = frame.timestamp.to_numpy(float), frame.value.to_numpy(float)
    mask = (t >= start) & (t < end)
    t, x = t[mask], x[mask]
    if len(t) < 2 or np.any(~np.isfinite(t)) or np.any(np.diff(t) <= 0):
        raise ValueError('missing_or_nonmonotonic_timestamps')
    good = np.isfinite(x)
    tg, xg = t[good], x[good]
    n = int(round((end - start) * fs))
    coverage = min(1., len(tg) / n)
    if len(tg) < 2 or coverage < 1 - cfg.get('max_missing_fraction', .05):
        raise ValueError('insufficient_coverage')
    grid = start + np.arange(n) / fs
    max_gap = float(np.max(np.diff(tg)))
    # Include endpoints, do not hide absent leading/trailing samples.
    if tg[0] - start > 1.1 / fs or grid[-1] - tg[-1] > 1.1 / fs:
        raise ValueError('incomplete_boundaries')
    if max_gap > cfg.get('max_gap_seconds', .5) + 1.1 / fs:
        raise ValueError('long_gap')
    out = np.interp(grid, tg, xg).astype(np.float32)
    return out, dict(missing_fraction=1-coverage, native_samples=len(t),
                     finite_samples=len(tg), max_gap_seconds=max_gap,
                     available_seconds=(tg[-1]-tg[0]+1/fs))


def filter_wave(x, modality, cfg):
    if not cfg.get(modality+'_filter', cfg.get('filter', True)):
        return x.copy()
    if modality == 'ppg':
        sos = butter(cfg.get('filter_order', 3), cfg.get('ppg_band', [.5, 4.]),
                     btype='bandpass', fs=64, output='sos')
    else:
        sos = butter(cfg.get('filter_order', 3), cfg.get('eda_lowpass', .8),
                     btype='lowpass', fs=4, output='sos')
    return sosfiltfilt(sos, x).astype(np.float32)


class FoldNormalizer:
    """All fitted parameters come from explicit training indices only."""
    def __init__(self, ppg='window_zscore', eda='log_robust'):
        if ppg not in ('window_zscore', 'standard') or eda not in ('log_robust', 'standard'):
            raise ValueError('Unknown normalization')
        self.ppg, self.eda, self.stats = ppg, eda, {}

    def fit(self, arrays, train_indices):
        if not len(train_indices):
            raise ValueError('Empty training split')
        self.fit_indices = list(map(int, train_indices))
        for name in ('ppg', 'eda'):
            x = arrays[name][train_indices].astype(np.float64).ravel()
            if name == 'eda' and self.eda == 'log_robust':
                x = np.log1p(np.maximum(x, 0))
                center = np.median(x)
                scale = np.quantile(x, .75) - np.quantile(x, .25)
            else:
                center, scale = x.mean(), x.std()
            self.stats[name] = (float(center), float(max(scale, 1e-6)))
        return self

    def transform(self, name, x):
        x = np.asarray(x, dtype=np.float32)
        if name == 'ppg' and self.ppg == 'window_zscore':
            return (x-x.mean(axis=-1, keepdims=True)) / np.maximum(x.std(axis=-1, keepdims=True), 1e-6)
        if name == 'eda' and self.eda == 'log_robust':
            x = np.log1p(np.maximum(x, 0))
        c, s = self.stats[name]
        return ((x-c)/s).astype(np.float32)

    def state_dict(self):
        return dict(ppg=self.ppg, eda=self.eda, stats=self.stats, fit_indices=self.fit_indices)

    @classmethod
    def from_state(cls, state):
        obj = cls(state['ppg'], state['eda'])
        obj.stats, obj.fit_indices = state['stats'], state['fit_indices']
        return obj

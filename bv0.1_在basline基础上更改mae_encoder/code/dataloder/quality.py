"""Descriptive quality flags; thresholds are configurable, not clinical SQI."""
import numpy as np


def inspect(t, x, fs, cfg=None):
    cfg = cfg or {}
    finite = np.isfinite(t) & np.isfinite(x)
    tf, xf = t[finite], x[finite]
    dt = np.diff(tf)
    dx = np.diff(xf)
    scale = np.median(np.abs(dx - np.median(dx))) if len(dx) else 0.
    flags = []
    if not len(xf):
        flags.append('no_finite_signal')
    flat = float(np.mean(np.abs(dx) <= cfg.get('flat_epsilon', 1e-8))) if len(dx) else 1.
    if flat > cfg.get('flat_fraction', .98):
        flags.append('flat')
    if len(dx) and np.any(np.abs(dx) > cfg.get('jump_mad', 50) * max(scale, 1e-6)):
        flags.append('abrupt_jump')
    if cfg.get('nonnegative', False) and np.any(xf < 0):
        flags.append('negative_conductance')
    ceiling = cfg.get('saturation_value')
    if ceiling is not None and np.any(np.abs(xf) >= ceiling):
        flags.append('saturation')
    return dict(n=len(x), nonfinite=int((~finite).sum()),
                duplicate_timestamps=int(np.sum(dt == 0)),
                backward_timestamps=int(np.sum(dt < 0)),
                timestamp_gaps=int(np.sum(dt > 1.5 / fs)),
                max_gap_seconds=float(dt.max()) if len(dt) else None,
                sampling_rate=float(1 / np.median(dt[dt > 0])) if np.any(dt > 0) else None,
                start=float(tf.min()) if len(tf) else None,
                end=float(tf.max()) if len(tf) else None,
                flat_fraction=flat, flags=flags,
                quality_score=max(0., 1. - .25 * len(flags)))

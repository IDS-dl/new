from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from .labels import ordinal, binary
from .preprocess import align_window, filter_wave
from .quality import inspect

REQUIRED = ('markers-phase2.csv', 'surveys.csv', 'signals-e4-bvp.csv', 'signals-e4-eda.csv')


def subject_dirs(root):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f'CSV root not found: {root}')
    # Include incomplete subjects, so missing files are visible in audit.
    dirs = sorted({p.parent for pattern in (*REQUIRED, 'params.csv') for p in root.rglob(pattern)})
    if not dirs:
        raise ValueError(f'No EmoWear CSV subjects under {root}')
    if len({p.name for p in dirs}) != len(dirs):
        raise ValueError('Duplicate subject directory names; choose a single csv root')
    return dirs


def read_csv(path, columns):
    df = pd.read_csv(path)
    if not set(columns) <= set(df.columns):
        raise ValueError(f'{path}: required columns {columns}, found {list(df.columns)}')
    for c in columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def build_windows(root, cfg):
    arrays = {k: [] for k in ('ppg', 'eda', 'ppg_raw', 'eda_raw')}
    rows, records, subjects, sources = [], [], [], []
    policy = cfg.get('window_policy', 'last30')
    duration = {'last30': 30, 'first30': 30, 'full60': 60}.get(policy)
    if duration is None or cfg.get('window_seconds', duration) != duration:
        raise ValueError('Use last30/first30 with 30s or full60 with 60s')
    qc = cfg.get('quality', {})
    for folder in subject_dirs(root):
        sid = folder.name
        report = dict(subject_id=sid, signals={}, errors=[], synchronization=None)
        subjects.append(report)
        for filename in (*REQUIRED, 'params.csv', 'provenance.json'):
            p = folder / filename
            if p.exists():
                sources.append(dict(path=str(p.resolve()), sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        trials = None
        try:
            markers = read_csv(folder / REQUIRED[0], ['seq', 'exp', 'vidB', 'postB'])
            surveys = read_csv(folder / REQUIRED[1], ['seq', 'exp', 'valence', 'arousal'])
            for table in (markers, surveys):
                keys = table[['seq', 'exp']].to_numpy()
                if not np.isfinite(keys).all() or not (keys == np.floor(keys)).all():
                    raise ValueError('missing_or_invalid_trial_keys')
            if markers.duplicated(['seq', 'exp']).any() or surveys.duplicated(['seq', 'exp']).any():
                raise ValueError('ambiguous_duplicate_trial_keys')
            trials = markers.merge(surveys[['seq','exp','valence','arousal']], on=['seq','exp'],
                                   how='outer', validate='one_to_one', indicator=True)
            report['marker_trials'], report['survey_trials'] = len(markers), len(surveys)
            signals = {}
            for name, filename, fs in [('ppg', REQUIRED[2], 64), ('eda', REQUIRED[3], 4)]:
                signals[name] = read_csv(folder / filename, ['timestamp', 'value'])
                f = signals[name]
                report['signals'][name] = inspect(f.timestamp.values, f.value.values, fs,
                                                 {**qc, 'nonnegative': name == 'eda'})
            if (folder / 'params.csv').exists():
                report['synchronization'] = pd.read_csv(folder / 'params.csv').to_dict('records')
            else:
                report['errors'].append('params.csv_missing_sync_provenance_unverified')
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            report['errors'].append(str(exc))
            report['accepted'] = 0
            if trials is None:
                records.append(dict(subject_id=sid, retained=False, reason=str(exc), scope='subject'))
            else:
                # Missing a whole modality must not erase the excluded V/A distribution.
                for trial in trials.to_dict('records'):
                    rec = dict(subject_id=sid, trial_id=f"{trial['seq']:g}:{trial['exp']:g}",
                               seq=trial['seq'], exp=trial['exp'], retained=False, reason=str(exc),
                               scope='trial', valence_raw=trial['valence'], arousal_raw=trial['arousal'])
                    for dim in ('valence', 'arousal'):
                        value = rec[dim+'_raw']
                        if np.isfinite(value) and 1 <= value <= 9:
                            rec[dim+'_ord'] = int(ordinal(value))
                            rec[dim+'_binary'] = int(binary(value))
                    records.append(rec)
            continue
        for trial in trials.to_dict('records'):
            rec = dict(subject_id=sid, trial_id=f"{trial['seq']:g}:{trial['exp']:g}",
                       seq=trial['seq'], exp=trial['exp'], window_id=policy,
                       valence_raw=trial['valence'], arousal_raw=trial['arousal'])
            rec['retained'] = False
            try:
                if trial['_merge'] != 'both':
                    raise ValueError('unmatched_marker_or_survey')
                raw = np.array([trial['valence'], trial['arousal']], dtype=np.float32)
                labels = ordinal(raw)
                if cfg.get('exclude_neutral', False) and np.any(raw == 5):
                    raise ValueError('neutral_excluded')
                start, end = float(trial['vidB']), float(trial['postB'])
                if not np.isfinite([start,end]).all() or end-start < duration:
                    raise ValueError('invalid_or_short_video')
                if policy == 'last30':
                    start = end-duration
                else:
                    end = start+duration
                rec.update(window_start=start, window_end=end, vidB=trial['vidB'], postB=trial['postB'],
                           valence_ord=int(labels[0]), arousal_ord=int(labels[1]),
                           valence_binary=int(binary(raw)[0]), arousal_binary=int(binary(raw)[1]))
                waves = {}
                for name, fs in [('ppg',64),('eda',4)]:
                    f = signals[name]
                    segment = f[(f.timestamp >= start) & (f.timestamp < end)]
                    quality = inspect(segment.timestamp.values, segment.value.values, fs,
                                      {**qc, 'nonnegative': name == 'eda'})
                    rec[f'{name}_quality'] = quality
                    if set(quality['flags']) & set(qc.get('reject_flags', ['negative_conductance'])):
                        raise ValueError(f'{name}_quality_rejected')
                    wave, coverage = align_window(f, start, end, fs, qc)
                    rec[f'{name}_coverage'] = coverage
                    waves[name+'_raw'] = wave
                    waves[name] = filter_wave(wave, name, cfg.get('preprocess', {}))
                for k in arrays:
                    arrays[k].append(waves[k])
                rec.update(retained=True, reason='accepted')
                rows.append({k:v for k,v in rec.items() if not isinstance(v,dict)})
            except ValueError as exc:
                rec['reason'] = str(exc)
            records.append(rec)
        report['accepted'] = sum(r['subject_id']==sid for r in rows)
    arrays = {k: np.stack(v).astype(np.float32) if v else np.empty((0, (64 if k.startswith('ppg') else 4)*duration), np.float32) for k,v in arrays.items()}
    return arrays, pd.DataFrame(rows), records, subjects, sources


class EmoWearDataset(Dataset):
    def __init__(self, arrays, meta, indices, normalizer):
        self.meta = meta.iloc[indices].reset_index(drop=True)
        self.ppg = normalizer.transform('ppg', arrays['ppg'][indices])
        self.eda = normalizer.transform('eda', arrays['eda'][indices])

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):
        r = self.meta.iloc[i]
        return dict(ppg=torch.from_numpy(self.ppg[i][None]), eda=torch.from_numpy(self.eda[i][None]),
                    raw=torch.tensor([r.valence_raw,r.arousal_raw],dtype=torch.float32),
                    ordinal=torch.tensor([r.valence_ord,r.arousal_ord],dtype=torch.long),
                    binary=torch.tensor([r.valence_binary,r.arousal_binary],dtype=torch.long))


def make_loaders(arrays, meta, splits, normalizer, batch_size, seed):
    generator = torch.Generator().manual_seed(seed)
    return [
        DataLoader(
            EmoWearDataset(arrays, meta, indices, normalizer),
            batch_size=batch_size,
            shuffle=split == 0,
            num_workers=0,
            generator=generator,
        )
        for split, indices in enumerate(splits)
    ]

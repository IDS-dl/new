import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from dataloder.dataset import build_windows


def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k):json_safe(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj,np.generic):
        return json_safe(obj.item())
    if isinstance(obj,float) and not np.isfinite(obj):
        return None
    return obj


def save_json(path, obj):
    Path(path).write_text(json.dumps(json_safe(obj),ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def distributions(frame):
    result = {}
    for dimension in ('valence','arousal'):
        for kind in ('raw','ord','binary'):
            key = dimension+'_'+kind
            if key in frame:
                result[key] = frame[key].value_counts(dropna=False).sort_index().to_dict()
        key = dimension+'_ord'
        if key in frame:
            counts = frame[key].value_counts()
            result[dimension+'_rare_levels_lt10'] = {k:int(counts.get(k,0)) for k in range(1,10) if counts.get(k,0)<10}
    return result


def run(root, cfg, output):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    arrays, meta, records, subjects, sources = build_windows(root,cfg)
    meta.to_csv(output/'samples.csv',index=False)
    pd.DataFrame(records).to_csv(output/'quality.csv',index=False)
    save_json(output/'quality.json',records)
    save_json(output/'sources.json',sources)
    rejected = pd.DataFrame([r for r in records if not r['retained']])
    summary = dict(data_root=str(Path(root).resolve()),subject_count=len(subjects),
                   subjects=subjects,paired_windows=len(meta),
                   designed_trials_per_subject=38,
                   observed_trial_deficits={s['subject_id']:38-s['marker_trials'] for s in subjects if s.get('marker_trials',38)<38},
                   all_observed_distribution=distributions(pd.DataFrame(records)),
                   quality_flag_counts={name:pd.Series([flag for r in records for flag in r.get(name+'_quality',{}).get('flags',[])],dtype=str).value_counts().to_dict() for name in ('ppg','eda')},
                   retained_distribution=distributions(meta),
                   rejected_distribution=distributions(rejected),
                   rejected_reasons=rejected.reason.value_counts().to_dict() if len(rejected) else {},
                   config=cfg, synthetic=bool((Path(root)/'SYNTHETIC.json').exists()))
    save_json(output/'audit.json',summary)
    (output/'audit.md').write_text(f"# EmoWear audit\n\nRoot: `{summary['data_root']}`\n\nSubjects: {len(subjects)}; paired windows: {len(meta)}; rejected records: {len(rejected)}.\n\nSynthetic: {summary['synthetic']}.\n\nSee audit.json, quality.json, samples.csv and sources.json for per-subject, per-trial, exclusion and source hash records.\n")
    return arrays,meta,summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',required=True)
    p.add_argument('--config')
    p.add_argument('--output',default='outputs/emowear/audit')
    a=p.parse_args()
    cfg=yaml.safe_load(Path(a.config).read_text()) if a.config else {}
    _,meta,s=run(a.data_root,cfg,a.output)
    print(f"subjects={s['subject_count']} paired_windows={len(meta)} output={a.output}")

if __name__=='__main__':
    main()

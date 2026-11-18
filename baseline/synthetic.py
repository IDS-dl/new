"""Small explicitly synthetic CSV fixture, never a scientific result."""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd


def generate(root, subjects=10, trials=3, seed=42):
    root=Path(root)
    root.mkdir(parents=True,exist_ok=True)
    (root/'SYNTHETIC.json').write_text(json.dumps(dict(synthetic=True,seed=seed)))
    rng=np.random.default_rng(seed)
    for s in range(subjects):
        folder=root/f'SYNTH-{s:02}'
        folder.mkdir(exist_ok=True)
        markers=[]
        surveys=[]
        for j in range(trials):
            start=10+j*75
            markers.append(dict(seq=j+1,exp=100+j,preB=start-5,vidB=start,postB=start+61,surveyB=start+64,walkB=start+65,walkE=start+70))
            surveys.append(dict(seq=j+1,exp=100+j,valence=1+((s*3+j)%81)/10,arousal=9-((s*7+j*11)%81)/10))
        pd.DataFrame(markers).to_csv(folder/'markers-phase2.csv',index=False)
        # Reversed file order deliberately exercises key-based alignment.
        pd.DataFrame(surveys[::-1]).to_csv(folder/'surveys.csv',index=False)
        for name,fs in [('bvp',64),('eda',4)]:
            t=np.arange(0,10+trials*75,1/fs)
            x=np.sin(2*np.pi*1.2*t)+rng.normal(0,.05,len(t)) if name=='bvp' else 2+.1*np.sin(.1*t)+rng.normal(0,.005,len(t))
            pd.DataFrame(dict(timestamp=t,value=x)).to_csv(folder/f'signals-e4-{name}.csv',index=False)
        pd.DataFrame([dict(device='e4',shift=0,cf=1)]).to_csv(folder/'params.csv',index=False)
    return root

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    p.add_argument('--subjects',type=int,default=10)
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    generate(a.output,a.subjects,seed=a.seed)

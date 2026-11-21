from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from model.moment_wrapper import RATES


def plot_examples(output, name, ratio, meta, target, observed, methods, *, styles=None):
    output.mkdir(parents=True,exist_ok=True)
    rate=RATES[name]
    labels = styles if styles is not None else {
        'zero-shot': ('Zero-shot MOMENT', '#d55e00', '--'),
        'full shared': ('Shared MOMENT', '#7b3294', '-.'),
        'full independent': ('Independent MOMENT', '#009e73', '-'),
        'linear': ('Linear interpolation', '#656565', ':'),
    }
    for idx in meta.drop_duplicates('subject_id').index[:3]:
        row=meta.iloc[idx]
        signal,seen=target[idx],observed[idx]
        seconds=np.arange(len(signal))/rate
        fig,axes=plt.subplots(2,1,figsize=(13,6),layout='constrained')
        changes=np.flatnonzero(np.diff(np.r_[False,~seen,False]))
        # Use the earliest hidden patch as the center of the detail panel.
        focus=changes[0]/rate
        for ax in axes:
            for start,end in zip(changes[::2],changes[1::2]):
                ax.axvspan(start/rate,end/rate,alpha=.16,color='orange')
            ax.plot(seconds,signal,color='black',lw=1,label='Target')
            ax.plot(seconds,np.where(seen,signal,np.nan),color='tab:blue',lw=1.4,label='Observed')
            for key,(label,color,style) in labels.items():
                ax.plot(seconds,np.where(seen,np.nan,methods[key][idx]),lw=1.15,
                        label=label,color=color,ls=style)
            if name=='ppg':
                for boundary in (8,16,24):
                    ax.axvline(boundary,color='grey',ls=':',lw=.7)
            ax.set(ylabel=f'{name.upper()} (normalized)',xlabel='Seconds')
        axes[0].set(xlim=(0,len(signal)/rate),title=f'{row.subject_id} / {row.trial_id} — {rate} Hz — mask {ratio:g}')
        axes[0].legend(ncol=3,fontsize=8)
        axes[1].set(xlim=(max(0,focus-1),min(len(signal)/rate,focus+(4 if name=='ppg' else 8))))
        fig.savefig(output/f'{name}_mask{ratio:g}_sample{idx}.png',dpi=125)
        plt.close(fig)

"""Adaptation of upstream CNN patches + pre-norm Transformer + V/A heads."""
import math
import torch
from torch import nn
from .fusion import fusion_mlp
from .heads import make_head


class SignalEncoder(nn.Module):
    def __init__(self, modality, hidden=40, depth=2, heads=4, dropout=.3):
        super().__init__()
        # 0.52s PPG / 1.25s EDA kernels; native-rate ~1s patches.
        kernel,pool,stride = (33,64,32) if modality=='ppg' else (5,4,2)
        self.patch=nn.Sequential(nn.Conv1d(1,hidden,kernel,padding=kernel//2),
                                 nn.GroupNorm(1,hidden),nn.ELU(),
                                 nn.AvgPool1d(pool,stride),nn.Dropout(dropout),
                                 nn.Conv1d(hidden,hidden,1))
        block=nn.TransformerEncoderLayer(hidden,heads,hidden*4,dropout,
                                         activation='gelu',batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(block,depth,enable_nested_tensor=False)
        self.norm=nn.LayerNorm(hidden)

    def forward(self,x):
        if x.ndim!=3 or x.shape[1]!=1:
            raise ValueError('Expected [B,1,T]')
        x=self.patch(x).transpose(1,2)
        # Explicit temporal position; supports either 30s or 60s inputs.
        pos=torch.arange(x.shape[1],device=x.device,dtype=x.dtype)[:,None]
        freq=torch.exp(torch.arange(0,x.shape[-1],2,device=x.device,dtype=x.dtype)*(-math.log(10000.)/x.shape[-1]))
        pe=torch.zeros_like(x[0])
        pe[:,0::2]=torch.sin(pos*freq)
        pe[:,1::2]=torch.cos(pos*freq[:pe[:,1::2].shape[-1]])
        return self.norm(self.encoder(x+pe)).mean(1)


class EmoWearDualConformer(nn.Module):
    def __init__(self, task='coral', modalities='ppg_eda', hidden_dim=40, depth=2, heads=4, dropout=.3):
        super().__init__()
        if isinstance(modalities,list):
            modalities='_'.join(modalities)
        if modalities not in ('ppg_only','eda_only','ppg_eda'):
            raise ValueError('Use ppg_only, eda_only, ppg_eda or [ppg,eda]')
        if hidden_dim%heads:
            raise ValueError('hidden_dim must be divisible by heads')
        self.task=task
        self.branches=nn.ModuleDict({m:SignalEncoder(m,hidden_dim,depth,heads,dropout)
                                   for m in ('ppg','eda') if m in modalities})
        self.fusion=fusion_mlp(len(self.branches)*hidden_dim,hidden_dim,dropout)
        self.valence_head=make_head(task,hidden_dim)
        self.arousal_head=make_head(task,hidden_dim)

    def forward(self,ppg=None,eda=None):
        inputs=dict(ppg=ppg,eda=eda)
        z=self.fusion(torch.cat([b(inputs[m]) for m,b in self.branches.items()],-1))
        out={f'{name}_logits':head(z) for name,head in [('valence',self.valence_head),('arousal',self.arousal_head)]}
        if self.task=='regression':
            out={k:1+8*v for k,v in out.items()}
        return out

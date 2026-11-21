import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download
from momentfm import MOMENTPipeline

REPO = 'AutonLab/MOMENT-1-small'
REVISION = '411e288267f82cce86296dbe4d6c8bc533cc162f'
OFFICIAL_SHA256 = {
    'config.json': '6456b8e6f017323e8ef372c338c993380e246cb6291ffb8ca21fe7dfc53d0a6f',
    'model.safetensors': '785e6c6f57ffa7cac7e2a1fff6369618d49f2f441563ddea76c866231e5aa877',
}


def load_moment(source=None, mode='full'):
    if mode not in ('full', 'head', 'zero'):
        raise ValueError(f'Unknown mode: {mode}')
    if source is None:
        source = snapshot_download(REPO, revision=REVISION,
                                   allow_patterns=['config.json', 'model.safetensors'])
    source = Path(source)
    for name, expected in OFFICIAL_SHA256.items():
        with (source / name).open('rb') as file:
            if hashlib.file_digest(file, 'sha256').hexdigest() != expected:
                raise ValueError(f'{name} does not match {REPO} revision {REVISION}')
    config = json.loads((source / 'config.json').read_text())
    if tuple(config[k] for k in ('seq_len', 'patch_len', 'patch_stride_len')) != (512, 8, 8):
        raise ValueError('Expected the official 512/8/8 configuration')
    model = MOMENTPipeline.from_pretrained(
        str(source), strict=True, model_kwargs=dict(task_name='reconstruction',
            freeze_encoder=mode != 'full', freeze_embedder=mode != 'full',
            freeze_head=mode == 'zero', enable_gradient_checkpointing=False))
    model.init()
    return model

from .dual_conformer import EmoWearDualConformer


def model_from_config(config):
    keys = ('task', 'modalities', 'hidden_dim', 'depth', 'heads', 'dropout')
    return EmoWearDualConformer(**{key: config[key] for key in keys})

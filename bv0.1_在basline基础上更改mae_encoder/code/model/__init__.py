from .dual_conformer import EmoWearDualConformer


def model_from_config(config):
    if config.get("model") == "mae_classifier":
        from .mae_classifier import MAEClassifier
        return MAEClassifier(config)
    keys = ('task', 'modalities', 'hidden_dim', 'depth', 'heads', 'dropout')
    return EmoWearDualConformer(**{key: config[key] for key in keys})

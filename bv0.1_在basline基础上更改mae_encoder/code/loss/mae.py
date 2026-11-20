from torch.nn import functional as F
from model.mae_encoder import patchify, RATES


def masked_mean(elements, mask):
    return elements.mean(-1)[mask].mean()


def masked_difference_loss(prediction, target, mask):
    return masked_mean((prediction.diff(dim=-1) - target.diff(dim=-1)).abs(), mask)


def reconstruction_loss(outputs, batch, masks, cfg):
    components = {}
    total = 0
    for name, width in RATES.items():
        target = patchify(batch[name], width)
        prediction = outputs[name]
        errors = F.huber_loss(prediction, target, reduction='none', delta=cfg['huber_delta'])
        components[name] = masked_mean(errors, masks[name])
        total = total + cfg['lambda_' + name] * components[name]
        if cfg['use_difference_loss']:
            key = name + '_diff'
            components[key] = masked_difference_loss(prediction, target, masks[name])
            total = total + cfg['lambda_' + key] * components[key]
    return {'loss': total, **components}

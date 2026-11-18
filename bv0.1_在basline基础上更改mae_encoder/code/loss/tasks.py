from torch.nn import functional as F

from .coral import coral_loss


REGRESSION_LOSSES = {'huber': F.huber_loss, 'mae': F.l1_loss, 'mse': F.mse_loss}


def loss_fn(outputs, batch, task, regression_loss='huber'):
    loss = 0
    for index, dimension in enumerate(('valence', 'arousal')):
        logits = outputs[f'{dimension}_logits']
        if task == 'binary':
            loss += F.cross_entropy(logits, batch['binary'][:, index])
        elif task == 'softmax':
            loss += F.cross_entropy(logits, batch['ordinal'][:, index] - 1)
        elif task == 'coral':
            loss += coral_loss(logits, batch['ordinal'][:, index])
        else:
            loss += REGRESSION_LOSSES[regression_loss](logits[:, 0], batch['raw'][:, index])
    return loss

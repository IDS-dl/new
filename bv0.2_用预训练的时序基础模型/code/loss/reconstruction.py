def masked_mse(pred, target, valid, observed):
    hidden = (valid & ~observed)[:, None]
    return (pred[hidden].float() - target[hidden]).square().mean()

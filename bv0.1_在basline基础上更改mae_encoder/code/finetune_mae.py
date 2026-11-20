from train import load_config_args, run_training


if __name__ == '__main__':
    cfg = load_config_args()
    if cfg['model'] != 'mae_classifier' or cfg['task'] != 'coral':
        raise ValueError('Expected mae_classifier configuration with CORAL downstream task')
    run_training(cfg)

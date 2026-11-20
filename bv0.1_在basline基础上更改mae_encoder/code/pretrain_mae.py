from train import load_config_args, run_training


if __name__ == '__main__':
    cfg = load_config_args()
    if cfg['model'] != 'waveform_mae' or cfg['task'] != 'coral':
        raise ValueError('Expected waveform_mae configuration with CORAL downstream task')
    run_training(cfg)

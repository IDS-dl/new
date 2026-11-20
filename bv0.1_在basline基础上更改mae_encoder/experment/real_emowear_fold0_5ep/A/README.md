# A

bv0.1 · waveform_mae · coral · seed 42

Run from code/: `python pretrain_mae.py --config /mnt/nvme1n1p1/chentianjiao/emo/bv0.1_在basline基础上更改mae_encoder/experment/real_emowear_fold0_5ep/A/results/config.yaml --output-dir NEW_DIRECTORY`

See results/config.yaml for preprocessing, architecture and budget, results/splits.json for subject splits. Zero-phase filtering precedes masking and can mix neighboring patches.

View loss curves with `tensorboard --logdir results/tensorboard`.

Fold 0: best epoch 5, validation loss 0.20920.

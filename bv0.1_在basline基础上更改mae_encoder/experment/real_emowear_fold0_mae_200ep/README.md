# real_emowear_fold0_mae_200ep

bv0.1 · waveform_mae · coral · seed 42

Run from code/: `python pretrain_mae.py --config /mnt/nvme1n1p1/chentianjiao/emo/bv0.1_在basline基础上更改mae_encoder/experment/real_emowear_fold0_mae_200ep/results/config.yaml --output-dir NEW_DIRECTORY`

See results/config.yaml for preprocessing, architecture and budget, results/splits.json for subject splits. Zero-phase filtering precedes masking and can mix neighboring patches.

View loss curves with `tensorboard --logdir results/tensorboard`.

Fold 0: best epoch 71, validation loss 0.19859.

## Fold 0 延长预训练的结果

以同一 seed 从头训练，沿用原 5 轮实验的数据、划分、归一化、batch 和验证 mask。已核对前 5 轮各项 loss 与原运行一致。训练上限 200 轮，patience 15，min_delta 0。第 71 轮验证总 Huber 最低，第 86 轮触发早停。最佳权重为 `weights/fold_0/best_mae.pt`。

| 固定的 228 个验证窗口 | 第 5 轮 | 第 71 轮 | 可见点均值填充 |
|---|---:|---:|---:|
| PPG 遮挡 Huber | 0.201828 | 0.198283 | 0.198817 |
| EDA 遮挡 Huber | 0.007370 | 0.000312 | 0.000167 |
| 总 Huber | 0.209198 | 0.198595 | 0.198985 |

总损失下降约 5.1%，约三分之二的绝对改善来自 EDA。PPG 略优于可见均值，但重建标准差与目标标准差之比的中位数从 0.167 降至 0.035，波动进一步压缩。EDA 的对应比值从 5.48 降至 1.16，更接近目标振幅，点对点误差仍高于可见均值。实验评估的是预训练波形重建，下游 V/A 收益仍待同预算微调验证。

[指标比较](results/comparison.csv) · [训练曲线](results/fold_0/loss.png) · [逐轮记录](results/fold_0/history.csv) · [固定验证样本重建](results/reconstruction_val_0.png)

在版本 code/ 下使用旧实验配置可复现本次设置。
```bash
python pretrain_mae.py --config ../experment/real_emowear_fold0_5ep/A/results/config.yaml \
  --output-dir ../experment/real_emowear_fold0_mae_200ep_repeat --set epochs=200
```

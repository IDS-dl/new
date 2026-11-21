# 独立 MOMENT 全量微调 100 轮

沿用真实 EmoWear Fold 0 的 883 个训练、228 个验证和 304 个测试窗口。两档遮挡率下分别训练独立的 PPG 和 EDA 模型，各自从官方 MOMENT-1-small checkpoint 初始化。四组全部训练 100 轮，关闭早停，保存验证最优和第 100 轮权重。学习率、归一化、批大小、随机种子与原实验一致。

## 固定测试 mask 下的结果

表中为归一化空间中遮挡有效点上的 RMSE。变化率以原实验保存的验证最优 checkpoint 为分母。先重算原 checkpoint 的成绩，再逐窗口核对测试身份和 mask 与原实验一致。

| 模态 | 标称遮挡率 | 原实验最优轮及 RMSE | 100 轮内验证最优轮及 RMSE | 第 100 轮 RMSE | 第 100 轮 RMSE 降幅 |
|---|---:|---:|---:|---:|---:|
| PPG | 30% | 30 / 0.137113 | 100 / 0.120295 | 0.120295 | +12.27% |
| EDA | 30% | 30 / 0.006674 | 96 / 0.007831 | 0.007515 | -12.61% |
| PPG | 60% | 30 / 0.275716 | 92 / 0.241859 | 0.239806 | +13.02% |
| EDA | 60% | 12 / 0.010496 | 86 / 0.009617 | 0.009636 | +8.19% |

正值表示 RMSE 下降。

[完整指标与验证 MSE](results/comparison.csv) · [Loss 曲线](results/loss_curves.png) · [训练与 TensorBoard 核验](results/verification.csv)

以下命令在 bv0.2 的 code/ 目录执行。输出目录需为新目录。

```bash
python train_moment_independent.py --reference ../experment/moment_stage1 \
  --output ../experment/moment_independent_100ep_repeat --epochs 100
```

每组的两份 checkpoint 保存在 weights/ 对应子目录。实验根目录下可运行 `tensorboard --logdir results/tensorboard`。

结论基于本次固定受试者划分与既有滤波。先滤波后遮挡会带来跨 patch 的信息传递。

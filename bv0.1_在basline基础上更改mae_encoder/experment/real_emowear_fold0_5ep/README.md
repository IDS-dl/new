# bv0.1 真实 EmoWear Fold 0 小规模验证

[开发集数据审查报告](results/data_review.md)记录了数据来源、排除统计、质量标记、标签分布和五折划分。只训练 Fold 0，各运行 5 epochs，seed 42，使用现有归一化与滤波，不额外筛除 trial。

## 实验与结果

| 实验 | 配置 | 训练 loss 首轮 → 末轮 | 验证 loss 首轮 → 末轮 | 最佳 epoch |
|---|---|---:|---:|---:|
| A，双模态 MAE | 60% 独立随机 mask，masked Huber，batch 64 | 0.5852 → 0.3235 | 0.3990 → 0.2092 | 5 |
| B，E1 随机初始化 | 共享 encoder + 双 CORAL，batch 32 | 7.1547 → 7.0321 | 7.0790 → 7.0932 | 1 |
| C，E2 同折预训练 | 从 A 初始化，其余监督配置与 B 一致 | 7.1271 → 7.0402 | 7.1055 → 7.0843 | 4 |

MAE 训练 PPG/EDA 分项分别从 0.3845/0.2006 到 0.3151/0.0084，验证分项从 0.2527/0.1463 到 0.2018/0.0074。三组所有 epoch 的 loss 均有限。训练曲线和逐轮数值见 [A 曲线](A/results/fold_0/loss.png)、[A 分项 CSV](A/results/fold_0/history.csv)、[B 曲线](B/results/fold_0/loss.png)、[C 曲线](C/results/fold_0/loss.png)。

在全部 228 个验证窗口上，用与验证训练循环相同的固定 mask 和最佳 checkpoint 比较遮挡位置 Huber。PPG 的模型、恒零、可见点均值常数三种方法分别为 0.20183、0.19802、0.19882。EDA 分别为 0.00737、0.57160、0.00017。两种输出都有波动，但 PPG 的重建与目标标准差之比中位数仅 0.167，波动明显被压缩。EDA 的比值中位数为 5.48，原始窗口内的 EDA 往往非常平稳，输出波动偏大。总 loss 有初步收敛迹象，但两种模态的动态恢复均未超过可见均值参照。[逐窗口统计](results/masked_waveforms.csv) · [验证样本 1](results/reconstruction_val_0.png) · [验证样本 2](results/reconstruction_val_38.png) · [验证样本 3](results/reconstruction_val_76.png)。三张图来自三个不同验证受试者，橙色为遮挡区域。

以下指标比较 304 个测试窗口的 CORAL 离散预测与原始连续评分。

| 实验 | 维度 | MAE | RMSE | Spearman | 等级频数 |
|---|---|---:|---:|---:|---|
| E1 | V | 1.785 | 2.166 | 0.102 | 5级 279，6级 25 |
| E1 | A | 1.768 | 2.111 | 0.129 | 4级 16，5级 288 |
| E2 | V | 1.816 | 2.237 | 0.066 | 5级 191，6级 113 |
| E2 | A | 1.748 | 2.059 | 0.189 | 4级 83，5级 221 |

两个模型均出现明显的中心等级集中。测试预测全部在 4–6 级，每个头只输出两个等级。E2 的 Arousal 有小幅改善，Valence 变差。5 epochs 单折结果不足以支持预训练带来普遍收益。[真实与预测等级图](results/predicted_distributions.png) · [含整数标签与期望评分的指标](results/scores.csv)

## 实现核验与下一步

6 项现有单元测试通过。三阶段 checkpoint 记录的受试者划分、源文件身份和归一化状态一致，使用训练索引重新拟合后统计量一致。重新执行 C 的初始化步骤后逐张量验证其 encoder 与 A 的最佳权重完全相同。E1/E2 监督配置仅输出目录和预训练来源不同。E0 配置的窗口、滤波、质量阈值、归一化、划分、监督 epoch、patience 和 batch 与监督对照匹配。本次仅核对 E0 配置。用每个最佳监督 checkpoint 在同一个训练 minibatch 上分别反传两个头的 loss，头部和共享 encoder 梯度范数均有限且非零。[梯度范数](results/gradient_norms.csv)

受试者与拟合统计方面未发现泄漏。PPG standard、EDA log_robust 仅从训练折拟合。既有零相位滤波在窗口内 masking 之前进行，信息可能跨 patch 边界传播。单测隔离结论只适用于已经预处理的输入。

建议暂缓完整 100 epochs × 五折。优先在训练和验证集上做一个可见均值及局部插值参照，按信号和受试者核查低振幅 PPG 与平稳 EDA；对照关闭滤波，检查 PPG 重建能否超过简单参照。对监督头检查连续期望分数、CORAL 各阈值通过率、训练集与验证集的等级覆盖，排查中心预测集中。固定下一轮方案后再进入五折；避免用这次 Fold 0 测试指标反复挑选超参数。

## 复现与 TensorBoard

从本版本 code/ 运行。复现时将 OUT 指向新目录。配置保存在 A、B、C 的 results/config.yaml，权重在各自 weights/fold_0/。TensorBoard 的训练与验证标量已与历史 CSV 逐轮核对。

```bash
DATA="$(realpath ../../datasets/EmoWear/EmoWear_development)"
OUT=../experment/real_emowear_fold0_repeat
OPTS=(--data-root "$DATA" --set epochs=5 --set 'fold_ids=[0]' --set pilot=true)
python pretrain_mae.py --config configs/emowear_mae.yaml "${OPTS[@]}" --output-dir "$OUT/A"
python finetune_mae.py --config configs/emowear_mae_coral.yaml "${OPTS[@]}" --output-dir "$OUT/B"
python finetune_mae.py --config configs/emowear_mae_coral.yaml "${OPTS[@]}" --pretrained-root "$OUT/A" --output-dir "$OUT/C"
```

在本实验目录运行 `tensorboard --logdir_spec A:A/results/tensorboard,B:B/results/tensorboard,C:C/results/tensorboard` 查看三组 loss。

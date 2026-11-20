# bv0.1 双模态 MAE 合成数据验证

目标是检验按受试者独立预训练、迁移和推理的流程。E0 使用原有 DualConformer，E1 使用随机初始化的共享 Transformer，E2 与 E1 共用监督配置，并从对应折的 MAE checkpoint 初始化 encoder。

每秒的 PPG 64 点和 EDA 4 点独立线性映射到 64 维，加上共享时间位置和独立模态编码。30 秒窗口中每个模态独立遮挡 18 个 patch，共 24 个可见 token 进入两层共享 encoder。两层 32 维 decoder 将可学习的 mask token 插回原位置，输出 30×64 和 30×4 的重建波形。60 秒窗口的 patch 数为 60。各模态在遮挡区域分别取 Huber 均值后相加，默认 delta 和两项权重均为 1。可通过配置启用 patch 内差分 L1。微调时编码全部 patch，各模态单独平均池化后拼接，融合到 64 维，以两个八阈值 CORAL 头分别预测 V/A，优化两项 CORAL loss 之和。

## 复现

在本版本的 `code/` 目录、安装了 `requirements-emowear.txt` 的环境中运行。以下命令生成 10 名合成受试者的数据，以五折、每阶段每折一个 epoch 验证所有流程。

```bash
python test_mae.py
bash smoke_mae.sh ../experment/mae_smoke_repeat
```

针对已有 EmoWear CSV，以下为各入口的独立运行示例。将 DATA 指向数据根目录，并为新运行使用新的输出路径。

```bash
DATA=/path/to/emowear/csv
python train.py --config configs/emowear_e0.yaml --data-root "$DATA" --output-dir ../experment/e0
python pretrain_mae.py --config configs/emowear_mae.yaml --data-root "$DATA" --output-dir ../experment/pre
python finetune_mae.py --config configs/emowear_mae_coral.yaml --data-root "$DATA" --output-dir ../experment/e1
python finetune_mae.py --config configs/emowear_mae_coral.yaml --data-root "$DATA" --pretrained-root ../experment/pre --output-dir ../experment/e2
python compare_mae.py ../experment/e0 ../experment/e1 ../experment/e2 --output-dir ../experment/comparison
python inference.py --checkpoint ../experment/e2/weights/fold_0/best.pt --sample aligned_raw_ppg_eda.npz
```

预训练配置默认采用 100 epochs、AdamW、学习率 0.001、batch 64、patience 15。监督配置默认采用 60 epochs、batch 32、patience 10，encoder 和监督头的学习率分别为 0.0001 和 0.0003。微调加上 `--set freeze_encoder=true` 可冻结 encoder。预训练加上 `--set use_difference_loss=true` 可启用差分项。两阶段均可使用 `--set window_seconds=60 --set window_policy=full60`。在各运行目录启动 `tensorboard --logdir results/tensorboard`。

## 结果和限制

合成数据共 30 个窗口。五折均完成预训练、E0、E1、E2 的一次训练与验证，三个监督实验分别输出 30 条留出测试预测。[逐指标五折均值和标准差](results/mae_comparison.csv) 含原始与整数标签两个视角。若某折的预测等级恒定，其 Spearman 相关系数无定义，表中保留空值。此实验验证实现的可运行性，真实受试者上的预训练收益仍需正式对照实验。

每折使用训练索引拟合 PPG standard 与 EDA log_robust，验证集选择最佳 epoch，测试集只用于监督模型的最终评估。预训练验证使用固定随机种子以比较不同 epoch。加载权重时核对折号、受试者与样本划分、数据来源、归一化参数和编码器配置。数据源清单包含路径，迁移到新路径会触发检查。窗口内零相位滤波发生在 masking 之前，可能跨 patch 边界传播信息。隐藏 patch 隔离测试针对预处理后的输入。当前支持独立随机 mask，Block Mask 接口保留待实现。

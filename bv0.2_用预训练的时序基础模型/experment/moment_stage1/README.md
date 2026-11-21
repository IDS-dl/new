# bv0.2 MOMENT Stage 1

在 EmoWear 开发集 Fold 0 上比较官方 MOMENT zero-shot、共享参数微调以及 PPG、EDA 各自独立微调。以全量微调为主，head-only 为对照。两个 mask ratio 对应独立训练。主流程保留 MOMENT 官方 Patch Embedding、Mask Embedding、位置编码、T5 Encoder、RevIN 和重建头。

训练、验证和测试分别为 24、6、8 名受试者及 883、228、304 个窗口。整条 trial 在受试者划分后才分段。30 秒 PPG 保留 1,920 点，分成 512、512、512、384 点。30 秒 EDA 保留 120 点，填充到 512。两路 patch 物理时长分别为 0.125 秒和 2 秒。仅有效遮挡点计入损失。训练折拟合 PPG standard 与 EDA log-robust 后，官方 RevIN 使用当前片段的可见点统计。

## 复现

从本版本的 `code/` 目录运行，使用 Python 3.11。默认目标数据为项目内 EmoWear 开发集，输出参数指定一个新目录。省略 `--official` 时，加载器从 Hugging Face 下载固定 revision。

```bash
python -m pip install -r requirements.txt
OFFICIAL=../experment/moment_stage1/weights/official
python test_moment.py --official "$OFFICIAL"
python train_moment.py --official "$OFFICIAL" --output ../experment/moment_stage1_repeat
```

配置采用 batch 16，每组共享与独立方法的训练样本相同。共享 MOMENT 对两个前向的 masked MSE 取等权平均后进行一次 AdamW 更新。全量微调学习率分别为 backbone 1e-5、head 1e-4，最多 30 epochs，patience 7。head-only 学习率 1e-4，最多 20 epochs，patience 5。实际 epoch、最佳轮次、耗时和显存见各运行的 `training.json`。

实验目录下运行 `tensorboard --logdir results/tensorboard`。正式结果及诊断见 [实验报告](results/report.md)。官方来源为 [MOMENT](https://github.com/moment-timeseries-foundation-model/moment) 和 [MOMENT-1-small](https://huggingface.co/AutonLab/MOMENT-1-small/tree/411e288267f82cce86296dbe4d6c8bc533cc162f)。

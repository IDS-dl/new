# CORAL 五折训练验证

本实验验证 bv0.1 拆分后的代码能独立完成训练、保存 checkpoint 和评估。模型沿用双分支编码器，以 PPG 和 EDA 预测 valence 与 arousal。

使用真实 EmoWear 数据，按受试者分为五折，每折在 CPU 上训练 1 个 epoch。五折共生成 1,415 条测试预测。合并测试集上的原始评分 MAE 为 valence 1.727、arousal 1.731。

在同版本的 `code/` 下运行以下命令复现。将数据路径替换为本地的 EmoWear CSV 根目录。

```bash
python train.py --config configs/emowear_coral.yaml \
  --data-root /path/to/emowear/csv \
  --output-dir ../experment/smoke_coral_repro \
  --set epochs=1 --set device=cpu --set pilot=true
```

五折汇总见 [evaluation.json](evaluation.json) 和 [metrics_summary.csv](metrics_summary.csv)。

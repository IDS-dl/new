#!/usr/bin/env bash
# Run from this version's code directory; the output directory must be new.
set -eu
output=${1:?Usage: bash smoke_mae.sh NEW_DIRECTORY}
mkdir "$output"
output="$(cd "$output" && pwd)"
python synthetic.py --output "$output/data"
options=(--data-root "$output/data" --set device=cpu --set epochs=1 --set pilot=true)
python pretrain_mae.py --config configs/emowear_mae.yaml "${options[@]}" --output-dir "$output/pre"
python train.py --config configs/emowear_e0.yaml "${options[@]}" --output-dir "$output/e0"
python finetune_mae.py --config configs/emowear_mae_coral.yaml "${options[@]}" --output-dir "$output/e1"
python finetune_mae.py --config configs/emowear_mae_coral.yaml "${options[@]}" --pretrained-root "$output/pre" --output-dir "$output/e2"
python compare_mae.py "$output/e0" "$output/e1" "$output/e2" --output-dir "$output/comparison"
python verify_mae_smoke.py "$output"

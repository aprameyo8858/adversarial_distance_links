#!/bin/bash
set -e

DATASETS=("cifar10" "cifar100")
MODELS=("resnet18" "resnet34" "resnet50")

for DATASET in "${DATASETS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        echo "=========================================================="
        echo " Running Full Training & C&W: $DATASET | $MODEL"
        echo "=========================================================="
        python train_eval_full.py \
            --dataset "$DATASET" \
            --model "$MODEL" \
            --seeds 42 43 44 \
            --epochs 200 \
            --batch-size 128 \
            --lr 0.1 \
            --out-dir "./results"
    done
done

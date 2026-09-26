#!/bin/bash
set -e

DATASETS=("cifar10" "cifar100")
MODELS=("resnet18" "resnet34" "resnet50")

for DATASET in "${DATASETS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        echo "=========================================================="
        echo " Running MIA Half-Dataset Pipeline: $DATASET | $MODEL"
        echo "=========================================================="
        python train_eval_mia.py \
            --dataset "$DATASET" \
            --model "$MODEL" \
            --seeds 42 43 44 \
            --num-shadows 64 \
            --epochs 200 \
            --batch-size 128 \
            --lr 0.1 \
            --out-dir "./results"

        echo " Generating 10-bin Figure 1 Verification Grid: $DATASET | $MODEL"
        python plot_verification.py \
            --full-csv "./results/full_${DATASET}_${MODEL}_3seeds.csv" \
            --mia-csv "./results/mia_${DATASET}_${MODEL}_3seeds.csv" \
            --dataset "$DATASET" \
            --model "$MODEL" \
            --out-dir "./plots"
    done
done

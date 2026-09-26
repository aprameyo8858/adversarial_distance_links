#!/bin/bash
set -e

DATASETS=("cifar10" "cifar100")

echo "=========================================================="
echo " Starting Appendix C.5 Auditing (Full Horizon Only)      "
echo " Configuration: 3 Seeds | Max Eps = 6.0 | ResNet-18       "
echo "=========================================================="

for DATASET in "${DATASETS[@]}"; do
    echo "=========================================================="
    echo " Executing Pipeline on: $DATASET"
    echo "=========================================================="
    
    python pgd_and_d_adv.py \
        --dataset "$DATASET" \
        --seeds 42 43 44 \
        --epochs 200 \
        --batch-size 128 \
        --lr 0.1 \
        --max-eps 6.0 \
        --out-dir "./results"
        
    echo "[+] Completed evaluation for $DATASET."
done

echo "=========================================================="
echo " All Appendix C.5 full-horizon evaluations completed.     "
echo "=========================================================="

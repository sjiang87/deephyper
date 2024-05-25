#!/bin/bash
conda activate agu      # or source activate agu

ROOT_DIR="./autognnuq/"
POST_DIR="./autognnuq/"
DATA_DIR="./autognnuq/data/"
SPLIT_TYPE="811"

for SEED in {0..7}; do
  python ood_pc9.py --ROOT_DIR "$ROOT_DIR" \
    --POST_DIR "$POST_DIR" \
    --DATA_DIR "$DATA_DIR" \
    --SPLIT_TYPE "$SPLIT_TYPE" \
    --seed $SEED
done
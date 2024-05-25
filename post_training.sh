#!/bin/bash
conda activate agu      # or source activate agu

ROOT_DIR="./autognnuq/"
POST_DIR="./autognnuq/"
DATA_DIR="./autognnuq/data/"
BATCH_SIZE=128
LEARNING_RATE=0.001
EPOCH=1000

for SEED in {0..7}; do
  for DATASET in "delaney" "freesolv" "lipo" "qm7" "qm9"; do
    if [ "$DATASET" == "qm9" ]; then
      SPLIT_TYPE="811"
    else
      SPLIT_TYPE="523"
    fi
    
    for MODE in "normal" "simple" "random"; do
      python post_training.py --ROOT_DIR "$ROOT_DIR" \
        --POST_DIR "$POST_DIR" \
        --DATA_DIR "$DATA_DIR" \
        --SPLIT_TYPE "$SPLIT_TYPE" \
        --seed $SEED \
        --dataset "$DATASET" \
        --batch_size $BATCH_SIZE \
        --learning_rate $LEARNING_RATE \
        --epoch $EPOCH \
        --mode "$MODE"
    done
  done
done
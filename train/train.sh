#! /bin/bash

torchrun --standalone --nnodes 1 --nproc-per-node 8 train.py \
  --grid_size 16 \
  --history_frames 2 \
  --pretrained_checkpoint YOUR_CHECKPOINT_PATH \


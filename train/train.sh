#! /bin/bash

torchrun --standalone --nnodes 1 --nproc-per-node 8 train.py \
  --grid_size 16 \
  --history_frames 2 \
  --pretrained_checkpoint YOUR_CHECKPOINT_PATH \
  --output_dir YOUR_OUTPUT_PATH \
  --batch_size 8 \
  --num_workers 8 \
  --num_epochs 100 \
  --learning_rate 0.0001 \
  --lr_decay_rate 0.9 \
  --lr_decay_steps 5000 \
  --weight_decay 0.0001 \
  --num_checkpoints 10 \

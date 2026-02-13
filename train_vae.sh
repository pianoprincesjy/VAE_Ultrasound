#!/bin/bash

# Set GPU device
export CUDA_VISIBLE_DEVICES=5

# Training parameters
TRAIN_DIR="../data/breast_tumors/train_images"
CONFIG_PATH="../stable-diffusion/configs/stable-diffusion/v1-inference.yaml"
OUTPUT_DIR="trained_models"
BATCH_SIZE=8
EPOCHS=100
LEARNING_RATE=1e-4
KL_WEIGHT=1.0
IMG_SIZE=256
DEVICE="cuda:0"  # Since we set CUDA_VISIBLE_DEVICES=5, cuda:0 will be GPU 5
SAVE_INTERVAL=10

# Run training
python train_vae.py \
    --train_dir $TRAIN_DIR \
    --config_path $CONFIG_PATH \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LEARNING_RATE \
    --kl_weight $KL_WEIGHT \
    --img_size $IMG_SIZE \
    --device $DEVICE \
    --save_interval $SAVE_INTERVAL

echo "Training completed! Results saved in $OUTPUT_DIR"

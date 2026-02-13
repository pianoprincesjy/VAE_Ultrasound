#!/bin/bash

# Set GPU device
export CUDA_VISIBLE_DEVICES=5

# Training parameters
TRAIN_DIR="../data/breast_tumors/train_images"
CONFIG_PATH="../stable-diffusion/configs/stable-diffusion/v1-inference.yaml"

# Pretrained checkpoint path (leave empty for training from scratch)
# Example: CHECKPOINT_PATH="../stable-diffusion/models/ldm/stable-diffusion-v1/model.ckpt"
CHECKPOINT_PATH="../stable-diffusion/models/ldm/text2img-large/model.ckpt"

OUTPUT_DIR="trained_models"
BATCH_SIZE=8
EPOCHS=4
LEARNING_RATE=1e-4
KL_WEIGHT=0.00000001
IMG_SIZE=256
DEVICE="cuda:0"  # Since we set CUDA_VISIBLE_DEVICES=5, cuda:0 will be GPU 5
SAVE_INTERVAL=1

# Build command
CMD="python train_vae.py \
    --train_dir $TRAIN_DIR \
    --config_path $CONFIG_PATH \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LEARNING_RATE \
    --kl_weight $KL_WEIGHT \
    --img_size $IMG_SIZE \
    --device $DEVICE \
    --save_interval $SAVE_INTERVAL"

# Add checkpoint path if provided
if [ ! -z "$CHECKPOINT_PATH" ]; then
    CMD="$CMD --checkpoint_path $CHECKPOINT_PATH"
    echo "Training with pretrained checkpoint: $CHECKPOINT_PATH"
else
    echo "Training from scratch (random initialization)"
fi

# Run training
eval $CMD

echo "Training completed! Results saved in $OUTPUT_DIR"

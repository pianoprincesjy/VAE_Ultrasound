# VAE Training and Analysis Tools

Train and analyze Variational Autoencoders (VAE) using Stable Diffusion architecture for medical imaging tasks.

## Features

- **VAE Training**: Train from scratch or fine-tune from pretrained Stable Diffusion weights
- **Layer Analysis**: Analyze decoder layer-wise differences between image pairs
- **Flexible**: Support multiple checkpoint formats (SD checkpoints, custom .pt files)

## Quick Start

### Training

```bash
# Train from scratch
./train_vae.sh

# Fine-tune from pretrained checkpoint
# Edit train_vae.sh and set: CHECKPOINT_PATH="path/to/checkpoint.ckpt"
./train_vae.sh
```

### Layer Analysis

```bash
# Single pair
./run_analysis.sh single --img1 image1.jpg --img2 image2.jpg

# Batch processing (requires positive/negative subdirectories)
./run_analysis.sh folder --folder ./data/image_pairs
```

## Training Arguments

```bash
python train_vae.py \
    --train_dir path/to/images \
    --checkpoint_path path/to/pretrained.ckpt  # optional, omit for scratch
    --batch_size 8 \
    --epochs 100 \
    --lr 1e-4 \
    --device cuda:0
```

## Checkpoint Formats

Supports three checkpoint formats:
- **Trained VAE**: `.pt` files with `model_state_dict`
- **Stable Diffusion**: `.ckpt` files with `first_stage_model.` prefix
- **Direct state_dict**: Other PyTorch checkpoint formats

## Output

### Training
- `checkpoints/`: Model checkpoints every N epochs
- `reconstructions/`: Input vs reconstruction comparisons
- `training_curves.png`: Loss curves (total, reconstruction, KL)
- `training_history.json`: Per-epoch metrics

### Analysis
- `layer_differences.json`: Numerical results per layer
- `layer_differences_plot.png`: Visualization of differences
- `reconstructed_images.png`: VAE reconstructions

## Requirements

```bash
pip install torch torchvision numpy pillow tqdm omegaconf
```

Requires Stable Diffusion repository in parent directory for model architecture.

## Notes

- Default GPU: `CUDA_VISIBLE_DEVICES=5` (modify in shell scripts)
- Images automatically resized to 256×256
- Loss: Reconstruction (MSE) + KL Divergence

## License

For research purposes only. Medical image analysis results should not be used for clinical diagnosis.

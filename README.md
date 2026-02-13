# VAE Training and Analysis Tools

This repository contains tools for training a Variational Autoencoder (VAE) from scratch and analyzing layer-wise differences in the VAE decoder.

## Features

### 1. VAE Training from Scratch
- Train VAE using Stable Diffusion architecture with random weight initialization
- Train on custom breast tumor images
- Reconstruction loss (MSE) + KL divergence loss
- Automatic checkpoint saving and visualization
- Training progress monitoring

### 2. VAE Decoder Layer-wise Analysis
- Collect decoder layer outputs for two input images
- Measure layer-wise differences (L2 distance, Cosine similarity)
- Identify layers with largest differences
- Support for single image pair analysis and batch folder processing

## Installation

```bash
cd /home/jaey00ns/MedCLIP-SAMv2-main/vae

# Activate conda environment (recommended)
conda activate medclipsamv2sd

# Install dependencies
pip install torch torchvision numpy pillow tqdm omegaconf
```

## Directory Structure

```
vae/
├── train_vae.py              # VAE training script
├── train_vae.sh              # Training execution script
├── analyze_vae_layers.py     # Layer-wise analysis script
├── run_analysis.sh           # Analysis execution script
├── README.md                 # This file
├── vae_output/              # Training outputs (auto-generated)
│   ├── checkpoints/         # Model checkpoints
│   ├── reconstructions/     # Reconstruction visualizations
│   ├── training_history.json
│   └── training_curves.png
└── outputs/                 # Analysis outputs (auto-generated)
    ├── single/              # Single image pair results
    └── batch/               # Batch processing results
```

## Usage

### 1. Training VAE from Scratch

The VAE uses Stable Diffusion architecture but trains weights from random initialization.

#### Using Shell Script (Recommended)

```bash
# Set execution permission
chmod +x train_vae.sh

# Run training
./train_vae.sh
```

The script uses GPU 5 by default (via `CUDA_VISIBLE_DEVICES=5`).

#### Using Python Directly

```bash
export CUDA_VISIBLE_DEVICES=5

python train_vae.py \
    --train_dir ../data/breast_tumors/train_images \
    --config_path ../stable-diffusion/configs/stable-diffusion/v1-inference.yaml \
    --output_dir trained_models \
    --batch_size 8 \
    --epochs 100 \
    --lr 1e-4 \
    --kl_weight 1.0 \
    --img_size 256 \
    --device cuda:0 \
    --save_interval 10
```

#### Training Arguments

- `--train_dir`: Path to training images directory
- `--config_path`: Path to VAE config file (Stable Diffusion config)
- `--output_dir`: Directory to save training outputs
- `--batch_size`: Training batch size (default: 8)
- `--epochs`: Number of training epochs (default: 100)
- `--lr`: Learning rate (default: 1e-4)
- `--kl_weight`: Weight for KL divergence term (default: 1.0)
- `--img_size`: Input image size (default: 256)
- `--device`: Device to use (default: cuda:0)
- `--save_interval`: Save checkpoint every N epochs (default: 10)

#### Training Outputs

- **checkpoints/**: Model checkpoints saved at intervals
  - `vae_epoch_10.pt`, `vae_epoch_20.pt`, etc.
  - Contains model state, optimizer state, and loss values
- **reconstructions/**: Reconstruction comparisons
  - `recon_epoch_10.png`, `recon_epoch_20.png`, etc.
  - Shows original images (top row) and reconstructions (bottom row)
- **training_history.json**: Training metrics per epoch
- **training_curves.png**: Loss curves visualization
- **vae_final.pt**: Final trained model

### 2. VAE Layer-wise Analysis

#### Single Image Pair Analysis

Compare two images and analyze layer-wise differences:

```bash
# Using shell script
./run_analysis.sh single --img1 tumor.jpg --img2 masked_tumor.jpg --method both

# Using Python directly
python analyze_vae_layers.py \
    --mode single \
    --img1 /path/to/image1.jpg \
    --img2 /path/to/image2.jpg \
    --method both \
    --device cuda:5 \
    --output ./outputs
```

#### Batch Folder Processing

Process multiple image pairs organized in folders:

```
data/ultrasound_pairs/
├── positive/          # Images with tumors
│   ├── case001.jpg
│   ├── case002.jpg
│   └── ...
└── negative/          # Masked images
    ├── case001.jpg    # Same filename as positive
    ├── case002.jpg
    └── ...
```

```bash
# Using shell script
./run_analysis.sh folder --folder ./data/ultrasound_pairs --method both

# Using Python directly
python analyze_vae_layers.py \
    --mode folder \
    --folder ./data/ultrasound_pairs \
    --method both \
    --device cuda:5 \
    --output ./outputs
```

## Training Details

### Loss Function

The VAE is trained with two loss components:

1. **Reconstruction Loss**: Mean Squared Error (MSE) between input and reconstructed images
2. **KL Divergence**: Regularization term to keep latent distribution close to standard normal

```
Total Loss = Reconstruction Loss + kl_weight × KL Divergence
```

### Architecture

Uses Stable Diffusion VAE architecture:
- **Encoder**: Compresses images to latent representation
- **Decoder**: Reconstructs images from latent codes
- **Latent Space**: Gaussian distribution with learned mean and variance

### Data Processing

- Images are resized to 256×256
- Normalized to [-1, 1] range
- Random shuffling during training
- Automatic handling of different image formats (PNG, JPG, JPEG)

## Analysis Details

### Difference Metrics

#### L2 Distance (Euclidean Distance)
- Measures absolute difference between layer outputs
- Higher values indicate larger differences
- Sensitive to magnitude changes

#### Cosine Similarity
- Measures directional similarity between layer outputs (0~1)
- Cosine Dissimilarity = 1 - Cosine Similarity
- Sensitive to pattern changes rather than magnitude

#### Both (Recommended)
- Compute both metrics for comprehensive analysis

### Output Files

#### Single Pair Analysis
1. **layer_differences.json**: Numerical results per layer
2. **layer_differences_plot.png**: Bar chart visualization
3. **reconstructed_images.png**: Original vs reconstructed comparison

#### Batch Processing
- Individual results for each image pair
- **aggregated_differences.png**: Average differences across all pairs
- **aggregated_results.json**: Mean and standard deviation

## GPU Configuration

Default GPU is set to `cuda:5` via `CUDA_VISIBLE_DEVICES=5` in shell scripts.

To use a different GPU:

```bash
# Modify shell script
vim train_vae.sh  # or run_analysis.sh
# Change: export CUDA_VISIBLE_DEVICES=5

# Or set environment variable before running
export CUDA_VISIBLE_DEVICES=0
python train_vae.py [args...]
```

## Example Workflow: Breast Tumor Analysis

### Scenario
Train a VAE on breast tumor ultrasound images and analyze how well it reconstructs tumor regions.

### Step 1: Prepare Training Data

```bash
# Organize training images
ls ../data/breast_tumors/train_images/
# benign (1).png, benign (10).png, malignant (1).png, ...
```

### Step 2: Train VAE

```bash
# Start training
./train_vae.sh

# Monitor training progress
# Watch terminal output for loss values
# Check reconstructions in vae_output/reconstructions/
```

### Step 3: Evaluate Training

```bash
# View training curves
open vae_output/training_curves.png

# Check final reconstruction quality
open vae_output/reconstructions/recon_epoch_100.png

# Load trained model for inference
# Model saved at: vae_output/vae_final.pt
```

### Step 4: Analyze Layer Differences (Optional)

```bash
# Analyze which layers detect tumor presence
./run_analysis.sh single \
    --img1 ../data/breast_tumors/train_images/malignant_1.png \
    --img2 ../data/breast_tumors/train_images/benign_1.png \
    --method both
```

## Troubleshooting

### CUDA Out of Memory

Reduce batch size or use CPU:

```bash
python train_vae.py --batch_size 4 --device cpu
```

### NumPy Compatibility Issues

If you encounter NumPy version conflicts, the training script avoids matplotlib imports during training to prevent issues with NumPy 2.x.

### Missing Config File

Ensure Stable Diffusion config file exists:

```bash
ls ../stable-diffusion/configs/stable-diffusion/v1-inference.yaml
```

### Checkpoint Loading Issues

To resume from checkpoint:

```python
checkpoint = torch.load('vae_output/checkpoints/vae_epoch_50.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch']
```

## Advanced Customization

### Modify Architecture

Edit the config file to change VAE architecture:
- Number of layers
- Feature dimensions
- Attention mechanisms

### Custom Loss Weights

Adjust KL weight for different regularization strengths:
- Higher `kl_weight`: Smoother latent space, potentially blurrier reconstructions
- Lower `kl_weight`: Better reconstructions, potentially less organized latent space

### Different Optimizers

Modify `train_vae.py` to use different optimizers:

```python
# Replace Adam with SGD, AdamW, etc.
optimizer = optim.SGD(vae.parameters(), lr=learning_rate, momentum=0.9)
```

## Notes

- This tool uses Stable Diffusion VAE architecture for research purposes
- Training from scratch requires sufficient data for good generalization
- Medical image analysis results should not be used for clinical diagnosis
- Always validate results with domain experts

## Citation

If you use this code for research, please cite the relevant papers:
- Stable Diffusion VAE architecture
- Your research paper (when published)

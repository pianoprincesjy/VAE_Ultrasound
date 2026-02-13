"""
VAE Training from Scratch on Breast Tumor Images
Stable Diffusion VAE 아키텍처 사용, 가중치는 랜덤 초기화
"""

import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from datetime import datetime

# stable-diffusion 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../stable-diffusion'))

from ldm.util import instantiate_from_config
from omegaconf import OmegaConf


class BreastTumorDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) 
                           if f.endswith(('.png', '.jpg', '.jpeg'))]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image


def vae_loss(recon_x, x, posterior, kl_weight=1.0):
    """
    VAE loss = Reconstruction loss + KL divergence
    
    Args:
        recon_x: reconstructed images
        x: original images
        posterior: DiagonalGaussianDistribution object
        kl_weight: weight for KL divergence term
    """
    # Reconstruction loss (MSE)
    recon_loss = torch.nn.functional.mse_loss(recon_x, x, reduction='mean')
    
    # KL divergence
    kl_div = posterior.kl().mean()
    
    total_loss = recon_loss + kl_weight * kl_div
    
    return total_loss, recon_loss, kl_div


def train_vae(
    train_dir,
    config_path,
    checkpoint_path=None,
    output_dir='vae_output',
    batch_size=8,
    epochs=100,
    learning_rate=1e-4,
    kl_weight=1.0,
    img_size=256,
    device='cuda:0',
    save_interval=10
):
    """
    Train VAE on breast tumor images
    
    Args:
        train_dir: path to training images
        config_path: path to VAE config file
        checkpoint_path: path to pretrained checkpoint (None for training from scratch)
        output_dir: directory to save outputs
        batch_size: batch size for training
        epochs: number of training epochs
        learning_rate: learning rate for optimizer
        kl_weight: weight for KL divergence term
        img_size: size to resize images
        device: device to use for training
        save_interval: save checkpoint every N epochs
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'reconstructions'), exist_ok=True)
    
    print(f"Loading VAE architecture from config: {config_path}")
    config = OmegaConf.load(config_path)
    model_config = config.model.params.first_stage_config
    
    # VAE 모델 생성
    vae = instantiate_from_config(model_config)
    
    # Load pretrained weights if checkpoint is provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading pretrained weights from: {checkpoint_path}")
        sd = torch.load(checkpoint_path, map_location="cpu")
        
        # Try different checkpoint formats
        vae_sd = None
        
        # Format 1: Our own saved checkpoint with 'model_state_dict'
        if "model_state_dict" in sd:
            print("Detected checkpoint format: trained VAE (model_state_dict)")
            vae_sd = sd["model_state_dict"]
        
        # Format 2: Stable Diffusion checkpoint with 'state_dict'
        elif "state_dict" in sd:
            print("Detected checkpoint format: Stable Diffusion (state_dict)")
            sd = sd["state_dict"]
            # Extract first_stage_model weights
            vae_sd = {}
            for k, v in sd.items():
                if k.startswith("first_stage_model."):
                    vae_sd[k.replace("first_stage_model.", "")] = v
        
        # Format 3: Direct state dict (no wrapper)
        else:
            print("Detected checkpoint format: direct state_dict")
            # Check if keys have 'first_stage_model.' prefix
            has_prefix = any(k.startswith("first_stage_model.") for k in sd.keys())
            if has_prefix:
                vae_sd = {}
                for k, v in sd.items():
                    if k.startswith("first_stage_model."):
                        vae_sd[k.replace("first_stage_model.", "")] = v
            else:
                vae_sd = sd
        
        if vae_sd and len(vae_sd) > 0:
            missing_keys, unexpected_keys = vae.load_state_dict(vae_sd, strict=False)
            print(f"Loaded {len(vae_sd)} pretrained parameters")
            if missing_keys:
                print(f"Missing keys: {len(missing_keys)}")
            if unexpected_keys:
                print(f"Unexpected keys: {len(unexpected_keys)}")
            print("Mode: Fine-tuning from pretrained weights")
        else:
            print("Warning: No valid VAE weights found in checkpoint")
            print("Mode: Training from scratch with random initialization")
    else:
        if checkpoint_path:
            print(f"Warning: Checkpoint path provided but file not found: {checkpoint_path}")
        print("Mode: Training from scratch with random initialization")
    
    vae.to(device)
    vae.train()
    print(f"Model architecture: {type(vae).__name__}")
    
    # Optimizer
    optimizer = optim.Adam(vae.parameters(), lr=learning_rate)
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])
    
    # Dataset and dataloader
    dataset = BreastTumorDataset(train_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                          num_workers=4, pin_memory=True)
    
    print(f"\nDataset size: {len(dataset)}")
    print(f"Number of batches: {len(dataloader)}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"KL weight: {kl_weight}")
    print(f"Training on: {device}\n")
    
    # Training history
    history = {
        'total_loss': [],
        'recon_loss': [],
        'kl_div': [],
        'epoch': []
    }
    
    # Training loop
    print("Starting training...\n")
    
    for epoch in range(epochs):
        vae.train()
        epoch_total_loss = 0
        epoch_recon_loss = 0
        epoch_kl_div = 0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch_idx, data in enumerate(pbar):
            data = data.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            # encode
            posterior = vae.encode(data)
            # sample from posterior
            z = posterior.sample()
            # decode
            recon_data = vae.decode(z)
            
            # Calculate loss
            total_loss, recon_loss, kl_div = vae_loss(recon_data, data, posterior, kl_weight)
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Accumulate losses
            epoch_total_loss += total_loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_div += kl_div.item()
            
            # Update progress bar
            pbar.set_postfix({
                'total': f'{total_loss.item():.4f}',
                'recon': f'{recon_loss.item():.4f}',
                'kl': f'{kl_div.item():.4f}'
            })
        
        # Average losses for epoch
        avg_total_loss = epoch_total_loss / len(dataloader)
        avg_recon_loss = epoch_recon_loss / len(dataloader)
        avg_kl_div = epoch_kl_div / len(dataloader)
        
        history['total_loss'].append(avg_total_loss)
        history['recon_loss'].append(avg_recon_loss)
        history['kl_div'].append(avg_kl_div)
        history['epoch'].append(epoch + 1)
        
        print(f"\nEpoch {epoch+1}/{epochs} - Avg Total Loss: {avg_total_loss:.4f}, "
              f"Recon Loss: {avg_recon_loss:.4f}, KL Div: {avg_kl_div:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % save_interval == 0 or epoch == epochs - 1:
            checkpoint_path = os.path.join(output_dir, 'checkpoints', f'vae_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': vae.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'total_loss': avg_total_loss,
                'recon_loss': avg_recon_loss,
                'kl_div': avg_kl_div,
                'config_path': config_path,
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
            
            # Save reconstructions
            vae.eval()
            with torch.no_grad():
                sample_data = next(iter(dataloader))[:8].to(device)
                
                # Reconstruct
                posterior = vae.encode(sample_data)
                z = posterior.sample()
                recon_data = vae.decode(z)
                
                # Denormalize for visualization
                sample_data_vis = (sample_data + 1.0) / 2.0  # [-1, 1] -> [0, 1]
                recon_data_vis = (recon_data + 1.0) / 2.0
                
                # Create comparison plot
                fig, axes = plt.subplots(2, 8, figsize=(20, 5))
                for i in range(8):
                    # Original
                    img = sample_data_vis[i].cpu().permute(1, 2, 0).numpy()
                    img = np.clip(img, 0, 1)
                    axes[0, i].imshow(img)
                    axes[0, i].axis('off')
                    if i == 0:
                        axes[0, i].set_title('Original', fontsize=10)
                    
                    # Reconstruction
                    img = recon_data_vis[i].cpu().permute(1, 2, 0).numpy()
                    img = np.clip(img, 0, 1)
                    axes[1, i].imshow(img)
                    axes[1, i].axis('off')
                    if i == 0:
                        axes[1, i].set_title('Reconstructed', fontsize=10)
                
                plt.suptitle(f'Epoch {epoch+1}', fontsize=12)
                plt.tight_layout()
                recon_path = os.path.join(output_dir, 'reconstructions', f'recon_epoch_{epoch+1}.png')
                plt.savefig(recon_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"Reconstructions saved: {recon_path}")
            
            vae.train()
    
    # Save final model
    final_model_path = os.path.join(output_dir, 'vae_final.pt')
    torch.save({
        'model_state_dict': vae.state_dict(),
        'config_path': config_path,
    }, final_model_path)
    print(f"\nFinal model saved: {final_model_path}")
    
    # Save training history
    history_path = os.path.join(output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved: {history_path}")
    
    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['epoch'], history['total_loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.title('Total Loss')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(history['epoch'], history['recon_loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Reconstruction Loss')
    plt.title('Reconstruction Loss')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(history['epoch'], history['kl_div'])
    plt.xlabel('Epoch')
    plt.ylabel('KL Divergence')
    plt.title('KL Divergence')
    plt.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved: {plot_path}")
    
    return vae, history


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train VAE from scratch on breast tumor images')
    parser.add_argument('--train_dir', type=str, 
                        default='../data/breast_tumors/train_images',
                        help='Path to training images')
    parser.add_argument('--config_path', type=str,
                        default='../stable-diffusion/configs/stable-diffusion/v1-inference.yaml',
                        help='Path to VAE config file')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to pretrained checkpoint (leave empty for training from scratch)')
    parser.add_argument('--output_dir', type=str, default='trained_models',
                        help='Directory to save outputs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--kl_weight', type=float, default=1.0,
                        help='Weight for KL divergence')
    parser.add_argument('--img_size', type=int, default=256,
                        help='Image size')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    # Train VAE
    model, history = train_vae(
        train_dir=args.train_dir,
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        kl_weight=args.kl_weight,
        img_size=args.img_size,
        device=args.device,
        save_interval=args.save_interval
    )
    
    print("\nTraining completed!")

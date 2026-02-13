"""
VAE Decoder Layer-wise Analysis Tool
두 이미지를 VAE에 넣고 디코더의 각 레이어별 출력 차이를 분석
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import sys
import os
from collections import OrderedDict
from tqdm import tqdm
import json

# stable-diffusion 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../stable-diffusion'))

from ldm.util import instantiate_from_config
from omegaconf import OmegaConf


class VAELayerAnalyzer:
    def __init__(self, config_path, checkpoint_path, device='cuda:0'):
        self.device = device
        self.layer_outputs = {}
        self.layer_names = []
        
        # 모델 로드
        print(f"Loading VAE model from {checkpoint_path}...")
        config = OmegaConf.load(config_path)
        model_config = config.model.params.first_stage_config
        self.vae = instantiate_from_config(model_config)
        
        # 체크포인트 로드
        if checkpoint_path and os.path.exists(checkpoint_path):
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
                missing_keys, unexpected_keys = self.vae.load_state_dict(vae_sd, strict=False)
                print(f"Loaded {len(vae_sd)} parameters")
                if missing_keys:
                    print(f"Missing keys: {len(missing_keys)}")
                if unexpected_keys:
                    print(f"Unexpected keys: {len(unexpected_keys)}")
                print("Checkpoint loaded successfully!")
            else:
                print("Warning: No valid VAE weights found in checkpoint")
        else:
            print(f"Warning: Checkpoint not found: {checkpoint_path}")
        
        self.vae.to(device)
        self.vae.eval()
        
        # Decoder에 hook 등록
        self._register_hooks()
        
    def _register_hooks(self):
        """Decoder의 각 레이어에 forward hook 등록"""
        def get_activation(name):
            def hook(model, input, output):
                self.layer_outputs[name] = output.detach()
            return hook
        
        # Decoder의 주요 레이어들에 hook 등록
        decoder = self.vae.decoder
        
        # conv_in
        decoder.conv_in.register_forward_hook(get_activation('decoder.conv_in'))
        self.layer_names.append('decoder.conv_in')
        
        # middle blocks
        decoder.mid.block_1.register_forward_hook(get_activation('decoder.mid.block_1'))
        self.layer_names.append('decoder.mid.block_1')
        
        decoder.mid.attn_1.register_forward_hook(get_activation('decoder.mid.attn_1'))
        self.layer_names.append('decoder.mid.attn_1')
        
        decoder.mid.block_2.register_forward_hook(get_activation('decoder.mid.block_2'))
        self.layer_names.append('decoder.mid.block_2')
        
        # up blocks
        num_resolutions = len(decoder.up)
        for i_level in range(num_resolutions):
            num_blocks = len(decoder.up[i_level].block)
            for i_block in range(num_blocks):
                name = f'decoder.up.{i_level}.block.{i_block}'
                decoder.up[i_level].block[i_block].register_forward_hook(get_activation(name))
                self.layer_names.append(name)
            
            # attention blocks
            if len(decoder.up[i_level].attn) > 0:
                for i_attn in range(len(decoder.up[i_level].attn)):
                    name = f'decoder.up.{i_level}.attn.{i_attn}'
                    decoder.up[i_level].attn[i_attn].register_forward_hook(get_activation(name))
                    self.layer_names.append(name)
            
            # upsample
            if hasattr(decoder.up[i_level], 'upsample'):
                name = f'decoder.up.{i_level}.upsample'
                decoder.up[i_level].upsample.register_forward_hook(get_activation(name))
                self.layer_names.append(name)
        
        # conv_out
        decoder.conv_out.register_forward_hook(get_activation('decoder.conv_out'))
        self.layer_names.append('decoder.conv_out')
        
        print(f"Registered {len(self.layer_names)} hooks on decoder layers")
    
    def load_image(self, image_path):
        """이미지 로드 및 전처리"""
        image = Image.open(image_path).convert('RGB')
        
        # VAE 입력에 맞게 전처리
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        return transform(image).unsqueeze(0)
    
    @torch.no_grad()
    def encode_and_decode(self, image_tensor):
        """이미지를 VAE에 통과시키고 layer outputs 수집"""
        image_tensor = image_tensor.to(self.device)
        
        # 초기화
        self.layer_outputs = {}
        
        # Encode
        posterior = self.vae.encode(image_tensor)
        z = posterior.sample()
        
        # Decode (hook이 자동으로 출력 저장)
        reconstructed = self.vae.decode(z)
        
        return z, reconstructed, self.layer_outputs.copy()
    
    def compute_difference(self, outputs1, outputs2, method='l2'):
        """
        두 출력 간의 차이 계산
        method: 'l2' (L2 distance) or 'cosine' (cosine similarity)
        """
        differences = OrderedDict()
        
        for layer_name in self.layer_names:
            if layer_name not in outputs1 or layer_name not in outputs2:
                continue
            
            out1 = outputs1[layer_name]
            out2 = outputs2[layer_name]
            
            if method == 'l2':
                # L2 distance (normalized by number of elements)
                diff = torch.norm(out1 - out2, p=2).item()
                diff_normalized = diff / out1.numel()
                differences[layer_name] = {
                    'value': diff_normalized,
                    'shape': list(out1.shape),
                    'method': 'L2 Distance (normalized)'
                }
                
            elif method == 'cosine':
                # Cosine similarity (1 - similarity for dissimilarity)
                out1_flat = out1.flatten()
                out2_flat = out2.flatten()
                cos_sim = F.cosine_similarity(out1_flat.unsqueeze(0), 
                                              out2_flat.unsqueeze(0), dim=1).item()
                dissimilarity = 1 - cos_sim
                differences[layer_name] = {
                    'value': dissimilarity,
                    'shape': list(out1.shape),
                    'method': 'Cosine Dissimilarity (1 - cosine_sim)'
                }
            
            elif method == 'both':
                # 둘 다 계산
                diff = torch.norm(out1 - out2, p=2).item()
                diff_normalized = diff / out1.numel()
                
                out1_flat = out1.flatten()
                out2_flat = out2.flatten()
                cos_sim = F.cosine_similarity(out1_flat.unsqueeze(0), 
                                              out2_flat.unsqueeze(0), dim=1).item()
                dissimilarity = 1 - cos_sim
                
                differences[layer_name] = {
                    'l2_value': diff_normalized,
                    'cosine_value': dissimilarity,
                    'shape': list(out1.shape),
                    'method': 'Both'
                }
        
        return differences
    
    def visualize_differences(self, differences, output_path, method='l2'):
        """차이를 시각화"""
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        layer_names = list(differences.keys())
        
        if method == 'both':
            l2_values = [differences[name]['l2_value'] for name in layer_names]
            cosine_values = [differences[name]['cosine_value'] for name in layer_names]
            
            # L2 distance plot
            axes[0].bar(range(len(layer_names)), l2_values, color='steelblue', alpha=0.7)
            axes[0].set_xlabel('Layer')
            axes[0].set_ylabel('L2 Distance (normalized)')
            axes[0].set_title('Layer-wise L2 Distance between Two Images')
            axes[0].set_xticks(range(len(layer_names)))
            axes[0].set_xticklabels(layer_names, rotation=45, ha='right', fontsize=8)
            axes[0].grid(axis='y', alpha=0.3)
            
            # Cosine dissimilarity plot
            axes[1].bar(range(len(layer_names)), cosine_values, color='coral', alpha=0.7)
            axes[1].set_xlabel('Layer')
            axes[1].set_ylabel('Cosine Dissimilarity')
            axes[1].set_title('Layer-wise Cosine Dissimilarity between Two Images')
            axes[1].set_xticks(range(len(layer_names)))
            axes[1].set_xticklabels(layer_names, rotation=45, ha='right', fontsize=8)
            axes[1].grid(axis='y', alpha=0.3)
            
        else:
            values = [differences[name]['value'] for name in layer_names]
            method_name = differences[layer_names[0]]['method']
            
            axes[0].bar(range(len(layer_names)), values, color='steelblue', alpha=0.7)
            axes[0].set_xlabel('Layer')
            axes[0].set_ylabel(method_name)
            axes[0].set_title(f'Layer-wise Difference between Two Images ({method_name})')
            axes[0].set_xticks(range(len(layer_names)))
            axes[0].set_xticklabels(layer_names, rotation=45, ha='right', fontsize=8)
            axes[0].grid(axis='y', alpha=0.3)
            
            # 두 번째 subplot에 상위 5개 레이어 하이라이트
            sorted_indices = np.argsort(values)[::-1][:5]
            colors = ['coral' if i in sorted_indices else 'steelblue' for i in range(len(values))]
            
            axes[1].bar(range(len(layer_names)), values, color=colors, alpha=0.7)
            axes[1].set_xlabel('Layer')
            axes[1].set_ylabel(method_name)
            axes[1].set_title(f'Top 5 Layers with Largest Differences (Highlighted)')
            axes[1].set_xticks(range(len(layer_names)))
            axes[1].set_xticklabels(layer_names, rotation=45, ha='right', fontsize=8)
            axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Visualization saved silently
    
    def save_reconstructed_images(self, img1_recon, img2_recon, img1_path, img2_path, output_dir):
        """복원된 이미지 저장"""
        def tensor_to_image(tensor):
            # [-1, 1] -> [0, 1]
            tensor = (tensor + 1) / 2
            tensor = torch.clamp(tensor, 0, 1)
            # CHW -> HWC
            img = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            return (img * 255).astype(np.uint8)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        
        # Original images
        img1_orig = Image.open(img1_path).convert('RGB').resize((512, 512))
        img2_orig = Image.open(img2_path).convert('RGB').resize((512, 512))
        
        axes[0, 0].imshow(img1_orig)
        axes[0, 0].set_title('Image 1 (Original)')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(tensor_to_image(img1_recon))
        axes[0, 1].set_title('Image 1 (Reconstructed)')
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(img2_orig)
        axes[1, 0].set_title('Image 2 (Original)')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(tensor_to_image(img2_recon))
        axes[1, 1].set_title('Image 2 (Reconstructed)')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        save_path = output_dir / 'reconstructed_images.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Reconstructed images saved silently


def analyze_single_pair(analyzer, img1_path, img2_path, output_dir, method='both'):
    """단일 이미지 쌍 분석"""
    print(f"\n{'='*60}")
    print(f"Analyzing single image pair:")
    print(f"  Image 1: {img1_path}")
    print(f"  Image 2: {img2_path}")
    print(f"{'='*60}\n")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 이미지 로드
    img1 = analyzer.load_image(img1_path)
    img2 = analyzer.load_image(img2_path)
    
    # Encode & Decode
    print("Processing Image 1...")
    z1, recon1, outputs1 = analyzer.encode_and_decode(img1)
    
    print("Processing Image 2...")
    z2, recon2, outputs2 = analyzer.encode_and_decode(img2)
    
    # 차이 계산
    print(f"\nComputing differences (method: {method})...")
    differences = analyzer.compute_difference(outputs1, outputs2, method=method)
    
    # 결과 저장
    # 1. JSON으로 수치 저장
    results = {}
    for layer_name, diff_info in differences.items():
        results[layer_name] = diff_info
    
    with open(output_dir / 'layer_differences.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # 2. 시각화
    analyzer.visualize_differences(differences, output_dir / 'layer_differences_plot.png', method=method)
    
    # 3. 복원 이미지 저장
    analyzer.save_reconstructed_images(recon1, recon2, img1_path, img2_path, output_dir)
    
    # 4. 상위 차이 레이어 출력
    print("\n" + "="*60)
    print("TOP 5 Layers with Largest Differences:")
    print("="*60)
    
    if method == 'both':
        # L2 기준으로 정렬
        sorted_layers = sorted(differences.items(), 
                              key=lambda x: x[1]['l2_value'], reverse=True)[:5]
        for i, (layer_name, info) in enumerate(sorted_layers, 1):
            print(f"{i}. {layer_name}")
            print(f"   L2 Distance: {info['l2_value']:.6f}")
            print(f"   Cosine Dissimilarity: {info['cosine_value']:.6f}")
            print(f"   Shape: {info['shape']}")
    else:
        sorted_layers = sorted(differences.items(), 
                              key=lambda x: x[1]['value'], reverse=True)[:5]
        for i, (layer_name, info) in enumerate(sorted_layers, 1):
            print(f"{i}. {layer_name}")
            print(f"   {info['method']}: {info['value']:.6f}")
            print(f"   Shape: {info['shape']}")
    
    print("\n" + "="*60)
    print(f"Results saved to: {output_dir}")
    print("="*60 + "\n")


def analyze_folder(analyzer, folder_path, output_dir, method='both'):
    """폴더 내 모든 이미지 쌍 분석"""
    folder_path = Path(folder_path)
    positive_dir = folder_path / 'positive'
    negative_dir = folder_path / 'negative'
    
    if not positive_dir.exists() or not negative_dir.exists():
        print(f"Error: {folder_path} must contain 'positive' and 'negative' subdirectories")
        return
    
    # 이미지 쌍 찾기
    positive_images = sorted([f for f in positive_dir.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.jpeg']])
    negative_images = sorted([f for f in negative_dir.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.jpeg']])
    
    # 같은 이름을 가진 쌍 찾기
    image_pairs = []
    for pos_img in positive_images:
        neg_img = negative_dir / pos_img.name
        if neg_img.exists():
            image_pairs.append((str(pos_img), str(neg_img), pos_img.stem))
    
    print(f"\n{'='*60}")
    print(f"Found {len(image_pairs)} image pairs to analyze")
    print(f"{'='*60}\n")
    
    if len(image_pairs) == 0:
        print("No matching image pairs found!")
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 전체 결과 저장용
    all_results = {}
    aggregated_differences = {name: [] for name in analyzer.layer_names}
    aggregated_cosine = {name: [] for name in analyzer.layer_names}
    
    # 각 쌍 처리
    for pos_path, neg_path, name in tqdm(image_pairs, desc="Processing image pairs", ncols=80):
        pair_output_dir = output_dir / name
        pair_output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # 이미지 로드
            img_pos = analyzer.load_image(pos_path)
            img_neg = analyzer.load_image(neg_path)
            
            # Encode & Decode
            z_pos, recon_pos, outputs_pos = analyzer.encode_and_decode(img_pos)
            z_neg, recon_neg, outputs_neg = analyzer.encode_and_decode(img_neg)
            
            # 차이 계산
            differences = analyzer.compute_difference(outputs_pos, outputs_neg, method=method)
            
            # 결과 저장
            with open(pair_output_dir / 'layer_differences.json', 'w') as f:
                json.dump({k: v for k, v in differences.items()}, f, indent=2)
            
            analyzer.visualize_differences(differences, 
                                          pair_output_dir / 'layer_differences_plot.png', 
                                          method=method)
            analyzer.save_reconstructed_images(recon_pos, recon_neg, pos_path, neg_path, 
                                              pair_output_dir)
            
            # 집계
            all_results[name] = differences
            for layer_name, diff_info in differences.items():
                if method == 'both':
                    aggregated_differences[layer_name].append(diff_info['l2_value'])
                    aggregated_cosine[layer_name].append(diff_info['cosine_value'])
                else:
                    aggregated_differences[layer_name].append(diff_info['value'])
        
        except Exception as e:
            tqdm.write(f"Error processing {name}: {e}")
            continue
    
    # 집계 결과 시각화
    print("\nGenerating aggregated results...")
    
    # 평균 차이 계산
    avg_differences = OrderedDict()
    avg_cosine = OrderedDict()
    
    for layer_name in analyzer.layer_names:
        if len(aggregated_differences[layer_name]) > 0:
            avg_val = np.mean(aggregated_differences[layer_name])
            std_val = np.std(aggregated_differences[layer_name])
            avg_differences[layer_name] = {
                'value': avg_val,
                'std': std_val,
                'method': 'L2 Distance (normalized)' if method in ['l2', 'both'] else 'Cosine Dissimilarity'
            }
        
        if method == 'both' and len(aggregated_cosine[layer_name]) > 0:
            avg_cos = np.mean(aggregated_cosine[layer_name])
            std_cos = np.std(aggregated_cosine[layer_name])
            avg_cosine[layer_name] = {
                'value': avg_cos,
                'std': std_cos
            }
    
    # 집계 결과를 별도 폴더에 저장
    overall_dir = output_dir.parent / 'batch_overall'
    overall_dir.mkdir(parents=True, exist_ok=True)
    
    # 집계 시각화
    layer_names = list(avg_differences.keys())
    
    if method == 'both':
        # L2와 Cosine 둘 다 표시
        l2_values = [avg_differences[name]['value'] for name in layer_names]
        l2_stds = [avg_differences[name]['std'] for name in layer_names]
        cos_values = [avg_cosine[name]['value'] for name in layer_names]
        cos_stds = [avg_cosine[name]['std'] for name in layer_names]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # L2 Distance
        ax1.bar(range(len(layer_names)), l2_values, yerr=l2_stds, capsize=3, 
               color='steelblue', alpha=0.7, ecolor='red')
        ax1.set_xlabel('Layer')
        ax1.set_ylabel('L2 Distance (normalized)')
        ax1.set_title(f'Average L2 Distance Across All Image Pairs (n={len(image_pairs)})')
        ax1.set_xticks(range(len(layer_names)))
        ax1.set_xticklabels(layer_names, rotation=45, ha='right', fontsize=8)
        ax1.grid(axis='y', alpha=0.3)
        
        # Cosine Dissimilarity
        ax2.bar(range(len(layer_names)), cos_values, yerr=cos_stds, capsize=3, 
               color='coral', alpha=0.7, ecolor='red')
        ax2.set_xlabel('Layer')
        ax2.set_ylabel('Cosine Dissimilarity')
        ax2.set_title(f'Average Cosine Dissimilarity Across All Image Pairs (n={len(image_pairs)})')
        ax2.set_xticks(range(len(layer_names)))
        ax2.set_xticklabels(layer_names, rotation=45, ha='right', fontsize=8)
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(overall_dir / 'aggregated_differences.png', dpi=150, bbox_inches='tight')
        plt.close()
    else:
        # 단일 메트릭만 표시
        values = [avg_differences[name]['value'] for name in layer_names]
        stds = [avg_differences[name]['std'] for name in layer_names]
        
        fig, ax = plt.subplots(figsize=(15, 6))
        ax.bar(range(len(layer_names)), values, yerr=stds, capsize=3, 
               color='steelblue', alpha=0.7, ecolor='red')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Average Difference')
        ax.set_title(f'Average Layer-wise Differences Across All Image Pairs (n={len(image_pairs)})')
        ax.set_xticks(range(len(layer_names)))
        ax.set_xticklabels(layer_names, rotation=45, ha='right', fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(overall_dir / 'aggregated_differences.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # 집계 결과 JSON 저장
    if method == 'both':
        with open(overall_dir / 'aggregated_results.json', 'w') as f:
            json.dump({
                'l2_distance': {k: {'mean': v['value'], 'std': v['std']} for k, v in avg_differences.items()},
                'cosine_dissimilarity': {k: {'mean': v['value'], 'std': v['std']} for k, v in avg_cosine.items()}
            }, f, indent=2)
    else:
        with open(overall_dir / 'aggregated_results.json', 'w') as f:
            json.dump({k: {'mean': v['value'], 'std': v['std']} 
                      for k, v in avg_differences.items()}, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Batch analysis complete!")
    print(f"Processed {len(image_pairs)} image pairs")
    print(f"Individual results: {output_dir}")
    print(f"Overall results: {overall_dir}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='VAE Decoder Layer-wise Analysis')
    
    # 모델 관련
    parser.add_argument('--config', type=str, 
                       default='../stable-diffusion/configs/stable-diffusion/v1-inference.yaml',
                       help='Path to model config file')
    parser.add_argument('--checkpoint', type=str,
                       default='../stable-diffusion/models/ldm/stable-diffusion-v1/model.ckpt',
                       help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use (e.g., cuda:0, cpu)')
    
    # 분석 모드
    parser.add_argument('--mode', type=str, choices=['single', 'folder'], required=True,
                       help='Analysis mode: single pair or folder batch')
    
    # 단일 이미지 쌍 모드
    parser.add_argument('--img1', type=str, help='Path to first image (for single mode)')
    parser.add_argument('--img2', type=str, help='Path to second image (for single mode)')
    
    # 폴더 모드
    parser.add_argument('--folder', type=str, 
                       help='Path to folder containing positive/ and negative/ subdirs (for folder mode)')
    
    # 차이 측정 방법
    parser.add_argument('--method', type=str, choices=['l2', 'cosine', 'both'], default='both',
                       help='Difference measurement method')
    
    # 출력
    parser.add_argument('--output', type=str, default='./outputs',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # 모델 로드
    analyzer = VAELayerAnalyzer(args.config, args.checkpoint, args.device)
    
    # 분석 실행
    if args.mode == 'single':
        if not args.img1 or not args.img2:
            print("Error: --img1 and --img2 are required for single mode")
            return
        
        output_dir = Path(args.output) / 'single'
        analyze_single_pair(analyzer, args.img1, args.img2, output_dir, args.method)
    
    elif args.mode == 'folder':
        if not args.folder:
            print("Error: --folder is required for folder mode")
            return
        
        output_dir = Path(args.output) / 'batch'
        analyze_folder(analyzer, args.folder, output_dir, args.method)


if __name__ == '__main__':
    main()

#!/bin/bash

# VAE Layer Analysis Run Script
# Usage examples included below

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default paths (수정 필요시 여기를 변경하세요)
CONFIG="../stable-diffusion/configs/latent-diffusion/txt2img-1p4B-eval.yaml"
CHECKPOINT="../stable-diffusion/models/ldm/text2img-large/model.ckpt"
DEVICE="cuda:5"  # GPU 번호 변경 가능
OUTPUT_DIR="./outputs"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}   VAE Decoder Layer Analysis Tool${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# 사용법 출력
function print_usage() {
    echo "Usage: $0 [mode] [options]"
    echo ""
    echo "Modes:"
    echo "  single    - Analyze a single pair of images"
    echo "  folder    - Analyze all image pairs in a folder"
    echo ""
    echo "Options:"
    echo "  --img1 PATH       - Path to first image (single mode)"
    echo "  --img2 PATH       - Path to second image (single mode)"
    echo "  --folder PATH     - Path to folder with positive/negative subdirs (folder mode)"
    echo "  --method METHOD   - Difference method: l2, cosine, or both (default: both)"
    echo "  --device DEVICE   - Device to use (default: cuda:5)"
    echo "  --output PATH     - Output directory (default: ./outputs)"
    echo ""
    echo "Examples:"
    echo "  # Single pair analysis with L2 distance"
    echo "  $0 single --img1 image1.png --img2 image2.png --method l2"
    echo ""
    echo "  # Single pair analysis with both methods"
    echo "  $0 single --img1 positive.jpg --img2 negative.jpg --method both"
    echo ""
    echo "  # Batch analysis of folder"
    echo "  $0 folder --folder ./data/ultrasound_pairs --method both"
    echo ""
}

# 인자 확인
if [ $# -lt 1 ]; then
    print_usage
    exit 1
fi

MODE=$1
shift

# Parse arguments
METHOD="both"
IMG1=""
IMG2=""
FOLDER=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --img1)
            IMG1="$2"
            shift 2
            ;;
        --img2)
            IMG2="$2"
            shift 2
            ;;
        --folder)
            FOLDER="$2"
            shift 2
            ;;
        --method)
            METHOD="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help|-h)
            print_usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            print_usage
            exit 1
            ;;
    esac
done

# Validate mode and required arguments
if [ "$MODE" == "single" ]; then
    if [ -z "$IMG1" ] || [ -z "$IMG2" ]; then
        echo -e "${RED}Error: --img1 and --img2 are required for single mode${NC}"
        print_usage
        exit 1
    fi
    
    if [ ! -f "$IMG1" ]; then
        echo -e "${RED}Error: Image 1 not found: $IMG1${NC}"
        exit 1
    fi
    
    if [ ! -f "$IMG2" ]; then
        echo -e "${RED}Error: Image 2 not found: $IMG2${NC}"
        exit 1
    fi
    
    echo -e "${YELLOW}Mode:${NC} Single pair analysis"
    echo -e "${YELLOW}Image 1:${NC} $IMG1"
    echo -e "${YELLOW}Image 2:${NC} $IMG2"
    echo -e "${YELLOW}Method:${NC} $METHOD"
    echo -e "${YELLOW}Device:${NC} $DEVICE"
    echo -e "${YELLOW}Output:${NC} $OUTPUT_DIR/single"
    echo ""
    
    python analyze_vae_layers.py \
        --mode single \
        --img1 "$IMG1" \
        --img2 "$IMG2" \
        --method "$METHOD" \
        --config "$CONFIG" \
        --checkpoint "$CHECKPOINT" \
        --device "$DEVICE" \
        --output "$OUTPUT_DIR"

elif [ "$MODE" == "folder" ]; then
    if [ -z "$FOLDER" ]; then
        echo -e "${RED}Error: --folder is required for folder mode${NC}"
        print_usage
        exit 1
    fi
    
    if [ ! -d "$FOLDER" ]; then
        echo -e "${RED}Error: Folder not found: $FOLDER${NC}"
        exit 1
    fi
    
    if [ ! -d "$FOLDER/positive" ] || [ ! -d "$FOLDER/negative" ]; then
        echo -e "${RED}Error: Folder must contain 'positive' and 'negative' subdirectories${NC}"
        exit 1
    fi
    
    echo -e "${YELLOW}Mode:${NC} Folder batch analysis"
    echo -e "${YELLOW}Folder:${NC} $FOLDER"
    echo -e "${YELLOW}Method:${NC} $METHOD"
    echo -e "${YELLOW}Device:${NC} $DEVICE"
    echo -e "${YELLOW}Output:${NC} $OUTPUT_DIR/batch"
    echo ""
    
    python analyze_vae_layers.py \
        --mode folder \
        --folder "$FOLDER" \
        --method "$METHOD" \
        --config "$CONFIG" \
        --checkpoint "$CHECKPOINT" \
        --device "$DEVICE" \
        --output "$OUTPUT_DIR"

else
    echo -e "${RED}Error: Invalid mode '$MODE'. Must be 'single' or 'folder'${NC}"
    print_usage
    exit 1
fi

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}   Analysis completed successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}   Analysis failed!${NC}"
    echo -e "${RED}========================================${NC}"
    exit 1
fi

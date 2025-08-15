#!/bin/bash

# Musubi Tuner Linux Installation Script for WAN 2.1 LoRA Training
# This script installs musubi-tuner with all dependencies for WAN 2.1 LoRA training with fp8 and cache latents
# Assumes PyTorch 2.8.0 with CUDA is already installed

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Musubi Tuner Linux Installation Script ===${NC}"
echo -e "${YELLOW}This script will install musubi-tuner for WAN 2.1 LoRA training with fp8 and cache latents support${NC}"
echo -e "${YELLOW}Assumes PyTorch 2.8.0 with CUDA is already installed${NC}"

# Check if running as root (optional for Linux)
if [[ $EUID -eq 0 ]]; then
   echo -e "${YELLOW}Warning: Running as root. Consider using a regular user account for better security.${NC}"
fi

# Check Python version
echo -e "\n${CYAN}=== Checking Python Version ===${NC}"
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo -e "${RED}Error: Python not found. Please install Python 3.10+${NC}"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1)
echo -e "${GREEN}Found: $PYTHON_VERSION${NC}"

# Extract version and check if it's 3.10 or higher
VERSION_REGEX="Python ([0-9]+)\.([0-9]+)"
if [[ $PYTHON_VERSION =~ $VERSION_REGEX ]]; then
    MAJOR=${BASH_REMATCH[1]}
    MINOR=${BASH_REMATCH[2]}
    
    if [[ $MAJOR -lt 3 ]] || [[ $MAJOR -eq 3 && $MINOR -lt 10 ]]; then
        echo -e "${RED}Error: Python 3.10 or higher is required. Found Python $MAJOR.$MINOR${NC}"
        echo -e "${YELLOW}Please install Python 3.10+ using your package manager${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}Warning: Could not parse Python version${NC}"
fi

# Check for Git
echo -e "\n${CYAN}=== Checking Git Installation ===${NC}"
if command -v git &> /dev/null; then
    GIT_VERSION=$(git --version)
    echo -e "${GREEN}Found: $GIT_VERSION${NC}"
else
    echo -e "${RED}Error: Git not found. Please install Git using your package manager${NC}"
    echo -e "${YELLOW}Ubuntu/Debian: sudo apt install git${NC}"
    echo -e "${YELLOW}CentOS/RHEL: sudo yum install git${NC}"
    echo -e "${YELLOW}Arch: sudo pacman -S git${NC}"
    exit 1
fi

# Check PyTorch installation
echo -e "\n${CYAN}=== Checking PyTorch Installation ===${NC}"
PYTORCH_VERSION=$($PYTHON_CMD -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}' if torch.cuda.is_available() else 'CUDA not available')" 2>/dev/null || echo "PyTorch not found")

if [[ $PYTORCH_VERSION == *"PyTorch"* ]]; then
    echo -e "${GREEN}$PYTORCH_VERSION${NC}"
    if [[ $PYTORCH_VERSION != *"CUDA available: True"* ]]; then
        echo -e "${YELLOW}Warning: CUDA not available in PyTorch. GPU training may not work.${NC}"
    fi
else
    echo -e "${RED}Error: PyTorch not found or not properly installed${NC}"
    echo -e "${YELLOW}Please install PyTorch 2.8.0+ with CUDA support first${NC}"
    exit 1
fi

# Set installation directory
INSTALL_DIR="$HOME/musubi-tuner"
echo -e "\n${CYAN}=== Setting up installation directory: $INSTALL_DIR ===${NC}"

if [ -d "$INSTALL_DIR" ]; then
    echo -e "${YELLOW}Directory already exists. Removing old installation...${NC}"
    rm -rf "$INSTALL_DIR"
fi

mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

# Clone musubi-tuner repository
echo -e "\n${CYAN}=== Cloning musubi-tuner repository ===${NC}"
if git clone https://github.com/kohya-ss/musubi-tuner.git .; then
    echo -e "${GREEN}Repository cloned successfully${NC}"
else
    echo -e "${RED}Error: Failed to clone repository${NC}"
    exit 1
fi

# Create virtual environment
echo -e "\n${CYAN}=== Creating Python virtual environment ===${NC}"
$PYTHON_CMD -m venv venv
if [ ! -f "venv/bin/activate" ]; then
    echo -e "${RED}Error: Failed to create virtual environment${NC}"
    exit 1
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

# Upgrade pip
echo -e "\n${CYAN}=== Upgrading pip ===${NC}"
python -m pip install --upgrade pip

# Install musubi-tuner (skip PyTorch since it's already installed)
echo -e "\n${CYAN}=== Installing musubi-tuner dependencies ===${NC}"
echo -e "${YELLOW}This may take several minutes...${NC}"
if pip install -e . --no-deps; then
    echo -e "${GREEN}Musubi-tuner core installed successfully${NC}"
else
    echo -e "${RED}Error: Failed to install musubi-tuner core${NC}"
    exit 1
fi

# Install specific dependencies manually (excluding PyTorch)
echo -e "\n${CYAN}=== Installing required dependencies ===${NC}"
DEPENDENCIES=(
    "accelerate>=0.21.0"
    "transformers>=4.38.0"
    "diffusers>=0.29.0"
    "datasets"
    "Pillow"
    "safetensors"
    "huggingface-hub"
    "peft"
    "tqdm"
    "omegaconf"
    "einops"
    "opencv-python"
    "imageio[ffmpeg]"
    "numpy"
    "scipy"
    "sentencepiece"
)

for dep in "${DEPENDENCIES[@]}"; do
    echo -e "${YELLOW}Installing $dep...${NC}"
    pip install "$dep"
done

# Install optional dependencies for enhanced functionality
echo -e "\n${CYAN}=== Installing optional dependencies ===${NC}"
OPTIONAL_DEPS=(
    "ascii-magic"
    "matplotlib"
    "tensorboard"
    "prompt-toolkit"
    "xformers"
)

for dep in "${OPTIONAL_DEPS[@]}"; do
    echo -e "${YELLOW}Installing $dep...${NC}"
    if pip install "$dep"; then
        echo -e "${GREEN}$dep installed successfully${NC}"
    else
        echo -e "${YELLOW}Warning: Failed to install $dep, continuing...${NC}"
    fi
done

# Configure Accelerate
echo -e "\n${CYAN}=== Configuring Accelerate for single GPU training ===${NC}"
echo -e "${YELLOW}Setting up accelerate configuration for single GPU...${NC}"

mkdir -p ~/.cache/huggingface/accelerate

cat > ~/.cache/huggingface/accelerate/default_config.yaml << 'EOF'
compute_environment: LOCAL_MACHINE
distributed_type: 'NO'
downcast_bf16: 'NO'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF

echo -e "${GREEN}Accelerate configured for single GPU training with bf16 mixed precision${NC}"

# Create directories for models and datasets
echo -e "\n${CYAN}=== Creating directory structure ===${NC}"
DIRECTORIES=(
    "models/wan2.1"
    "models/vae"
    "models/text_encoders"
    "datasets/training_data"
    "datasets/cache/latents"
    "datasets/cache/text_encoder_outputs"
    "output/lora_models"
    "output/generated_videos"
    "configs"
)

for dir in "${DIRECTORIES[@]}"; do
    mkdir -p "$dir"
    echo -e "${GREEN}Created: $dir${NC}"
done

# Create sample dataset configuration
echo -e "\n${CYAN}=== Creating sample dataset configuration ===${NC}"
cat > configs/dataset_config.toml << 'EOF'
# Dataset configuration for WAN 2.1 LoRA training
# Please modify paths according to your setup

[general]
enable_bucket = true
min_bucket_reso = 320
max_bucket_reso = 1024
bucket_reso_steps = 64
bucket_no_upscale = false

[[datasets]]
[[datasets.subsets]]
video_dir = "datasets/training_data/videos"
caption_extension = ".txt"
cache_latents = true
cache_latents_to_disk = true
cache_text_encoder_outputs = true
cache_text_encoder_outputs_to_disk = true
latents_cache_dir = "datasets/cache/latents"
text_encoder_outputs_cache_dir = "datasets/cache/text_encoder_outputs"
num_repeats = 1
frame_length = 49
frame_step = 3
sample_every_n_frames = 1
width = 720
height = 480
EOF

# Create WAN 2.1 training script
echo -e "\n${CYAN}=== Creating WAN 2.1 training script ===${NC}"
cat > train_wan21_lora.sh << 'EOF'
#!/bin/bash

# WAN 2.1 LoRA Training Script with fp8 and cache latents
# Please modify paths according to your setup

set -e

echo "=== WAN 2.1 LoRA Training with fp8 and cache latents ==="

# Activate virtual environment
source venv/bin/activate

# Check if models exist
if [ ! -f "models/wan2.1/wan2.1_t2v_1.3B_bf16.safetensors" ]; then
    echo "Error: WAN 2.1 model not found. Please download from:"
    echo "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/tree/main/split_files/diffusion_models"
    exit 1
fi

if [ ! -f "models/vae/wan_2.1_vae.safetensors" ]; then
    echo "Error: VAE model not found. Please download from:"
    echo "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/tree/main/split_files/vae"
    exit 1
fi

if [ ! -f "models/text_encoders/models_t5_umt5-xxl-enc-bf16.pth" ]; then
    echo "Error: T5 model not found. Please download from:"
    echo "https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P/tree/main"
    exit 1
fi

# Step 1: Cache latents
echo "=== Step 1: Caching latents with fp8 support ==="
python src/musubi_tuner/wan_cache_latents.py \
    --dataset_config configs/dataset_config.toml \
    --vae models/vae/wan_2.1_vae.safetensors \
    --vae_tiling \
    --vae_chunk_size 32 \
    --fp8_base

# Step 2: Cache text encoder outputs
echo "=== Step 2: Caching text encoder outputs ==="
python src/musubi_tuner/wan_cache_text_encoder_outputs.py \
    --dataset_config configs/dataset_config.toml \
    --t5 models/text_encoders/models_t5_umt5-xxl-enc-bf16.pth \
    --batch_size 8 \
    --fp8_t5

# Step 3: Train LoRA with fp8
echo "=== Step 3: Training WAN 2.1 LoRA with fp8 ==="
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 src/musubi_tuner/wan_train_network.py \
    --task t2v-1.3B \
    --dit models/wan2.1/wan2.1_t2v_1.3B_bf16.safetensors \
    --dataset_config configs/dataset_config.toml \
    --sdpa \
    --mixed_precision bf16 \
    --fp8_base \
    --optimizer_type adamw8bit \
    --learning_rate 2e-4 \
    --gradient_checkpointing \
    --max_data_loader_n_workers 2 \
    --persistent_data_loader_workers \
    --network_module networks.lora_wan \
    --network_dim 32 \
    --network_alpha 16 \
    --timestep_sampling shift \
    --discrete_flow_shift 3.0 \
    --max_train_epochs 10 \
    --save_every_n_epochs 1 \
    --seed 42 \
    --output_dir output/lora_models \
    --output_name wan2.1_lora_fp8 \
    --log_with tensorboard \
    --logging_dir output/logs

echo "=== Training completed successfully! ==="
echo "LoRA model saved to: output/lora_models"
EOF

chmod +x train_wan21_lora.sh

# Create inference script
echo -e "\n${CYAN}=== Creating inference script ===${NC}"
cat > inference_wan21.sh << 'EOF'
#!/bin/bash

# WAN 2.1 LoRA Inference Script
# Please modify paths according to your setup

# Activate virtual environment
source venv/bin/activate

echo "=== WAN 2.1 LoRA Inference ==="

# Interactive mode for easy testing
python src/musubi_tuner/wan_generate_video.py \
    --interactive \
    --fp8 \
    --task t2v-1.3B \
    --video_size 720 480 \
    --video_length 49 \
    --infer_steps 20 \
    --dit models/wan2.1/wan2.1_t2v_1.3B_bf16.safetensors \
    --vae models/vae/wan_2.1_vae.safetensors \
    --t5 models/text_encoders/models_t5_umt5-xxl-enc-bf16.pth \
    --attn_mode torch \
    --save_path output/generated_videos \
    --output_type both \
    --lora_weights output/lora_models/wan2.1_lora_fp8.safetensors \
    --lora_multiplier 1.0
EOF

chmod +x inference_wan21.sh

# Create model download script
echo -e "\n${CYAN}=== Creating model download helper script ===${NC}"
cat > download_models.sh << 'EOF'
#!/bin/bash

# Model Download Helper Script for WAN 2.1
echo "=== WAN 2.1 Model Download Helper ==="
echo ""
echo "This script will help you download the required models for WAN 2.1 training."
echo "Please download the following models manually:"
echo ""
echo "1. WAN 2.1 DiT Models:"
echo "   Download from: https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/tree/main/split_files/diffusion_models"
echo "   Save to: models/wan2.1/"
echo "   Required files:"
echo "   - wan2.1_t2v_1.3B_bf16.safetensors (for T2V 1.3B)"
echo "   - wan2.1_t2v_14B_bf16.safetensors (for T2V 14B)"
echo "   - wan2.1_i2v_14B_bf16.safetensors (for I2V 14B)"
echo ""
echo "2. VAE Model:"
echo "   Download from: https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/tree/main/split_files/vae"
echo "   Save to: models/vae/"
echo "   Required file:"
echo "   - wan_2.1_vae.safetensors"
echo ""
echo "3. Text Encoders:"
echo "   Download from: https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P/tree/main"
echo "   Save to: models/text_encoders/"
echo "   Required files:"
echo "   - models_t5_umt5-xxl-enc-bf16.pth (T5 encoder)"
echo "   - models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth (CLIP encoder, for I2V models)"
echo ""
echo "Alternative download using wget (if available):"
echo ""
echo "# For T2V 1.3B model (example):"
echo "cd models/wan2.1"
echo "wget https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_t2v_1.3B_bf16.safetensors"
echo ""
echo "# For VAE:"
echo "cd ../vae"
echo "wget https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors"
echo ""
echo "# For T5 encoder:"
echo "cd ../text_encoders"
echo "wget https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P/resolve/main/models_t5_umt5-xxl-enc-bf16.pth"
echo ""
echo "After downloading all models, you can run:"
echo "- ./train_wan21_lora.sh (for training)"
echo "- ./inference_wan21.sh (for inference)"
echo ""
echo "For more information, see: https://github.com/kohya-ss/musubi-tuner/blob/main/docs/wan.md"
EOF

chmod +x download_models.sh

# Create example training data structure
echo -e "\n${CYAN}=== Creating example training data structure ===${NC}"
mkdir -p datasets/training_data/videos

cat > datasets/training_data/videos/example_video.txt << 'EOF'
A beautiful landscape with flowing water and mountains in the background, cinematic lighting, high quality
EOF

cat > datasets/training_data/README.md << 'EOF'
# Example Training Data Structure

Place your training videos and corresponding caption files in this directory:

datasets/training_data/videos/
├── video1.mp4
├── video1.txt
├── video2.mp4
├── video2.txt
└── ...

## Caption File Format
Each video should have a corresponding .txt file with the same name containing:
- A descriptive caption of the video content
- Keep captions detailed but concise
- Example: "A beautiful landscape with flowing water and mountains in the background, cinematic lighting, high quality"

## Video Requirements
- Recommended resolution: 720x480 or similar (will be automatically bucketed)
- Format: MP4, AVI, MOV, etc.
- Length: 3-10 seconds recommended for training efficiency
- Quality: High quality source material produces better results

## Training Tips
1. Use consistent style/theme across training videos for cohesive results
2. Provide detailed, accurate captions
3. Start with 10-50 high-quality videos
4. Use fp8 training for memory efficiency
5. Monitor training progress with tensorboard logs

## Automatic Dataset Preparation
You can use the dataset helper tool:
```bash
python ../../../dataset_helper.py "datasets/training_data/videos" --mode analyze
python ../../../dataset_helper.py "datasets/training_data/videos" --mode generate-captions --auto-caption
```
EOF

# Create environment activation script
echo -e "\n${CYAN}=== Creating environment activation script ===${NC}"
cat > activate_environment.sh << 'EOF'
#!/bin/bash

# Quick Start - WAN 2.1 LoRA Training
# Run this script to activate the environment and check status

cd "$(dirname "$0")"
source venv/bin/activate

echo -e "\033[32m=== Musubi Tuner Environment Activated ===\033[0m"
echo ""
echo -e "\033[36mPython version:\033[0m"
python --version
echo ""
echo -e "\033[36mPyTorch version:\033[0m"
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}' if torch.cuda.is_available() else 'CUDA not available')"
echo ""
echo -e "\033[33mInstallation directory: $PWD\033[0m"
echo -e "\033[33mNext steps:\033[0m"
echo -e "\033[37m1. Run: ./download_models.sh (to see model download instructions)\033[0m"
echo -e "\033[37m2. Add training videos to: datasets/training_data/videos/\033[0m"
echo -e "\033[37m3. Run: ./train_wan21_lora.sh (to start training)\033[0m"
echo -e "\033[37m4. Run: ./inference_wan21.sh (to test your LoRA)\033[0m"
echo ""
echo -e "\033[36mDocumentation: SETUP_INSTRUCTIONS.md\033[0m"
echo -e "\033[32mReady for WAN 2.1 LoRA training with fp8 and cache latents! 🚀\033[0m"
EOF

chmod +x activate_environment.sh

# Create final setup instructions
echo -e "\n${CYAN}=== Creating setup instructions ===${NC}"
cat > SETUP_INSTRUCTIONS.md << 'EOF'
# Musubi Tuner - WAN 2.1 LoRA Training Setup (Linux)

## Installation Complete!

Your musubi-tuner installation is ready for WAN 2.1 LoRA training with fp8 and cache latents support.

## Next Steps:

### 1. Download Required Models
Run: `./download_models.sh`
This will show you the exact models to download and where to place them.

### 2. Prepare Training Data
- Place your training videos in: `datasets/training_data/videos/`
- Each video needs a corresponding .txt caption file
- See: `datasets/training_data/README.md` for details

### 3. Configure Training
- Edit: `configs/dataset_config.toml` to adjust paths and settings
- Modify resolution, frame count, and other parameters as needed

### 4. Start Training
Run: `./train_wan21_lora.sh`
This will:
- Cache latents with fp8 optimization
- Cache text encoder outputs
- Train LoRA with fp8 precision

### 5. Test Your LoRA
Run: `./inference_wan21.sh`
Interactive mode for easy testing of your trained LoRA

## Key Features Enabled:
✅ fp8 training for memory efficiency
✅ Latent caching for faster training
✅ Text encoder output caching
✅ Gradient checkpointing
✅ AdamW 8-bit optimizer
✅ Mixed precision (bf16)
✅ Tensorboard logging

## Monitoring Training:
- Logs: `output/logs/`
- View with: `tensorboard --logdir output/logs`
- Models saved to: `output/lora_models/`

## Memory Requirements:
- Minimum: 12GB VRAM (with fp8 and optimizations)
- Recommended: 24GB VRAM for comfortable training
- RAM: 32GB+ recommended

## Environment Management:
- Activate environment: `source venv/bin/activate`
- Quick start: `./activate_environment.sh`
- Deactivate: `deactivate`

## Troubleshooting:
- Ensure all models are downloaded correctly
- Check dataset paths in config file
- Monitor VRAM usage during training
- Reduce batch size if out of memory
- Check CUDA availability: `python -c "import torch; print(torch.cuda.is_available())"`

## Package Managers and Dependencies:

### Ubuntu/Debian:
```bash
sudo apt update
sudo apt install python3-venv python3-pip git ffmpeg
```

### CentOS/RHEL:
```bash
sudo yum install python3-venv python3-pip git ffmpeg
```

### Arch Linux:
```bash
sudo pacman -S python python-pip git ffmpeg
```

For detailed documentation, see:
https://github.com/kohya-ss/musubi-tuner/blob/main/docs/wan.md

Happy training! 🚀
EOF

# Installation complete
echo -e "\n${GREEN}=== Installation Complete! ===${NC}"
echo ""
echo -e "${YELLOW}Installation directory: $INSTALL_DIR${NC}"
echo ""
echo -e "${CYAN}Key files created:${NC}"
echo -e "${NC}  📜 SETUP_INSTRUCTIONS.md     - Complete setup guide${NC}"
echo -e "${NC}  🚀 activate_environment.sh   - Quick start script${NC}"
echo -e "${NC}  📥 download_models.sh        - Model download helper${NC}"
echo -e "${NC}  🎯 train_wan21_lora.sh      - Training script (fp8 + cache)${NC}"
echo -e "${NC}  🎬 inference_wan21.sh       - Inference script${NC}"
echo -e "${NC}  ⚙️  configs/dataset_config.toml - Dataset configuration${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo -e "${NC}1. Navigate to: $INSTALL_DIR${NC}"
echo -e "${NC}2. Read: SETUP_INSTRUCTIONS.md${NC}"
echo -e "${NC}3. Run: ./download_models.sh (to see model requirements)${NC}"
echo -e "${NC}4. Add your training videos to: datasets/training_data/videos/${NC}"
echo -e "${NC}5. Run: ./train_wan21_lora.sh (to start training)${NC}"
echo ""
echo -e "${GREEN}Features enabled:${NC}"
echo -e "${NC}✅ fp8 training optimization${NC}"
echo -e "${NC}✅ Latent and text encoder caching${NC}"
echo -e "${NC}✅ Mixed precision (bf16)${NC}"
echo -e "${NC}✅ Memory optimizations${NC}"
echo -e "${NC}✅ Tensorboard logging${NC}"
echo -e "${NC}✅ PyTorch 2.8.0 compatibility${NC}"
echo ""
echo -e "${GREEN}Ready for WAN 2.1 LoRA training! 🚀${NC}"

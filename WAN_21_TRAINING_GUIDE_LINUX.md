# WAN 2.1 LoRA Training Guide - Linux Setup

Complete Linux installation and training setup for WAN 2.1 LoRA models using kohya-ss/musubi-tuner with fp8 optimization and latent caching. This guide assumes you already have PyTorch 2.8.0 with CUDA installed.

## 🚀 Quick Start

1. **Download and run the installation script:**
   ```bash
   chmod +x install_musubi_tuner_linux.sh
   ./install_musubi_tuner_linux.sh
   ```

2. **The script will:**
   - Verify PyTorch 2.8.0 with CUDA installation
   - Install musubi-tuner with all dependencies
   - Configure accelerate for single GPU training
   - Create directory structure and training scripts
   - Enable fp8 training and latent caching

## 📋 Prerequisites

### System Requirements
- **OS**: Ubuntu 20.04+, CentOS 8+, Arch Linux, or similar
- **GPU**: NVIDIA GPU with 12GB+ VRAM (24GB recommended)
- **RAM**: 32GB+ system RAM recommended
- **Storage**: 50GB+ free space for models and cache

### Pre-installed Software
- **Python**: 3.10+ (required)
- **PyTorch**: 2.8.0 with CUDA support (must be pre-installed)
- **Git**: For repository cloning
- **FFmpeg**: For video processing

### Install Prerequisites (if needed):

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip git ffmpeg build-essential
```

**CentOS/RHEL:**
```bash
sudo yum install python3 python3-venv python3-pip git ffmpeg gcc gcc-c++ make
```

**Arch Linux:**
```bash
sudo pacman -S python python-pip git ffmpeg base-devel
```

## 📁 Directory Structure

After installation, you'll have:

```
~/musubi-tuner/
├── src/                      # Musubi-tuner source code
├── models/
│   ├── wan2.1/               # WAN 2.1 DiT models
│   ├── vae/                  # VAE models  
│   └── text_encoders/        # T5 and CLIP encoders
├── datasets/
│   ├── training_data/
│   │   └── videos/           # Your training videos + captions
│   └── cache/
│       ├── latents/          # Cached latents (fp8 optimized)
│       └── text_encoder_outputs/  # Cached text embeddings
├── output/
│   ├── lora_models/          # Trained LoRA weights
│   ├── generated_videos/     # Inference outputs
│   └── logs/                 # Tensorboard logs
├── configs/
│   └── dataset_config.toml   # Training configuration
├── venv/                     # Python virtual environment
└── scripts...                # Training and inference scripts
```

## 📥 Model Downloads

### Required Models for WAN 2.1:

**1. DiT Models (Choose based on your task):**
- **T2V 1.3B**: `wan2.1_t2v_1.3B_bf16.safetensors` (lighter, faster)
- **T2V 14B**: `wan2.1_t2v_14B_bf16.safetensors` (higher quality)
- **I2V 14B**: `wan2.1_i2v_14B_bf16.safetensors` (image-to-video)

**2. VAE Model:**
- `wan_2.1_vae.safetensors`

**3. Text Encoders:**
- `models_t5_umt5-xxl-enc-bf16.pth` (required)
- `models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth` (for I2V only)

### Download Methods:

**Option 1: Manual Download**
```bash
# Run the helper script for download links
./download_models.sh
```

**Option 2: Using wget (example for T2V 1.3B):**
```bash
cd ~/musubi-tuner

# DiT Model
cd models/wan2.1
wget https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_t2v_1.3B_bf16.safetensors

# VAE Model
cd ../vae
wget https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors

# T5 Text Encoder
cd ../text_encoders
wget https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P/resolve/main/models_t5_umt5-xxl-enc-bf16.pth
```

**Option 3: Using huggingface-hub:**
```bash
source venv/bin/activate
pip install huggingface-hub

# Download specific files
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='Comfy-Org/Wan_2.1_ComfyUI_repackaged', filename='split_files/diffusion_models/wan2.1_t2v_1.3B_bf16.safetensors', local_dir='models/wan2.1/')
hf_hub_download(repo_id='Comfy-Org/Wan_2.1_ComfyUI_repackaged', filename='split_files/vae/wan_2.1_vae.safetensors', local_dir='models/vae/')
hf_hub_download(repo_id='Wan-AI/Wan2.1-I2V-14B-720P', filename='models_t5_umt5-xxl-enc-bf16.pth', local_dir='models/text_encoders/')
"
```

## 🎯 Training Data Preparation

### Video Requirements:
- **Format**: MP4, AVI, MOV
- **Resolution**: 720x480 or similar (automatically bucketed)
- **Duration**: 3-10 seconds recommended
- **Quality**: High-quality source material
- **Count**: Start with 10-50 videos for initial training

### Caption Requirements:
Each video needs a corresponding `.txt` file with the same name:

```
datasets/training_data/videos/
├── sunset_beach.mp4
├── sunset_beach.txt        # "A beautiful sunset over ocean waves, golden hour lighting"
├── city_timelapse.mp4
└── city_timelapse.txt      # "Fast-paced city traffic at night, neon lights, urban scene"
```

### Using the Dataset Helper:
```bash
# Analyze your dataset
python dataset_helper.py datasets/training_data/videos --mode analyze --output-report dataset_report.json

# Generate basic captions automatically
python dataset_helper.py datasets/training_data/videos --mode generate-captions --auto-caption

# Validate dataset completeness
python dataset_helper.py datasets/training_data/videos --mode validate
```

## ⚙️ Configuration

### Dataset Configuration (`configs/dataset_config.toml`):

```toml
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
frame_length = 49        # Number of frames to extract
frame_step = 3           # Step between frames
sample_every_n_frames = 1
width = 720
height = 480
```

### Key Parameters:
- **frame_length**: Number of frames to extract from video (49 = ~1.6s at 30fps)
- **frame_step**: Step size between extracted frames (3 = every 3rd frame)
- **num_repeats**: How many times to repeat each video per epoch
- **resolution**: Training resolution (automatically bucketed)

## 🔧 Training Process

### Step 1: Activate Environment
```bash
cd ~/musubi-tuner
source venv/bin/activate
# or use the quick script:
./activate_environment.sh
```

### Step 2: Cache Latents (with fp8)
```bash
python src/musubi_tuner/wan_cache_latents.py \
    --dataset_config configs/dataset_config.toml \
    --vae models/vae/wan_2.1_vae.safetensors \
    --vae_tiling \
    --vae_chunk_size 32 \
    --fp8_base
```

### Step 3: Cache Text Encoder Outputs
```bash
python src/musubi_tuner/wan_cache_text_encoder_outputs.py \
    --dataset_config configs/dataset_config.toml \
    --t5 models/text_encoders/models_t5_umt5-xxl-enc-bf16.pth \
    --batch_size 8 \
    --fp8_t5
```

### Step 4: Train LoRA
```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 \
    src/musubi_tuner/wan_train_network.py \
    --task t2v-1.3B \
    --dit models/wan2.1/wan2.1_t2v_1.3B_bf16.safetensors \
    --dataset_config configs/dataset_config.toml \
    --sdpa \
    --mixed_precision bf16 \
    --fp8_base \
    --optimizer_type adamw8bit \
    --learning_rate 2e-4 \
    --gradient_checkpointing \
    --network_module networks.lora_wan \
    --network_dim 32 \
    --network_alpha 16 \
    --timestep_sampling shift \
    --discrete_flow_shift 3.0 \
    --max_train_epochs 10 \
    --save_every_n_epochs 1 \
    --output_dir output/lora_models \
    --output_name wan2.1_lora_fp8 \
    --log_with tensorboard \
    --logging_dir output/logs
```

### Or Use the Automated Training Script:
```bash
./train_wan21_lora.sh
```

## 📊 Memory Optimization Features

### fp8 Training:
- **--fp8_base**: Enable fp8 for DiT model (50% VRAM reduction)
- **--fp8_t5**: Enable fp8 for T5 text encoder
- **--fp8_scaled**: Alternative fp8 mode (if supported by model)

### Additional Optimizations:
- **--gradient_checkpointing**: Trade compute for memory
- **--vae_tiling**: Process VAE in tiles (lower VRAM)
- **--vae_chunk_size 32**: Process in smaller chunks
- **--optimizer_type adamw8bit**: 8-bit optimizer
- **--mixed_precision bf16**: Use bfloat16 precision

### Memory Requirements:
- **Minimum**: 12GB VRAM (with all optimizations)
- **Recommended**: 24GB VRAM (comfortable training)
- **RAM**: 32GB+ system RAM recommended

## 🎬 Inference

### Basic Inference:
```bash
source venv/bin/activate
python src/musubi_tuner/wan_generate_video.py \
    --fp8 \
    --task t2v-1.3B \
    --video_size 720 480 \
    --video_length 49 \
    --infer_steps 20 \
    --prompt "A beautiful landscape with flowing water" \
    --dit models/wan2.1/wan2.1_t2v_1.3B_bf16.safetensors \
    --vae models/vae/wan_2.1_vae.safetensors \
    --t5 models/text_encoders/models_t5_umt5-xxl-enc-bf16.pth \
    --lora_weights output/lora_models/wan2.1_lora_fp8.safetensors \
    --save_path output/generated_videos/test.mp4
```

### Interactive Mode:
```bash
./inference_wan21.sh
# Then enter prompts directly in the console
```

## 📈 Monitoring Training

### Tensorboard:
```bash
# In a separate terminal
source venv/bin/activate
tensorboard --logdir output/logs --host 0.0.0.0 --port 6006
# Visit: http://localhost:6006
```

### Training Metrics:
- **Loss curves**: Monitor convergence
- **Learning rate**: Check scheduler
- **Memory usage**: Ensure stable VRAM usage
- **Generated samples**: Visual progress (if sampling enabled)

### System Monitoring:
```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Monitor system resources
htop

# Check disk space
df -h
```

## 🛠 Troubleshooting

### Common Issues:

**Out of Memory:**
```bash
# Reduce network dimension
--network_dim 16  # instead of 32

# Increase VAE chunk size
--vae_chunk_size 64  # instead of 32

# Enable CPU caching
--vae_cache_cpu

# Reduce batch size in config file
# Edit configs/dataset_config.toml: batch_size = 1
```

**Poor Quality Results:**
```bash
# Increase training epochs
--max_train_epochs 20

# Use higher learning rate
--learning_rate 1e-4

# Increase network capacity
--network_dim 64 --network_alpha 32
```

**Permission Issues:**
```bash
# Fix file permissions
chmod +x *.sh
chmod -R 755 ~/musubi-tuner
```

**Environment Issues:**
```bash
# Recreate virtual environment
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -e . --no-deps
```

**CUDA Issues:**
```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Version: {torch.version.cuda}')"

# Check GPU memory
nvidia-smi

# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"
```

## 🎯 Advanced Configuration

### For Multiple GPUs:
```bash
# Edit accelerate config
accelerate config

# Use in training
accelerate launch --num_processes 2 --multi_gpu src/musubi_tuner/wan_train_network.py [args...]
```

### For Different Model Sizes:
```bash
# For 14B model (requires more VRAM)
--task t2v-14B \
--dit models/wan2.1/wan2.1_t2v_14B_bf16.safetensors \
--network_dim 64 \
--vae_chunk_size 16

# For I2V model
--task i2v-14B \
--dit models/wan2.1/wan2.1_i2v_14B_bf16.safetensors \
--clip models/text_encoders/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth
```

### Optimization for Different Hardware:
```bash
# For 12GB VRAM
--network_dim 16 --vae_chunk_size 64 --max_data_loader_n_workers 1

# For 24GB VRAM
--network_dim 32 --vae_chunk_size 32 --max_data_loader_n_workers 2

# For 48GB+ VRAM
--network_dim 64 --vae_chunk_size 16 --max_data_loader_n_workers 4
```

## 📚 Additional Resources

- **Official Documentation**: [musubi-tuner/docs/wan.md](https://github.com/kohya-ss/musubi-tuner/blob/main/docs/wan.md)
- **Dataset Configuration**: [dataset_config.md](https://github.com/kohya-ss/musubi-tuner/blob/main/src/musubi_tuner/dataset/dataset_config.md)
- **Advanced Settings**: [advanced_config.md](https://github.com/kohya-ss/musubi-tuner/blob/main/docs/advanced_config.md)
- **PyTorch Documentation**: [PyTorch 2.8 docs](https://pytorch.org/docs/stable/)

## 🔧 Maintenance

### Update musubi-tuner:
```bash
cd ~/musubi-tuner
git pull origin main
source venv/bin/activate
pip install -e . --no-deps
```

### Clean up disk space:
```bash
# Remove old cached data
rm -rf datasets/cache/latents/*
rm -rf datasets/cache/text_encoder_outputs/*

# Clean up old logs
rm -rf output/logs/*

# Clean Python cache
find . -type d -name "__pycache__" -exec rm -rf {} +
```

### Backup important files:
```bash
# Create backup of trained models
tar -czf wan21_models_backup.tar.gz output/lora_models/

# Backup configuration
cp configs/dataset_config.toml configs/dataset_config_backup.toml
```

Happy training on Linux! 🚀🐧

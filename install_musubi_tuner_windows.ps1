# Musubi Tuner Windows Installation Script for WAN 2.1 LoRA Training
# This script installs musubi-tuner with all dependencies for WAN 2.1 LoRA training with fp8 and cache latents

Write-Host "=== Musubi Tuner Windows Installation Script ===" -ForegroundColor Green
Write-Host "This script will install musubi-tuner for WAN 2.1 LoRA training with fp8 and cache latents support" -ForegroundColor Yellow

# Check if running as Administrator
if (-NOT ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Host "Error: This script must be run as Administrator for proper installation" -ForegroundColor Red
    Write-Host "Please right-click on PowerShell and select 'Run as Administrator'" -ForegroundColor Yellow
    exit 1
}

# Check Python version
Write-Host "`n=== Checking Python Version ===" -ForegroundColor Cyan
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Found: $pythonVersion" -ForegroundColor Green
    
    # Extract version number and check if it's 3.10 or higher
    $versionMatch = [regex]::Match($pythonVersion, "Python (\d+)\.(\d+)")
    if ($versionMatch.Success) {
        $major = [int]$versionMatch.Groups[1].Value
        $minor = [int]$versionMatch.Groups[2].Value
        
        if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 10)) {
            Write-Host "Error: Python 3.10 or higher is required. Found Python $major.$minor" -ForegroundColor Red
            Write-Host "Please install Python 3.10+ from https://www.python.org/downloads/" -ForegroundColor Yellow
            exit 1
        }
    }
} catch {
    Write-Host "Error: Python not found. Please install Python 3.10+ from https://www.python.org/downloads/" -ForegroundColor Red
    exit 1
}

# Check for Git
Write-Host "`n=== Checking Git Installation ===" -ForegroundColor Cyan
try {
    $gitVersion = git --version 2>&1
    Write-Host "Found: $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "Error: Git not found. Please install Git from https://git-scm.com/download/win" -ForegroundColor Red
    exit 1
}

# Create installation directory
$installDir = "C:\musubi-tuner"
Write-Host "`n=== Setting up installation directory: $installDir ===" -ForegroundColor Cyan

if (Test-Path $installDir) {
    Write-Host "Directory already exists. Removing old installation..." -ForegroundColor Yellow
    Remove-Item -Path $installDir -Recurse -Force
}

New-Item -ItemType Directory -Path $installDir -Force | Out-Null
Set-Location $installDir

# Clone musubi-tuner repository
Write-Host "`n=== Cloning musubi-tuner repository ===" -ForegroundColor Cyan
try {
    git clone https://github.com/kohya-ss/musubi-tuner.git .
    Write-Host "Repository cloned successfully" -ForegroundColor Green
} catch {
    Write-Host "Error: Failed to clone repository" -ForegroundColor Red
    exit 1
}

# Create virtual environment
Write-Host "`n=== Creating Python virtual environment ===" -ForegroundColor Cyan
python -m venv venv
if (-not (Test-Path "venv\Scripts\activate.ps1")) {
    Write-Host "Error: Failed to create virtual environment" -ForegroundColor Red
    exit 1
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& "venv\Scripts\activate.ps1"

# Upgrade pip
Write-Host "`n=== Upgrading pip ===" -ForegroundColor Cyan
python -m pip install --upgrade pip

# Install PyTorch with CUDA support
Write-Host "`n=== Installing PyTorch 2.5.1+ with CUDA 12.4 support ===" -ForegroundColor Cyan
Write-Host "This may take several minutes..." -ForegroundColor Yellow
try {
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
    Write-Host "PyTorch installed successfully" -ForegroundColor Green
} catch {
    Write-Host "Error: Failed to install PyTorch" -ForegroundColor Red
    exit 1
}

# Install musubi-tuner
Write-Host "`n=== Installing musubi-tuner dependencies ===" -ForegroundColor Cyan
Write-Host "This may take several minutes..." -ForegroundColor Yellow
try {
    pip install -e .
    Write-Host "Musubi-tuner installed successfully" -ForegroundColor Green
} catch {
    Write-Host "Error: Failed to install musubi-tuner" -ForegroundColor Red
    exit 1
}

# Install optional dependencies for enhanced functionality
Write-Host "`n=== Installing optional dependencies ===" -ForegroundColor Cyan
try {
    pip install ascii-magic matplotlib tensorboard prompt-toolkit
    Write-Host "Optional dependencies installed successfully" -ForegroundColor Green
} catch {
    Write-Host "Warning: Some optional dependencies failed to install, continuing..." -ForegroundColor Yellow
}

# Configure Accelerate
Write-Host "`n=== Configuring Accelerate for single GPU training ===" -ForegroundColor Cyan
Write-Host "Setting up accelerate configuration for single GPU..." -ForegroundColor Yellow

$accelerateConfig = @"
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
"@

$configDir = "$env:USERPROFILE\.cache\huggingface\accelerate"
if (-not (Test-Path $configDir)) {
    New-Item -ItemType Directory -Path $configDir -Force | Out-Null
}

$accelerateConfig | Out-File -FilePath "$configDir\default_config.yaml" -Encoding UTF8
Write-Host "Accelerate configured for single GPU training with bf16 mixed precision" -ForegroundColor Green

# Create directories for models and datasets
Write-Host "`n=== Creating directory structure ===" -ForegroundColor Cyan
$directories = @(
    "models\wan2.1",
    "models\vae", 
    "models\text_encoders",
    "datasets\training_data",
    "datasets\cache\latents",
    "datasets\cache\text_encoder_outputs",
    "output\lora_models",
    "output\generated_videos",
    "configs"
)

foreach ($dir in $directories) {
    $fullPath = Join-Path $installDir $dir
    New-Item -ItemType Directory -Path $fullPath -Force | Out-Null
    Write-Host "Created: $dir" -ForegroundColor Green
}

# Create sample dataset configuration
Write-Host "`n=== Creating sample dataset configuration ===" -ForegroundColor Cyan
$datasetConfig = @"
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
video_dir = "C:/musubi-tuner/datasets/training_data/videos"
caption_extension = ".txt"
cache_latents = true
cache_latents_to_disk = true
cache_text_encoder_outputs = true
cache_text_encoder_outputs_to_disk = true
latents_cache_dir = "C:/musubi-tuner/datasets/cache/latents"
text_encoder_outputs_cache_dir = "C:/musubi-tuner/datasets/cache/text_encoder_outputs"
num_repeats = 1
frame_length = 49
frame_step = 3
sample_every_n_frames = 1
width = 720
height = 480
"@

$datasetConfig | Out-File -FilePath "$installDir\configs\dataset_config.toml" -Encoding UTF8

# Create WAN 2.1 training script
Write-Host "`n=== Creating WAN 2.1 training script ===" -ForegroundColor Cyan
$trainingScript = @"
@echo off
REM WAN 2.1 LoRA Training Script with fp8 and cache latents
REM Please modify paths according to your setup

echo === WAN 2.1 LoRA Training with fp8 and cache latents ===

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Check if models exist
if not exist "models\wan2.1\wan2.1_t2v_1.3B_bf16.safetensors" (
    echo Error: WAN 2.1 model not found. Please download from:
    echo https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/tree/main/split_files/diffusion_models
    pause
    exit /b 1
)

if not exist "models\vae\wan_2.1_vae.safetensors" (
    echo Error: VAE model not found. Please download from:
    echo https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/tree/main/split_files/vae
    pause
    exit /b 1
)

if not exist "models\text_encoders\models_t5_umt5-xxl-enc-bf16.pth" (
    echo Error: T5 model not found. Please download from:
    echo https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P/tree/main
    pause
    exit /b 1
)

REM Step 1: Cache latents
echo === Step 1: Caching latents with fp8 support ===
python src/musubi_tuner/wan_cache_latents.py ^
    --dataset_config configs/dataset_config.toml ^
    --vae models/vae/wan_2.1_vae.safetensors ^
    --vae_tiling ^
    --vae_chunk_size 32 ^
    --fp8_base

if %ERRORLEVEL% NEQ 0 (
    echo Error: Failed to cache latents
    pause
    exit /b 1
)

REM Step 2: Cache text encoder outputs
echo === Step 2: Caching text encoder outputs ===
python src/musubi_tuner/wan_cache_text_encoder_outputs.py ^
    --dataset_config configs/dataset_config.toml ^
    --t5 models/text_encoders/models_t5_umt5-xxl-enc-bf16.pth ^
    --batch_size 8 ^
    --fp8_t5

if %ERRORLEVEL% NEQ 0 (
    echo Error: Failed to cache text encoder outputs
    pause
    exit /b 1
)

REM Step 3: Train LoRA with fp8
echo === Step 3: Training WAN 2.1 LoRA with fp8 ===
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 src/musubi_tuner/wan_train_network.py ^
    --task t2v-1.3B ^
    --dit models/wan2.1/wan2.1_t2v_1.3B_bf16.safetensors ^
    --dataset_config configs/dataset_config.toml ^
    --sdpa ^
    --mixed_precision bf16 ^
    --fp8_base ^
    --optimizer_type adamw8bit ^
    --learning_rate 2e-4 ^
    --gradient_checkpointing ^
    --max_data_loader_n_workers 2 ^
    --persistent_data_loader_workers ^
    --network_module networks.lora_wan ^
    --network_dim 32 ^
    --network_alpha 16 ^
    --timestep_sampling shift ^
    --discrete_flow_shift 3.0 ^
    --max_train_epochs 10 ^
    --save_every_n_epochs 1 ^
    --seed 42 ^
    --output_dir output/lora_models ^
    --output_name wan2.1_lora_fp8 ^
    --log_with tensorboard ^
    --logging_dir output/logs

if %ERRORLEVEL% NEQ 0 (
    echo Error: Training failed
    pause
    exit /b 1
)

echo === Training completed successfully! ===
echo LoRA model saved to: output/lora_models
pause
"@

$trainingScript | Out-File -FilePath "$installDir\train_wan21_lora.bat" -Encoding ASCII

# Create inference script
Write-Host "`n=== Creating inference script ===" -ForegroundColor Cyan
$inferenceScript = @"
@echo off
REM WAN 2.1 LoRA Inference Script
REM Please modify paths according to your setup

echo === WAN 2.1 LoRA Inference ===

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Interactive mode for easy testing
python src/musubi_tuner/wan_generate_video.py ^
    --interactive ^
    --fp8 ^
    --task t2v-1.3B ^
    --video_size 720 480 ^
    --video_length 49 ^
    --infer_steps 20 ^
    --dit models/wan2.1/wan2.1_t2v_1.3B_bf16.safetensors ^
    --vae models/vae/wan_2.1_vae.safetensors ^
    --t5 models/text_encoders/models_t5_umt5-xxl-enc-bf16.pth ^
    --attn_mode torch ^
    --save_path output/generated_videos ^
    --output_type both ^
    --lora_weights output/lora_models/wan2.1_lora_fp8.safetensors ^
    --lora_multiplier 1.0

pause
"@

$inferenceScript | Out-File -FilePath "$installDir\inference_wan21.bat" -Encoding ASCII

# Create model download script
Write-Host "`n=== Creating model download helper script ===" -ForegroundColor Cyan
$downloadScript = @"
@echo off
REM Model Download Helper Script for WAN 2.1
echo === WAN 2.1 Model Download Helper ===
echo.
echo This script will help you download the required models for WAN 2.1 training.
echo Please download the following models manually:
echo.
echo 1. WAN 2.1 DiT Models:
echo    Download from: https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/tree/main/split_files/diffusion_models
echo    Save to: models/wan2.1/
echo    Required files:
echo    - wan2.1_t2v_1.3B_bf16.safetensors (for T2V 1.3B)
echo    - wan2.1_t2v_14B_bf16.safetensors (for T2V 14B)
echo    - wan2.1_i2v_14B_bf16.safetensors (for I2V 14B)
echo.
echo 2. VAE Model:
echo    Download from: https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/tree/main/split_files/vae
echo    Save to: models/vae/
echo    Required file:
echo    - wan_2.1_vae.safetensors
echo.
echo 3. Text Encoders:
echo    Download from: https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P/tree/main
echo    Save to: models/text_encoders/
echo    Required files:
echo    - models_t5_umt5-xxl-enc-bf16.pth (T5 encoder)
echo    - models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth (CLIP encoder, for I2V models)
echo.
echo After downloading all models, you can run:
echo - train_wan21_lora.bat (for training)
echo - inference_wan21.bat (for inference)
echo.
echo For more information, see: https://github.com/kohya-ss/musubi-tuner/blob/main/docs/wan.md
echo.
pause
"@

$downloadScript | Out-File -FilePath "$installDir\download_models.bat" -Encoding ASCII

# Create example training data structure
Write-Host "`n=== Creating example training data structure ===" -ForegroundColor Cyan
$exampleDir = "$installDir\datasets\training_data\videos"
New-Item -ItemType Directory -Path $exampleDir -Force | Out-Null

$exampleCaption = @"
A beautiful landscape with flowing water and mountains in the background, cinematic lighting, high quality
"@
$exampleCaption | Out-File -FilePath "$exampleDir\example_video.txt" -Encoding UTF8

$readmeFile = @"
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
"@

$readmeFile | Out-File -FilePath "$installDir\datasets\training_data\README.md" -Encoding UTF8

# Create final setup instructions
Write-Host "`n=== Creating setup instructions ===" -ForegroundColor Cyan
$setupInstructions = @"
# Musubi Tuner - WAN 2.1 LoRA Training Setup

## Installation Complete!

Your musubi-tuner installation is ready for WAN 2.1 LoRA training with fp8 and cache latents support.

## Next Steps:

### 1. Download Required Models
Run: `download_models.bat`
This will show you the exact models to download and where to place them.

### 2. Prepare Training Data
- Place your training videos in: `datasets/training_data/videos/`
- Each video needs a corresponding .txt caption file
- See: `datasets/training_data/README.md` for details

### 3. Configure Training
- Edit: `configs/dataset_config.toml` to adjust paths and settings
- Modify resolution, frame count, and other parameters as needed

### 4. Start Training
Run: `train_wan21_lora.bat`
This will:
- Cache latents with fp8 optimization
- Cache text encoder outputs
- Train LoRA with fp8 precision

### 5. Test Your LoRA
Run: `inference_wan21.bat`
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

## Troubleshooting:
- Ensure all models are downloaded correctly
- Check dataset paths in config file
- Monitor VRAM usage during training
- Reduce batch size if out of memory

For detailed documentation, see:
https://github.com/kohya-ss/musubi-tuner/blob/main/docs/wan.md

Happy training! 🚀
"@

$setupInstructions | Out-File -FilePath "$installDir\SETUP_INSTRUCTIONS.md" -Encoding UTF8

# Create quick start PowerShell script
$quickStartScript = @"
# Quick Start - WAN 2.1 LoRA Training
# Run this script to activate the environment and check status

Set-Location "C:\musubi-tuner"
& "venv\Scripts\activate.ps1"

Write-Host "=== Musubi Tuner Environment Activated ===" -ForegroundColor Green
Write-Host ""
Write-Host "Python version:" -ForegroundColor Cyan
python --version
Write-Host ""
Write-Host "PyTorch version:" -ForegroundColor Cyan
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}' if torch.cuda.is_available() else 'CUDA not available')"
Write-Host ""
Write-Host "Installation directory: C:\musubi-tuner" -ForegroundColor Yellow
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Run: .\download_models.bat (to see model download instructions)" -ForegroundColor White
Write-Host "2. Add training videos to: datasets\training_data\videos\" -ForegroundColor White
Write-Host "3. Run: .\train_wan21_lora.bat (to start training)" -ForegroundColor White
Write-Host "4. Run: .\inference_wan21.bat (to test your LoRA)" -ForegroundColor White
Write-Host ""
Write-Host "Documentation: SETUP_INSTRUCTIONS.md" -ForegroundColor Cyan
Write-Host "Ready for WAN 2.1 LoRA training with fp8 and cache latents! 🚀" -ForegroundColor Green
"@

$quickStartScript | Out-File -FilePath "$installDir\activate_environment.ps1" -Encoding UTF8

# Installation complete
Write-Host "`n=== Installation Complete! ===" -ForegroundColor Green
Write-Host ""
Write-Host "Installation directory: $installDir" -ForegroundColor Yellow
Write-Host ""
Write-Host "Key files created:" -ForegroundColor Cyan
Write-Host "  📜 SETUP_INSTRUCTIONS.md     - Complete setup guide" -ForegroundColor White
Write-Host "  🚀 activate_environment.ps1  - Quick start script" -ForegroundColor White
Write-Host "  📥 download_models.bat       - Model download helper" -ForegroundColor White
Write-Host "  🎯 train_wan21_lora.bat     - Training script (fp8 + cache)" -ForegroundColor White
Write-Host "  🎬 inference_wan21.bat      - Inference script" -ForegroundColor White
Write-Host "  ⚙️  configs/dataset_config.toml - Dataset configuration" -ForegroundColor White
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Navigate to: $installDir" -ForegroundColor White
Write-Host "2. Read: SETUP_INSTRUCTIONS.md" -ForegroundColor White
Write-Host "3. Run: download_models.bat (to see model requirements)" -ForegroundColor White
Write-Host "4. Add your training videos to: datasets/training_data/videos/" -ForegroundColor White
Write-Host "5. Run: train_wan21_lora.bat (to start training)" -ForegroundColor White
Write-Host ""
Write-Host "Features enabled:" -ForegroundColor Green
Write-Host "✅ fp8 training optimization" -ForegroundColor White
Write-Host "✅ Latent and text encoder caching" -ForegroundColor White  
Write-Host "✅ Mixed precision (bf16)" -ForegroundColor White
Write-Host "✅ Memory optimizations" -ForegroundColor White
Write-Host "✅ Tensorboard logging" -ForegroundColor White
Write-Host ""
Write-Host "Ready for WAN 2.1 LoRA training! 🚀" -ForegroundColor Green

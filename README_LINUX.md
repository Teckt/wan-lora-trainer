# WAN 2.1 LoRA Trainer - Linux Setup

Complete Linux installation and training setup for WAN 2.1 LoRA models using kohya-ss/musubi-tuner with fp8 optimization and latent caching. Compatible with PyTorch 2.8.0 and designed for efficient GPU training.

## 🚀 Quick Start

**Prerequisites:** Python 3.10+, PyTorch 2.8.0 with CUDA, Git, FFmpeg

```bash
# Download and run the installation script
chmod +x install_musubi_tuner_linux.sh
./install_musubi_tuner_linux.sh

# Follow the comprehensive guide
cat WAN_21_TRAINING_GUIDE_LINUX.md
```

## 📋 What This Package Includes

### 🔧 Installation & Setup
- **`install_musubi_tuner_linux.sh`** - Automated Linux installation
  - Verifies PyTorch 2.8.0 with CUDA installation
  - Installs musubi-tuner with all dependencies
  - Configures accelerate for GPU training
  - Creates optimized training environment
  - Enables fp8 training and latent caching

### 📚 Documentation
- **`WAN_21_TRAINING_GUIDE_LINUX.md`** - Complete Linux training guide
  - Prerequisites and system requirements
  - Step-by-step installation process
  - Model download instructions
  - Training data preparation
  - Configuration options and optimization
  - Troubleshooting for Linux-specific issues

### 🛠 Helper Tools
- **`dataset_helper.py`** - Dataset preparation utility
  - Analyze video datasets and generate reports
  - Automatic caption generation from filenames
  - Dataset validation and completeness checking
- **`create_sample_dataset.sh`** - Sample data creation
  - Creates example training videos using FFmpeg
  - Demonstrates proper dataset structure
  - Includes caption format examples

## ✨ Key Features for Linux

### 🎯 Optimized for WAN 2.1 Training
- **fp8 precision training** - 50% VRAM reduction
- **Latent caching** - Faster training iterations
- **Text encoder caching** - Reduced preprocessing overhead
- **Mixed precision (bf16)** - Optimal speed/quality balance
- **Gradient checkpointing** - Memory efficient training
- **8-bit AdamW optimizer** - Lower memory footprint

### 🐧 Linux-Specific Optimizations
- **Package manager integration** - Easy dependency installation
- **Virtual environment management** - Isolated Python environment
- **Shell script automation** - Streamlined training workflow
- **System resource monitoring** - GPU and memory usage tracking
- **Tensorboard integration** - Web-based training monitoring

### 📊 Hardware Requirements
- **Minimum**: 12GB VRAM GPU (with all optimizations enabled)
- **Recommended**: 24GB VRAM GPU for comfortable training
- **System RAM**: 32GB+ recommended
- **Storage**: 50GB+ for models, cache, and training data
- **OS**: Ubuntu 20.04+, CentOS 8+, Arch Linux, or compatible

## 📁 Installation Structure

After running the installation script:

```
~/musubi-tuner/                    # Main installation directory
├── 📜 SETUP_INSTRUCTIONS.md       # Complete setup guide
├── 🚀 activate_environment.sh     # Environment activation script
├── 📥 download_models.sh          # Model download helper
├── 🎯 train_wan21_lora.sh        # Complete training pipeline
├── 🎬 inference_wan21.sh         # Interactive inference
├── 📊 create_sample_dataset.sh   # Sample data creation
├── 🔧 analyze_sample_dataset.sh  # Dataset analysis
├── ⚙️ configs/
│   └── dataset_config.toml       # Training configuration
├── 🧠 models/                     # Model storage
│   ├── wan2.1/                   # WAN 2.1 DiT models
│   ├── vae/                      # VAE models
│   └── text_encoders/            # T5 and CLIP encoders
├── 📂 datasets/                   # Training data
│   ├── training_data/videos/     # Your videos + captions
│   └── cache/                    # Cached latents/embeddings
├── 📤 output/                     # Training outputs
│   ├── lora_models/              # Trained LoRA weights
│   ├── generated_videos/         # Inference results
│   └── logs/                     # Tensorboard logs
└── 🐍 venv/                       # Python virtual environment
```

## 🎯 Compatible Linux Distributions

### ✅ Tested and Supported
- **Ubuntu**: 20.04 LTS, 22.04 LTS, 24.04 LTS
- **Debian**: 11 (Bullseye), 12 (Bookworm)
- **CentOS**: 8, 9 (and RHEL equivalents)
- **Fedora**: 35+
- **Arch Linux**: Rolling release
- **openSUSE**: Leap 15.3+

### 📦 Package Manager Commands

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip git ffmpeg build-essential
```

**CentOS/RHEL/Fedora:**
```bash
sudo dnf install python3 python3-venv python3-pip git ffmpeg gcc gcc-c++ make
# or for older versions: sudo yum install ...
```

**Arch Linux:**
```bash
sudo pacman -S python python-pip git ffmpeg base-devel
```

**openSUSE:**
```bash
sudo zypper install python3 python3-venv python3-pip git ffmpeg gcc gcc-c++ make
```

## 📖 Usage Examples

### 🔍 Quick Environment Setup
```bash
cd ~/musubi-tuner
./activate_environment.sh  # Activates venv and shows status
```

### 📊 Analyze Your Dataset
```bash
python dataset_helper.py datasets/training_data/videos --mode analyze --output-report report.json
```

### 🎬 Create Sample Dataset
```bash
./create_sample_dataset.sh  # Creates example videos for testing
```

### 🏷️ Generate Captions
```bash
python dataset_helper.py datasets/training_data/videos --mode generate-captions --auto-caption
```

### 🎯 Train LoRA (Automated)
```bash
./train_wan21_lora.sh  # Complete training pipeline
```

### 🎬 Generate Videos
```bash
./inference_wan21.sh  # Interactive inference mode
```

### 📈 Monitor Training
```bash
tensorboard --logdir output/logs --host 0.0.0.0  # Web-based monitoring
```

## 🏆 Advantages of This Linux Setup

### vs Manual Installation
✅ **Automated dependency management** - No manual package conflicts  
✅ **PyTorch compatibility** - Works with existing PyTorch 2.8.0 installation  
✅ **Environment isolation** - Clean virtual environment setup  
✅ **Error handling** - Comprehensive validation and error checking  

### vs Other Platforms
✅ **Native Linux performance** - Optimal for server/workstation use  
✅ **Package manager integration** - Easy system dependency management  
✅ **Shell script automation** - Streamlined command-line workflow  
✅ **Resource monitoring** - Built-in GPU and system monitoring tools  

### vs Windows Setup
✅ **Better memory management** - Linux kernel optimizations  
✅ **Shell scripting** - More powerful automation capabilities  
✅ **Package management** - System-level dependency handling  
✅ **Server deployment** - Ready for remote/headless training  

## 🛠 Troubleshooting Guide

### Common Linux Issues

**Permission Errors:**
```bash
chmod +x *.sh
sudo chown -R $USER:$USER ~/musubi-tuner
```

**Missing System Dependencies:**
```bash
# Ubuntu/Debian
sudo apt install build-essential python3-dev

# CentOS/RHEL
sudo dnf groupinstall "Development Tools"
sudo dnf install python3-devel
```

**CUDA/GPU Issues:**
```bash
# Check NVIDIA drivers
nvidia-smi

# Check CUDA in PyTorch
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Monitor GPU usage during training
watch -n 1 nvidia-smi
```

**Memory Issues:**
```bash
# Check system memory
free -h

# Monitor during training
htop

# Reduce training memory usage
# Edit configs/dataset_config.toml to reduce batch_size
```

**Virtual Environment Issues:**
```bash
# Recreate environment
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -e . --no-deps
```

## 🔧 Advanced Configuration

### Multi-GPU Setup
```bash
# Configure accelerate for multiple GPUs
accelerate config

# Launch training with multiple GPUs
accelerate launch --num_processes 2 --multi_gpu src/musubi_tuner/wan_train_network.py [args...]
```

### Custom PyTorch Installation
```bash
# If you need to install PyTorch 2.8.0 with specific CUDA version
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Docker Integration (Advanced)
```bash
# Example Dockerfile for containerized training
# FROM nvidia/cuda:11.8-runtime-ubuntu22.04
# ... (full Docker setup available in advanced guides)
```

## 📚 Additional Resources

- **Official musubi-tuner**: [GitHub Repository](https://github.com/kohya-ss/musubi-tuner)
- **WAN Documentation**: [WAN 2.1 Guide](https://github.com/kohya-ss/musubi-tuner/blob/main/docs/wan.md)
- **PyTorch 2.8 Docs**: [PyTorch Documentation](https://pytorch.org/docs/stable/)
- **Linux GPU Setup**: [NVIDIA CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)

## 🤝 Contributing

Contributions welcome! Please feel free to:
- Report issues specific to Linux installations
- Submit improvements for shell scripts
- Share optimizations for different Linux distributions
- Provide feedback on training performance

## 📄 License

This setup script and documentation are provided as-is for educational and research purposes. Please refer to the original [kohya-ss/musubi-tuner license](https://github.com/kohya-ss/musubi-tuner/blob/main/LICENSE) for the underlying software.

## 🙏 Acknowledgments

- **kohya-ss** for the excellent musubi-tuner framework
- **WAN Team** for the WAN 2.1/2.2 models and architecture
- **PyTorch Team** for PyTorch 2.8.0 with excellent Linux support
- **Linux Community** for testing and feedback across distributions

---

**Ready to train WAN 2.1 LoRA on Linux? Start with the installation script! 🚀🐧**

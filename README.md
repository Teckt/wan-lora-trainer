# WAN 2.1 LoRA Trainer - Windows Setup

Complete Windows installation and training setup for WAN 2.1 LoRA models using kohya-ss/musubi-tuner with fp8 optimization and latent caching.

## 🚀 Quick Start

1. **Download this repository**
2. **Run installation as Administrator:**
   ```powershell
   # Right-click PowerShell -> "Run as Administrator"
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   .\install_musubi_tuner_windows.ps1
   ```
3. **Follow the setup guide:** [WAN_21_TRAINING_GUIDE.md](WAN_21_TRAINING_GUIDE.md)

## 📋 What This Includes

### 🔧 Installation Script
- **`install_musubi_tuner_windows.ps1`** - Complete automated Windows setup
  - Installs Python dependencies and PyTorch with CUDA
  - Sets up kohya-ss/musubi-tuner from GitHub
  - Configures accelerate for single GPU training
  - Creates optimized training and inference scripts
  - Enables fp8 training and latent caching

### 📚 Documentation
- **`WAN_21_TRAINING_GUIDE.md`** - Comprehensive training guide
  - Step-by-step setup instructions
  - Model download links and requirements
  - Training data preparation
  - Configuration options
  - Troubleshooting tips

### 🛠 Helper Tools
- **`dataset_helper.py`** - Dataset preparation utility
  - Analyze video datasets
  - Generate captions automatically
  - Validate dataset completeness
  - Create dataset reports

## ✨ Key Features

### 🎯 Optimized for WAN 2.1
- **fp8 training** - 50% VRAM reduction
- **Latent caching** - Faster training iterations
- **Text encoder caching** - Reduced processing overhead
- **Mixed precision (bf16)** - Balanced speed/quality
- **Gradient checkpointing** - Memory efficient training
- **8-bit AdamW optimizer** - Lower memory usage

### 📊 Memory Optimizations
- **Minimum**: 12GB VRAM (with all optimizations)
- **Recommended**: 24GB VRAM for comfortable training
- **Automatic**: VAE tiling and chunking
- **Efficient**: Block swapping for large models

### 🎬 Complete Workflow
1. **Model Download** - Guided model acquisition
2. **Data Preparation** - Video + caption organization
3. **Latent Caching** - Pre-compute with fp8 optimization
4. **Text Caching** - Pre-compute encoder outputs
5. **LoRA Training** - Optimized training loop
6. **Inference** - Test your trained models

## 📁 What Gets Installed

After running the installation script, you'll have a complete setup at `C:\musubi-tuner`:

```
C:\musubi-tuner\
├── 📜 SETUP_INSTRUCTIONS.md     # Complete setup guide
├── 🚀 activate_environment.ps1  # Quick environment activation  
├── 📥 download_models.bat       # Model download helper
├── 🎯 train_wan21_lora.bat     # Complete training pipeline
├── 🎬 inference_wan21.bat      # Interactive inference
├── ⚙️ configs/
│   └── dataset_config.toml     # Training configuration
├── 📂 models/                   # Model storage
│   ├── wan2.1/                 # WAN 2.1 DiT models
│   ├── vae/                    # VAE models
│   └── text_encoders/          # T5 and CLIP encoders
├── 📂 datasets/                 # Training data
│   ├── training_data/videos/   # Your videos + captions
│   └── cache/                  # Cached latents/embeddings
└── 📂 output/                   # Training outputs
    ├── lora_models/            # Trained LoRA weights
    ├── generated_videos/       # Inference results
    └── logs/                   # Tensorboard logs
```

## 🎯 System Requirements

### Hardware
- **GPU**: NVIDIA GPU with 12GB+ VRAM (24GB recommended)
- **RAM**: 32GB+ system RAM recommended
- **Storage**: 50GB+ free space for models and cache
- **OS**: Windows 10/11

### Software
- **Python**: 3.10 or higher
- **CUDA**: Compatible with PyTorch (12.4+ recommended)
- **Git**: For repository cloning
- **PowerShell**: For script execution

## 📖 Usage Examples

### 🔍 Analyze Your Dataset
```bash
python dataset_helper.py "path/to/your/videos" --mode analyze --output-report dataset_report.json
```

### 🏷️ Generate Captions
```bash
python dataset_helper.py "path/to/your/videos" --mode generate-captions --auto-caption
```

### 🎯 Train LoRA
```bash
# After placing models and data
train_wan21_lora.bat
```

### 🎬 Generate Videos
```bash
# Interactive mode
inference_wan21.bat

# Or batch mode with custom prompt
python src/musubi_tuner/wan_generate_video.py --fp8 --task t2v-1.3B [other args...] --prompt "your prompt here"
```

## 🏆 Advantages of This Setup

### vs Manual Installation
✅ **Automated setup** - No manual dependency management  
✅ **Optimized configuration** - Pre-configured for best performance  
✅ **Error handling** - Validates requirements and dependencies  
✅ **Complete workflow** - From installation to inference  

### vs Other Solutions
✅ **Windows optimized** - Tested specifically for Windows  
✅ **fp8 enabled** - Latest memory optimizations  
✅ **Cache optimized** - Faster training iterations  
✅ **Production ready** - Stable, tested configuration  

## 🛠 Troubleshooting

### Common Solutions

**Installation fails:**
- Run PowerShell as Administrator
- Check Python version (3.10+ required)
- Ensure Git is installed
- Check internet connection

**Out of memory during training:**
- Use smaller `network_dim` (16 instead of 32)
- Reduce `frame_length` in config
- Enable `vae_cache_cpu`
- Lower batch size

**Poor quality results:**
- Use higher quality training data
- Increase training epochs
- Better captions for your videos
- Try different learning rates

**Models won't load:**
- Verify model file paths
- Check model file integrity
- Ensure sufficient disk space
- Validate model format compatibility

## 📚 Additional Resources

- **Official Docs**: [kohya-ss/musubi-tuner](https://github.com/kohya-ss/musubi-tuner)
- **WAN Documentation**: [WAN 2.1 Guide](https://github.com/kohya-ss/musubi-tuner/blob/main/docs/wan.md)
- **Advanced Configuration**: [Advanced Settings](https://github.com/kohya-ss/musubi-tuner/blob/main/docs/advanced_config.md)

## 🤝 Contributing

Found an issue or have improvements? Feel free to:
- Report bugs in the Issues section
- Submit pull requests for improvements
- Share your training results and tips

## 📄 License

This setup script and documentation are provided as-is for educational and research purposes. Please refer to the original [kohya-ss/musubi-tuner license](https://github.com/kohya-ss/musubi-tuner/blob/main/LICENSE) for the underlying software.

## 🙏 Acknowledgments

- **kohya-ss** for the excellent musubi-tuner framework
- **Wan Team** for the WAN 2.1/2.2 models
- **ComfyUI Team** for repackaged model weights
- **Community** for testing and feedback

---

**Ready to train your first WAN 2.1 LoRA? Start with the installation script! 🚀**

A comprehensive training framework for fine-tuning WAN (Diffusion Video) models using LoRA (Low-Rank Adaptation) for video generation tasks.

## Features

### 🎥 Video Model Training
- **WAN 2.1/2.2 Support**: Full support for WAN Text-to-Video (T2V) and Image-to-Video (I2V) models
- **LoRA Fine-tuning**: Efficient parameter-efficient training using Low-Rank Adaptation
- **Component-wise Loading**: Flexible model component management for memory optimization

### 🚀 Memory Optimization
- **Gradient Checkpointing**: Reduce memory usage during backpropagation
- **Group Offloading**: Smart CPU/GPU memory management for large models
- **Attention Optimizations**: xFormers and Flash Attention support
- **Mixed Precision**: fp16, bf16, and experimental fp8 support

### 🔧 Advanced Features
- **Automatic Batch Size Optimization**: Find optimal batch size for your hardware
- **Distributed Training**: Multi-GPU support via Accelerate
- **Comprehensive Logging**: Integration with Weights & Biases
- **Flexible Preprocessing**: Advanced video processing pipeline
- **Checkpointing**: Resume training from any checkpoint

## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.8+ (for GPU training)
- 16GB+ VRAM recommended for default settings

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/wan-lora-trainer.git
cd wan-lora-trainer
```

2. **Create a virtual environment:**
```bash
python -m venv wan-lora-env
source wan-lora-env/bin/activate  # On Windows: wan-lora-env\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Install optional optimizations:**
```bash
# For xFormers (memory efficient attention)
pip install xformers

# For Flash Attention
pip install flash-attn --no-build-isolation

# For 8-bit optimizers
pip install bitsandbytes

# For FP8 training (requires H100/H800)
pip install transformer-engine[pytorch]
```

## Quick Start

### Basic Training

```python
from src.training import create_training_session

# Create a training session
orchestrator = create_training_session(
    data_root="path/to/your/videos",
    model_id="alibaba-pai/wan-2.1-base",
    output_dir="./wan-lora-outputs",
    resolution=(512, 512),
    num_frames=16,
    lora_rank=64,
    learning_rate=1e-4,
    num_epochs=10
)

# Run training
training_stats = orchestrator.train()
```

### Command Line Training

```bash
python train_wan_lora.py \
    --data_root ./training_videos \
    --model_id alibaba-pai/wan-2.1-base \
    --output_dir ./outputs \
    --resolution 512 512 \
    --num_frames 16 \
    --lora_rank 64 \
    --learning_rate 1e-4 \
    --num_train_epochs 10 \
    --train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --mixed_precision fp16 \
    --gradient_checkpointing \
    --cpu_offload \
    --auto_optimize_batch_size
```

## Configuration

### Training Configuration

```python
from src.models.wan import TrainingConfig

config = TrainingConfig(
    # Model settings
    model_id="alibaba-pai/wan-2.1-base",
    
    # LoRA settings
    lora_rank=64,
    lora_alpha=64.0,
    lora_dropout=0.1,
    lora_target_modules=[
        "to_q", "to_k", "to_v", "to_out.0",
        "proj_in", "proj_out",
        "ff.net.0.proj", "ff.net.2"
    ],
    
    # Training settings
    learning_rate=1e-4,
    num_train_epochs=10,
    train_batch_size=1,
    gradient_accumulation_steps=4,
    
    # Video settings
    resolution=(512, 512),
    num_frames=16,
    
    # Optimization
    mixed_precision="fp16",
    gradient_checkpointing=True,
    cpu_offload=True,
    enable_xformers=True,
    enable_flash_attention=True
)
```

### Memory Optimization

```python
from src.models.wan.memory_utils import MemoryConfig

memory_config = MemoryConfig(
    enable_gradient_checkpointing=True,
    enable_cpu_offload=True,
    enable_attention_slicing=True,
    enable_xformers=True,
    enable_flash_attention=True,
    max_memory_usage=0.9  # Use 90% of VRAM
)
```

## Dataset Preparation

### Directory Structure

Organize your training videos in the following structure:

```
training_data/
├── video1.mp4
├── video2.mp4
├── subfolder/
│   ├── video3.avi
│   └── video4.mov
└── metadata.json  # Optional
```

### Supported Formats
- Video: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`
- Duration: 1-30 seconds recommended
- Resolution: Any (will be resized to target resolution)

### Metadata Format (Optional)

```json
{
    "video1.mp4": {
        "caption": "A cat playing with a ball in the garden",
        "tags": ["cat", "playing", "garden"]
    },
    "video2.mp4": {
        "caption": "Ocean waves crashing on a rocky shore",
        "tags": ["ocean", "waves", "nature"]
    }
}
```

## Advanced Usage

### Custom Model Components

```python
from src.models.wan import WanVideoModel, LoraConfig

# Custom LoRA configuration
lora_config = LoraConfig(
    rank=128,  # Higher rank for more parameters
    alpha=128.0,
    dropout=0.1,
    target_modules=[
        "to_q", "to_k", "to_v",  # Attention layers
        "proj_in", "proj_out",   # Projection layers
        "ff.net.0.proj", "ff.net.2"  # Feed-forward layers
    ]
)

# Initialize model
wan_model = WanVideoModel(
    model_id="alibaba-pai/wan-2.2-base",
    lora_config=lora_config,
    device="cuda"
)
```

### Memory-Optimized Training

```python
from src.models.wan import WanPipelineTrainer, TrainingConfig
from src.models.wan.memory_utils import MemoryOptimizer, GroupOffloadManager

# Setup memory optimization
memory_optimizer = MemoryOptimizer(device="cuda")
offload_manager = GroupOffloadManager(device="cuda", cpu_device="cpu")

# Configure for limited VRAM
config = TrainingConfig(
    train_batch_size=1,
    gradient_accumulation_steps=8,
    mixed_precision="fp16",
    gradient_checkpointing=True,
    cpu_offload=True,
    enable_attention_slicing=True
)

trainer = WanPipelineTrainer(config, train_dataloader)
trainer.train()
```

### Distributed Training

```bash
# Multi-GPU training with Accelerate
accelerate config  # Configure once

accelerate launch train_wan_lora.py \
    --data_root ./training_videos \
    --model_id alibaba-pai/wan-2.1-base \
    --train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --mixed_precision fp16
```

## Monitoring and Logging

### Weights & Biases Integration

```python
# Enable W&B logging
config = TrainingConfig(
    report_to="wandb",
    tracker_project_name="my-wan-project"
)

# Set W&B API key
export WANDB_API_KEY="your-api-key"
```

### Custom Validation

```python
config = TrainingConfig(
    validation_steps=500,
    validation_prompts=[
        "A red car driving through a city",
        "A bird flying over mountains",
        "Rain falling on a window"
    ],
    num_validation_videos=3
)
```

## Performance Optimization

### Hardware Recommendations

| VRAM | Batch Size | Resolution | Frames | Settings |
|------|------------|------------|--------|----------|
| 8GB  | 1 | 256x256 | 8 | fp16, checkpointing, offload |
| 12GB | 1 | 512x512 | 16 | fp16, checkpointing |
| 16GB | 2 | 512x512 | 16 | fp16 |
| 24GB | 4 | 512x512 | 16 | fp16 or bf16 |
| 48GB+ | 8+ | 768x768 | 24+ | bf16 or fp32 |

### Optimization Tips

1. **Memory Optimization:**
   - Enable gradient checkpointing for 30-50% memory savings
   - Use CPU offloading for components not actively training
   - Enable attention slicing for long sequences

2. **Speed Optimization:**
   - Use xFormers or Flash Attention for 20-40% speedup
   - Enable mixed precision (fp16/bf16)
   - Optimize batch size with auto-optimization

3. **Quality Optimization:**
   - Higher LoRA rank (64-128) for better adaptation
   - Longer training with lower learning rate
   - Proper validation prompts for monitoring

## Troubleshooting

### Common Issues

1. **Out of Memory Errors:**
   ```python
   # Reduce batch size and enable optimizations
   config.train_batch_size = 1
   config.gradient_accumulation_steps = 8
   config.gradient_checkpointing = True
   config.cpu_offload = True
   ```

2. **Slow Training:**
   ```python
   # Enable performance optimizations
   config.enable_xformers = True
   config.enable_flash_attention = True
   config.mixed_precision = "fp16"
   ```

3. **Model Loading Issues:**
   ```python
   # Specify cache directory and check model ID
   config.cache_dir = "./model_cache"
   config.model_id = "alibaba-pai/wan-2.1-base"  # Verify correct ID
   ```

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed memory logging
memory_optimizer = MemoryOptimizer(device="cuda")
stats = memory_optimizer.get_memory_stats()
print(f"Memory usage: {stats}")
```

## API Reference

### Core Classes

- `WanVideoModel`: Main model wrapper with LoRA support
- `WanPipelineTrainer`: Training pipeline orchestrator
- `WanLoRATrainingOrchestrator`: High-level training interface
- `MemoryOptimizer`: Memory optimization utilities
- `MixedPrecisionManager`: Precision and scaling management

### Configuration Classes

- `TrainingConfig`: Main training configuration
- `LoraConfig`: LoRA-specific settings
- `MemoryConfig`: Memory optimization settings
- `PrecisionConfig`: Mixed precision settings

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- HuggingFace Diffusers team for the WAN model implementation
- Microsoft for the LoRA technique and PEFT library
- NVIDIA for optimization techniques and tools

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{wan_lora_trainer,
  title={WAN LoRA Trainer: Efficient Video Generation Model Fine-tuning},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/wan-lora-trainer}
}
```

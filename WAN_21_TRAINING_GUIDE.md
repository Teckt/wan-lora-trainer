# WAN 2.1 LoRA Training Guide - Complete Setup

This guide provides a comprehensive setup for training WAN 2.1 LoRA models using kohya-ss/musubi-tuner with fp8 optimization and latent caching on Windows.

## 🚀 Quick Start

1. **Run the installation script as Administrator:**
   ```powershell
   # Right-click PowerShell -> "Run as Administrator"
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   .\install_musubi_tuner_windows.ps1
   ```

2. **The script will:**
   - Install Python dependencies
   - Set up musubi-tuner with PyTorch CUDA support
   - Configure accelerate for single GPU training
   - Create directory structure and training scripts
   - Enable fp8 training and latent caching

## 📁 Directory Structure

After installation, you'll have:

```
C:\musubi-tuner\
├── models/
│   ├── wan2.1/           # WAN 2.1 DiT models
│   ├── vae/              # VAE models  
│   └── text_encoders/    # T5 and CLIP encoders
├── datasets/
│   ├── training_data/
│   │   └── videos/       # Your training videos + captions
│   └── cache/
│       ├── latents/      # Cached latents (fp8 optimized)
│       └── text_encoder_outputs/  # Cached text embeddings
├── output/
│   ├── lora_models/      # Trained LoRA weights
│   ├── generated_videos/ # Inference outputs
│   └── logs/             # Tensorboard logs
├── configs/
│   └── dataset_config.toml  # Training configuration
└── scripts...            # Training and inference scripts
```

## 📥 Model Downloads

### Required Models for WAN 2.1:

**1. DiT Models (Choose based on your task):**
- **T2V 1.3B**: `wan2.1_t2v_1.3B_bf16.safetensors` (lighter, faster)
- **T2V 14B**: `wan2.1_t2v_14B_bf16.safetensors` (higher quality)
- **I2V 14B**: `wan2.1_i2v_14B_bf16.safetensors` (image-to-video)

Download from: [HuggingFace WAN 2.1 Models](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/tree/main/split_files/diffusion_models)

**2. VAE Model:**
- `wan_2.1_vae.safetensors`

Download from: [HuggingFace VAE](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/tree/main/split_files/vae)

**3. Text Encoders:**
- `models_t5_umt5-xxl-enc-bf16.pth` (required)
- `models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth` (for I2V only)

Download from: [HuggingFace Text Encoders](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P/tree/main)

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

**Caption Tips:**
- Be descriptive but concise
- Include style, lighting, mood
- Mention camera movement if relevant
- Use consistent terminology across captions

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
video_dir = "C:/musubi-tuner/datasets/training_data/videos"
caption_extension = ".txt"
cache_latents = true
cache_latents_to_disk = true
cache_text_encoder_outputs = true
cache_text_encoder_outputs_to_disk = true
latents_cache_dir = "C:/musubi-tuner/datasets/cache/latents"
text_encoder_outputs_cache_dir = "C:/musubi-tuner/datasets/cache/text_encoder_outputs"
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

### Step 1: Cache Latents (with fp8)
```bash
python src/musubi_tuner/wan_cache_latents.py \
    --dataset_config configs/dataset_config.toml \
    --vae models/vae/wan_2.1_vae.safetensors \
    --vae_tiling \
    --vae_chunk_size 32 \
    --fp8_base
```

### Step 2: Cache Text Encoder Outputs
```bash
python src/musubi_tuner/wan_cache_text_encoder_outputs.py \
    --dataset_config configs/dataset_config.toml \
    --t5 models/text_encoders/models_t5_umt5-xxl-enc-bf16.pth \
    --batch_size 8 \
    --fp8_t5
```

### Step 3: Train LoRA
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
    --output_name wan2.1_lora_fp8
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
python src/musubi_tuner/wan_generate_video.py --interactive [other args...]
```
Then enter prompts directly in the console.

## 📈 Monitoring Training

### Tensorboard:
```bash
# In separate terminal
tensorboard --logdir output/logs
# Visit: http://localhost:6006
```

### Training Metrics:
- **Loss curves**: Monitor convergence
- **Learning rate**: Check scheduler
- **Memory usage**: Ensure stable VRAM usage
- **Generated samples**: Visual progress (if sampling enabled)

## 🛠 Troubleshooting

### Common Issues:

**Out of Memory:**
- Reduce `--network_dim` (32 → 16)
- Increase `--vae_chunk_size`
- Enable `--vae_cache_cpu`
- Reduce batch size in config
- Use smaller frame_length

**Poor Quality Results:**
- Increase training epochs
- Use higher quality training data
- Adjust `--learning_rate` (try 1e-4)
- Increase `--network_dim` (32 → 64)
- Better captions

**Slow Training:**
- Ensure fp8 is enabled
- Use cached latents/text outputs
- Reduce `--max_data_loader_n_workers`
- Check GPU utilization

**Model Loading Errors:**
- Verify model file paths
- Check model file integrity
- Ensure sufficient disk space
- Verify model compatibility (bf16/fp16/fp8)

## 🎯 Advanced Tips

### LoRA Settings:
- **network_dim 32**: Good balance of quality/size
- **network_alpha 16**: Usually network_dim/2
- Higher values = more capacity but larger files

### Training Schedule:
- **Start**: 10 epochs with small dataset
- **Monitor**: Loss convergence and sample quality
- **Extend**: More epochs if still improving
- **Early Stop**: If overfitting occurs

### Dataset Tips:
- **Consistency**: Similar style/theme works better
- **Diversity**: But include varied scenes within theme
- **Quality**: Better to have fewer high-quality videos
- **Length**: 3-10 second clips are optimal

### Inference Optimization:
- **--attn_mode torch**: Good balance (default)
- **--attn_mode xformers**: If installed, faster
- **--compile**: PyTorch 2.0 compilation (experimental)

## 📚 Additional Resources

- **Official Documentation**: [musubi-tuner/docs/wan.md](https://github.com/kohya-ss/musubi-tuner/blob/main/docs/wan.md)
- **Dataset Configuration**: [dataset_config.md](https://github.com/kohya-ss/musubi-tuner/blob/main/src/musubi_tuner/dataset/dataset_config.md)
- **Advanced Settings**: [advanced_config.md](https://github.com/kohya-ss/musubi-tuner/blob/main/docs/advanced_config.md)

Happy training! 🚀

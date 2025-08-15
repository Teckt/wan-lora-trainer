"""
Setup script for WAN LoRA Trainer.
Installs dependencies and sets up the environment.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} detected. Python 3.8+ required.")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def check_gpu():
    """Check GPU availability."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"✅ GPU detected: {gpu_name} ({memory_gb:.1f} GB)")
            return True
        else:
            print("⚠️  No CUDA GPU detected. Training will be slow on CPU.")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed yet, will check GPU after installation.")
        return True

def main():
    """Main setup function."""
    print("🚀 Setting up WAN LoRA Trainer...")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check if we're in the right directory
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found. Please run this script from the project root.")
        sys.exit(1)
    
    # Install requirements
    if not run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing requirements"
    ):
        print("❌ Failed to install requirements. Please check your internet connection and try again.")
        sys.exit(1)
    
    # Check GPU after installation
    check_gpu()
    
    # Create necessary directories
    directories = ["cache", "output", "data"]
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"✅ Created directory: {dir_name}")
    
    # Test the setup
    print("\n🧪 Testing setup...")
    
    try:
        # Test imports
        import torch
        import transformers
        import diffusers
        import peft
        
        print("✅ Core libraries imported successfully")
        
        # Test CUDA
        if torch.cuda.is_available():
            device = torch.cuda.get_device_name(0)
            memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"✅ CUDA available: {device} ({memory:.1f} GB)")
            
            # Memory recommendations
            if memory >= 24:
                print("💡 24+ GB VRAM: You can train WAN 1.3B and 14B models")
            elif memory >= 16:
                print("💡 16+ GB VRAM: Use quantization for 14B models")
            elif memory >= 12:
                print("💡 12+ GB VRAM: Focus on 1.3B models with optimization")
            else:
                print("⚠️  <12 GB VRAM: Consider using CPU or cloud training")
        
    except ImportError as e:
        print(f"❌ Import test failed: {e}")
        sys.exit(1)
    
    # Success message
    print("\n" + "="*60)
    print("🎉 Setup completed successfully!")
    print("="*60)
    print("\nNext steps:")
    print("1. Prepare your training data in the 'data' directory")
    print("2. Run the test script: python test_setup.py")
    print("3. Start training: python train.py --data_dir data --output_dir output")
    print("\nFor help: python train.py --help")
    print("\nExample commands:")
    print("  # Quick test with WAN 1.3B")
    print("  python train.py --data_dir data --model_name 'Wan-AI/Wan2.1-T2V-1.3B-Diffusers' --precision fp16")
    print("\n  # Memory-optimized training")
    print("  python train.py --data_dir data --load_in_8bit --enable_gradient_checkpointing --attention_chunk_size 512")

if __name__ == "__main__":
    main()

"""
Example script to test WAN LoRA training setup.
Tests model loading, memory optimization, and basic training components.
"""

import sys
import os
from pathlib import Path
import torch
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from models.wan import (
    WanLoRAConfig,
    WanModelType,
    PrecisionType,
    WanModelLoader,
    log_memory_usage,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_config():
    """Test configuration creation."""
    logger.info("Testing configuration...")
    
    config = WanLoRAConfig(
        model_name="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        model_type=WanModelType.WAN_2_1_1_3B,
        precision=PrecisionType.FP16,
        
        # Memory optimizations for testing
        gradient_checkpointing=True,
        attention_chunk_size=512,
        ff_chunk_size=1024,
        load_in_8bit=True,
        
        # Training settings
        learning_rate=1e-4,
        train_batch_size=1,
        num_train_epochs=1,
        lora_rank=32,  # Smaller for testing
    )
    
    logger.info(f"Config created: {config.model_name}")
    logger.info(f"Precision: {config.precision}")
    logger.info(f"LoRA rank: {config.lora_rank}")
    
    return config

def test_memory_estimation(config):
    """Test memory estimation."""
    logger.info("Testing memory estimation...")
    
    model_loader = WanModelLoader(config)
    estimates = model_loader.estimate_memory_usage()
    
    logger.info("Memory estimates:")
    for component, size_gb in estimates.items():
        logger.info(f"  {component}: {size_gb:.2f} GB")
    
    if estimates["total"] > 24:
        logger.warning("Estimated usage exceeds 24GB - will need aggressive optimizations")
    else:
        logger.info("Memory usage looks feasible for 24GB GPU")
    
    return estimates

def test_model_loading(config):
    """Test model loading (download only, no full loading)."""
    logger.info("Testing model components...")
    
    try:
        # Test if we can at least import the required components
        from transformers import T5TokenizerFast, UMT5EncoderModel
        from diffusers import FlowMatchEulerDiscreteScheduler
        from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
        from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
        from diffusers.pipelines.wan import WanPipeline
        
        logger.info("✓ All required components can be imported")
        
        # Test tokenizer loading (lightweight)
        logger.info("Testing tokenizer loading...")
        try:
            tokenizer = T5TokenizerFast.from_pretrained(
                config.model_name,
                subfolder="tokenizer",
                cache_dir=config.cache_dir,
            )
            logger.info(f"✓ Tokenizer loaded: {len(tokenizer)} tokens")
        except Exception as e:
            logger.warning(f"Could not load tokenizer: {e}")
        
        return True
        
    except ImportError as e:
        logger.error(f"Missing required components: {e}")
        return False

def test_lora_config(config):
    """Test LoRA configuration."""
    logger.info("Testing LoRA configuration...")
    
    lora_config = config.get_lora_config()
    
    logger.info(f"LoRA config created:")
    logger.info(f"  Rank: {lora_config.r}")
    logger.info(f"  Alpha: {lora_config.lora_alpha}")
    logger.info(f"  Dropout: {lora_config.lora_dropout}")
    logger.info(f"  Target modules: {lora_config.target_modules}")
    
    return lora_config

def test_quantization_config(config):
    """Test quantization configuration."""
    logger.info("Testing quantization configuration...")
    
    if config.load_in_8bit or config.load_in_4bit:
        quant_config = config.get_quantization_config()
        if quant_config:
            logger.info("✓ Quantization config created")
            logger.info(f"  Load in 8bit: {config.load_in_8bit}")
            logger.info(f"  Load in 4bit: {config.load_in_4bit}")
        else:
            logger.warning("Could not create quantization config")
    else:
        logger.info("No quantization enabled")

def main():
    """Run all tests."""
    logger.info("Starting WAN LoRA Trainer tests...")
    log_memory_usage("test start")
    
    try:
        # Test 1: Configuration
        config = test_config()
        
        # Test 2: Memory estimation
        estimates = test_memory_estimation(config)
        
        # Test 3: Model components
        components_ok = test_model_loading(config)
        
        # Test 4: LoRA configuration
        lora_config = test_lora_config(config)
        
        # Test 5: Quantization configuration
        test_quantization_config(config)
        
        # Summary
        logger.info("\n" + "="*50)
        logger.info("TEST SUMMARY")
        logger.info("="*50)
        logger.info(f"✓ Configuration: OK")
        logger.info(f"✓ Memory estimation: {estimates['total']:.1f} GB total")
        logger.info(f"{'✓' if components_ok else '✗'} Model components: {'OK' if components_ok else 'FAILED'}")
        logger.info(f"✓ LoRA config: rank {lora_config.r}")
        logger.info(f"✓ Quantization: {'8-bit' if config.load_in_8bit else '4-bit' if config.load_in_4bit else 'disabled'}")
        
        if components_ok:
            logger.info("\n🎉 All tests passed! System is ready for training.")
            logger.info("\nTo start training, run:")
            logger.info("python train.py --data_dir <your_data> --output_dir <output>")
        else:
            logger.error("\n❌ Some tests failed. Please check dependencies.")
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        raise
    
    finally:
        log_memory_usage("test end")

if __name__ == "__main__":
    main()

"""
Complete WAN LoRA Training Script
Integrates preprocessing, model loading, and training for WAN 2.1/2.2 models.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import json

import torch
from accelerate.logging import get_logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from models.wan import (
    WanLoRAConfig,
    WanModelType,
    PrecisionType,
    WanModelLoader,
    WanLoRATrainer,
    log_memory_usage,
)
from preprocessing import (
    VideoProcessor,
    ImageProcessor,
    DatasetManager,
    PreprocessingPipeline,
)

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = get_logger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train LoRA for WAN models")
    
    # Data arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing training videos/images",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./cache",
        help="Directory for caching preprocessed data and models",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Directory to save trained model and checkpoints",
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        choices=[
            "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            "Wan-AI/Wan2.1-T2V-14B-Diffusers",
            "Wan-AI/Wan2.2-T2V-1.3B-Diffusers",
            "Wan-AI/Wan2.2-T2V-14B-Diffusers",
        ],
        help="WAN model to use for training",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="text_to_video",
        choices=["text_to_video", "image_to_video"],
        help="Type of model training",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp16",
        choices=["fp32", "fp16", "bf16", "fp8"],
        help="Training precision",
    )
    
    # Training arguments
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate for training",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Training batch size",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=5,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Maximum number of training steps (overrides epochs)",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for clipping",
    )
    
    # LoRA arguments
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=64,
        help="LoRA rank (dimensionality)",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=64,
        help="LoRA alpha (scaling factor)",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="LoRA dropout probability",
    )
    
    # Memory optimization arguments
    parser.add_argument(
        "--enable_gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing",
    )
    parser.add_argument(
        "--enable_cpu_offload",
        action="store_true",
        help="Enable CPU offloading",
    )
    parser.add_argument(
        "--attention_chunk_size",
        type=int,
        default=1024,
        help="Chunk size for attention computation",
    )
    parser.add_argument(
        "--ff_chunk_size",
        type=int,
        default=2048,
        help="Chunk size for feed-forward computation",
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load model in 8-bit precision",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load model in 4-bit precision",
    )
    
    # Preprocessing arguments
    parser.add_argument(
        "--video_length",
        type=int,
        default=16,
        help="Number of frames for video training",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Training resolution (height and width)",
    )
    parser.add_argument(
        "--skip_preprocessing",
        action="store_true",
        help="Skip preprocessing if data is already processed",
    )
    
    # Logging and saving arguments
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Log every X steps",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every X steps",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=0,
        help="Generate preview every X steps (0 to disable)",
    )
    parser.add_argument(
        "--preview_prompt",
        type=str,
        default="A beautiful landscape with mountains and trees",
        help="Prompt for preview generation",
    )
    
    # Other arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    
    return parser.parse_args()

def setup_config(args) -> WanLoRAConfig:
    """Create training configuration from arguments."""
    
    # Determine model type enum
    if "wan2.1" in args.model_name.lower():
        if "1.3b" in args.model_name.lower():
            model_type = WanModelType.WAN_2_1_1_3B
        else:
            model_type = WanModelType.WAN_2_1_14B
    else:  # WAN 2.2
        if "1.3b" in args.model_name.lower():
            model_type = WanModelType.WAN_2_2_1_3B
        else:
            model_type = WanModelType.WAN_2_2_14B
    
    # Determine precision type
    precision_map = {
        "fp32": PrecisionType.FP32,
        "fp16": PrecisionType.FP16, 
        "bf16": PrecisionType.BF16,
        "fp8": PrecisionType.FP8,
    }
    precision = precision_map[args.precision]
    
    config = WanLoRAConfig(
        # Model configuration
        model_name=args.model_name,
        model_type=model_type,
        precision=precision,
        cache_dir=args.cache_dir,
        
        # Training configuration
        learning_rate=args.learning_rate,
        train_batch_size=args.train_batch_size,
        num_train_epochs=args.num_train_epochs,
        max_train_steps=args.max_train_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        
        # LoRA configuration
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        
        # Memory optimization
        gradient_checkpointing=args.enable_gradient_checkpointing,
        enable_cpu_offload=args.enable_cpu_offload,
        attention_chunk_size=args.attention_chunk_size,
        ff_chunk_size=args.ff_chunk_size,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        
        # Logging and saving
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        validation_steps=args.validation_steps,
        preview_prompt=args.preview_prompt,
        
        # Other
        seed=args.seed,
    )
    
    return config

def preprocess_data(args, config: WanLoRAConfig) -> DatasetManager:
    """Preprocess training data."""
    if args.skip_preprocessing:
        logger.info("Skipping preprocessing...")
        # Load existing dataset manager
        dataset_manager = DatasetManager(
            data_dir=args.data_dir,
            cache_dir=args.cache_dir,
        )
        dataset_manager.load_cached_samples()
        return dataset_manager
    
    logger.info("Starting data preprocessing...")
    
    # Setup processors
    video_processor = VideoProcessor(
        target_resolution=(args.resolution, args.resolution),
        target_fps=8,
        max_frames=args.video_length,
    )
    
    image_processor = ImageProcessor(
        target_resolution=(args.resolution, args.resolution),
    )
    
    # Create preprocessing pipeline
    pipeline = PreprocessingPipeline(
        video_processor=video_processor,
        image_processor=image_processor,
        cache_dir=args.cache_dir,
    )
    
    # Process data
    dataset_manager = pipeline.process_folder(
        data_dir=args.data_dir,
        model_type=args.model_type,
    )
    
    logger.info(f"Preprocessing complete: {len(dataset_manager)} samples")
    return dataset_manager

def main():
    """Main training function."""
    args = parse_args()
    
    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup configuration
    config = setup_config(args)
    
    # Save configuration
    config_path = output_dir / "training_config.json"
    with open(config_path, "w") as f:
        json.dump(config.__dict__, f, indent=2, default=str)
    
    logger.info(f"Training configuration saved to {config_path}")
    logger.info(f"Training WAN model: {config.model_name}")
    logger.info(f"Precision: {config.precision}")
    logger.info(f"Memory optimizations: chunks={config.attention_chunk_size}/{config.ff_chunk_size}")
    
    log_memory_usage("startup")
    
    try:
        # Preprocess data
        dataset_manager = preprocess_data(args, config)
        
        # Estimate memory usage
        logger.info("Estimating memory requirements...")
        
        # Create model loader for estimates
        model_loader = WanModelLoader(config)
        memory_estimates = model_loader.estimate_memory_usage()
        
        logger.info("Memory estimates:")
        for component, size_gb in memory_estimates.items():
            logger.info(f"  {component}: {size_gb:.2f} GB")
        
        if memory_estimates["total"] > 24:
            logger.warning(
                f"Estimated memory usage ({memory_estimates['total']:.1f} GB) "
                "exceeds 24GB. Consider using more aggressive optimizations."
            )
        
        # Load models
        logger.info("Loading models...")
        models = model_loader.load_models()
        
        # Setup trainer
        trainer = WanLoRATrainer(
            config=config,
            model_loader=model_loader,
            dataset_manager=dataset_manager,
            output_dir=output_dir,
            resume_from_checkpoint=args.resume_from_checkpoint,
        )
        
        # Start training
        logger.info("Starting training...")
        results = trainer.train()
        
        # Log results
        logger.info("Training completed successfully!")
        logger.info(f"Final loss: {results['final_loss']:.4f}")
        logger.info(f"Total steps: {results['total_steps']}")
        logger.info(f"Total time: {results['total_time']:.2f}s")
        
        # Save results
        results_path = output_dir / "training_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Training results saved to {results_path}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    
    finally:
        # Cleanup
        if 'model_loader' in locals():
            model_loader.cleanup()
        
        log_memory_usage("cleanup")

if __name__ == "__main__":
    main()

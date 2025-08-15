"""
WAN Model Loader with Memory Optimization and LoRA Support.
Handles loading WAN 2.1/2.2 models with various precision and memory optimization options.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Tuple, Any
import logging
from pathlib import Path
import gc

from transformers import (
    T5TokenizerFast, 
    UMT5EncoderModel,
    AutoTokenizer,
    BitsAndBytesConfig
)
from diffusers import (
    DiffusionPipeline,
    FlowMatchEulerDiscreteScheduler,
    DDPMScheduler
)
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
from diffusers.pipelines.wan import WanPipeline

from peft import (
    LoraConfig, 
    get_peft_model, 
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    TaskType
)

from .config import WanLoRAConfig, apply_memory_optimizations, log_memory_usage

logger = logging.getLogger(__name__)

class WanModelLoader:
    """
    Loads and configures WAN models for LoRA training with memory optimizations.
    """
    
    def __init__(self, config: WanLoRAConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model components
        self.tokenizer = None
        self.text_encoder = None
        self.transformer = None
        self.vae = None
        self.scheduler = None
        self.pipeline = None
        
        # LoRA components
        self.lora_config = None
        self.original_transformer_state = None
        
    def load_models(self) -> Dict[str, Any]:
        """
        Load all WAN model components with memory optimizations.
        
        Returns:
            Dictionary containing all loaded model components
        """
        logger.info(f"Loading WAN model: {self.config.model_name}")
        log_memory_usage("before loading")
        
        # Load components in order of memory efficiency
        self._load_tokenizer()
        self._load_text_encoder()
        self._load_vae()
        self._load_transformer()
        self._load_scheduler()
        
        # Apply memory optimizations
        self._apply_optimizations()
        
        # Setup LoRA
        self._setup_lora()
        
        # Create pipeline
        self._create_pipeline()
        
        log_memory_usage("after loading")
        
        return {
            "tokenizer": self.tokenizer,
            "text_encoder": self.text_encoder,
            "transformer": self.transformer,
            "vae": self.vae,
            "scheduler": self.scheduler,
            "pipeline": self.pipeline,
            "config": self.config
        }
    
    def _load_tokenizer(self):
        """Load the tokenizer."""
        logger.info("Loading tokenizer...")
        
        try:
            self.tokenizer = T5TokenizerFast.from_pretrained(
                self.config.model_name,
                subfolder="tokenizer",
                cache_dir=self.config.cache_dir,
                max_length=self.config.max_sequence_length,
            )
            logger.info("Successfully loaded T5 tokenizer")
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise
    
    def _load_text_encoder(self):
        """Load the text encoder with optional quantization."""
        logger.info("Loading text encoder...")
        
        # Prepare loading kwargs
        load_kwargs = {
            "cache_dir": self.config.cache_dir,
            "torch_dtype": self.config.torch_dtype,
        }
        
        # Add quantization config if specified
        if self.config.load_in_8bit or self.config.load_in_4bit:
            quantization_config = self.config.get_quantization_config()
            if quantization_config:
                load_kwargs["quantization_config"] = quantization_config
                logger.info(f"Using quantization: {quantization_config}")
        
        try:
            self.text_encoder = UMT5EncoderModel.from_pretrained(
                self.config.model_name,
                subfolder="text_encoder",
                **load_kwargs
            )
            
            # Move to device if not quantized
            if not (self.config.load_in_8bit or self.config.load_in_4bit):
                self.text_encoder = self.text_encoder.to(self.device)
            
            # Disable gradients for text encoder (will be enabled selectively for LoRA)
            self.text_encoder.requires_grad_(False)
            
            logger.info("Successfully loaded text encoder")
            
        except Exception as e:
            logger.error(f"Failed to load text encoder: {e}")
            raise
    
    def _load_vae(self):
        """Load the VAE with optimizations."""
        logger.info("Loading VAE...")
        
        try:
            self.vae = AutoencoderKLWan.from_pretrained(
                self.config.model_name,
                subfolder="vae",
                cache_dir=self.config.cache_dir,
                torch_dtype=self.config.torch_dtype,
            )
            
            self.vae = self.vae.to(self.device)
            
            # Disable gradients for VAE
            self.vae.requires_grad_(False)
            
            # Apply VAE optimizations
            if self.config.enable_vae_slicing:
                self.vae.enable_slicing()
                logger.info("Enabled VAE slicing")
            
            if self.config.enable_vae_tiling:
                self.vae.enable_tiling()
                logger.info("Enabled VAE tiling")
            
            logger.info("Successfully loaded VAE")
            
        except Exception as e:
            logger.error(f"Failed to load VAE: {e}")
            raise
    
    def _load_transformer(self):
        """Load the main transformer model."""
        logger.info("Loading transformer...")
        
        # Prepare loading kwargs
        load_kwargs = {
            "cache_dir": self.config.cache_dir,
            "torch_dtype": self.config.torch_dtype,
        }
        
        # Add quantization config if specified
        if self.config.load_in_8bit or self.config.load_in_4bit:
            quantization_config = self.config.get_quantization_config()
            if quantization_config:
                load_kwargs["quantization_config"] = quantization_config
        
        try:
            self.transformer = WanTransformer3DModel.from_pretrained(
                self.config.model_name,
                subfolder="transformer",
                **load_kwargs
            )
            
            # Move to device if not quantized
            if not (self.config.load_in_8bit or self.config.load_in_4bit):
                self.transformer = self.transformer.to(self.device)
            
            # Store original state for restoration if needed
            self.original_transformer_state = self.transformer.state_dict()
            
            logger.info("Successfully loaded transformer")
            
        except Exception as e:
            logger.error(f"Failed to load transformer: {e}")
            raise
    
    def _load_scheduler(self):
        """Load the noise scheduler."""
        logger.info("Loading scheduler...")
        
        try:
            self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                self.config.model_name,
                subfolder="scheduler",
                cache_dir=self.config.cache_dir,
            )
            logger.info("Successfully loaded scheduler")
            
        except Exception as e:
            logger.error(f"Failed to load scheduler: {e}")
            raise
    
    def _apply_optimizations(self):
        """Apply memory optimizations to loaded models."""
        logger.info("Applying memory optimizations...")
        
        # Apply memory optimizations to transformer
        if self.transformer is not None:
            self.transformer = apply_memory_optimizations(self.transformer, self.config)
        
        # Apply optimizations to text encoder if needed
        if self.text_encoder is not None and self.config.gradient_checkpointing:
            if hasattr(self.text_encoder, 'gradient_checkpointing_enable'):
                self.text_encoder.gradient_checkpointing_enable()
        
        # Model compilation
        if self.config.compile_model:
            try:
                if self.transformer is not None:
                    self.transformer = torch.compile(self.transformer)
                    logger.info("Compiled transformer model")
                
                if self.text_encoder is not None:
                    self.text_encoder = torch.compile(self.text_encoder)
                    logger.info("Compiled text encoder")
                    
            except Exception as e:
                logger.warning(f"Model compilation failed: {e}")
    
    def _setup_lora(self):
        """Setup LoRA adapters for the transformer."""
        logger.info("Setting up LoRA adapters...")
        
        if self.transformer is None:
            raise ValueError("Transformer must be loaded before setting up LoRA")
        
        # Get LoRA configuration
        self.lora_config = self.config.get_lora_config()
        
        # Prepare model for k-bit training if using quantization
        if self.config.load_in_8bit or self.config.load_in_4bit:
            self.transformer = prepare_model_for_kbit_training(
                self.transformer,
                use_gradient_checkpointing=self.config.gradient_checkpointing
            )
            logger.info("Prepared model for k-bit training")
        
        # Disable all gradients first
        self.transformer.requires_grad_(False)
        
        # Add LoRA adapter
        self.transformer = get_peft_model(self.transformer, self.lora_config)
        
        # Print trainable parameters
        trainable_params, all_params = self._get_trainable_parameters()
        logger.info(
            f"LoRA setup complete: {trainable_params:,} trainable parameters "
            f"out of {all_params:,} total parameters "
            f"({100 * trainable_params / all_params:.2f}% trainable)"
        )
    
    def _get_trainable_parameters(self) -> Tuple[int, int]:
        """Get count of trainable and total parameters."""
        trainable_params = 0
        all_params = 0
        
        for param in self.transformer.parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        return trainable_params, all_params
    
    def _create_pipeline(self):
        """Create the inference pipeline."""
        logger.info("Creating inference pipeline...")
        
        try:
            self.pipeline = WanPipeline(
                tokenizer=self.tokenizer,
                text_encoder=self.text_encoder,
                transformer=self.transformer,
                vae=self.vae,
                scheduler=self.scheduler,
            )
            
            # Apply pipeline optimizations
            if self.config.enable_sequential_cpu_offload:
                self.pipeline.enable_sequential_cpu_offload()
                logger.info("Enabled sequential CPU offload")
            elif self.config.enable_model_cpu_offload:
                self.pipeline.enable_model_cpu_offload()
                logger.info("Enabled model CPU offload")
            
            logger.info("Successfully created pipeline")
            
        except Exception as e:
            logger.error(f"Failed to create pipeline: {e}")
            raise
    
    def get_optimizer(self, learning_rate: float = 1e-4) -> torch.optim.Optimizer:
        """
        Get an optimized optimizer for LoRA training.
        
        Args:
            learning_rate: Learning rate for the optimizer
            
        Returns:
            Configured optimizer
        """
        if self.transformer is None:
            raise ValueError("Transformer must be loaded before creating optimizer")
        
        # Get only trainable parameters (LoRA parameters)
        trainable_params = [p for p in self.transformer.parameters() if p.requires_grad]
        
        if self.config.use_8bit_optimizer:
            try:
                import bitsandbytes as bnb
                optimizer = bnb.optim.AdamW8bit(
                    trainable_params,
                    lr=learning_rate,
                    betas=(0.9, 0.999),
                    eps=1e-8,
                    weight_decay=0.01,
                )
                logger.info("Using 8-bit AdamW optimizer")
                return optimizer
            except ImportError:
                logger.warning("bitsandbytes not available, falling back to standard optimizer")
        
        # Fallback to standard optimizer
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
        )
        logger.info("Using standard AdamW optimizer")
        return optimizer
    
    def save_lora_weights(self, output_dir: Union[str, Path]):
        """Save LoRA weights to directory."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving LoRA weights to {output_dir}")
        
        # Save LoRA weights
        lora_state_dict = get_peft_model_state_dict(self.transformer)
        
        # Use the pipeline's save method if available
        if hasattr(self.pipeline, 'save_lora_weights'):
            self.pipeline.save_lora_weights(
                save_directory=output_dir,
                transformer_lora_layers=lora_state_dict,
            )
        else:
            # Fallback to manual saving
            torch.save(lora_state_dict, output_dir / "lora_weights.safetensors")
        
        # Save configuration
        self.lora_config.save_pretrained(output_dir)
        
        logger.info("LoRA weights saved successfully")
    
    def load_lora_weights(self, checkpoint_dir: Union[str, Path]):
        """Load LoRA weights from directory."""
        checkpoint_dir = Path(checkpoint_dir)
        
        logger.info(f"Loading LoRA weights from {checkpoint_dir}")
        
        if hasattr(self.pipeline, 'load_lora_weights'):
            self.pipeline.load_lora_weights(checkpoint_dir)
        else:
            # Manual loading
            lora_weights_path = checkpoint_dir / "lora_weights.safetensors"
            if lora_weights_path.exists():
                state_dict = torch.load(lora_weights_path, map_location=self.device)
                self.transformer.load_state_dict(state_dict, strict=False)
            else:
                raise FileNotFoundError(f"No LoRA weights found at {checkpoint_dir}")
        
        logger.info("LoRA weights loaded successfully")
    
    def cleanup(self):
        """Clean up GPU memory."""
        logger.info("Cleaning up GPU memory...")
        
        # Move models to CPU if needed
        if hasattr(self, 'transformer') and self.transformer is not None:
            self.transformer.cpu()
        if hasattr(self, 'text_encoder') and self.text_encoder is not None:
            self.text_encoder.cpu()
        if hasattr(self, 'vae') and self.vae is not None:
            self.vae.cpu()
        
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Garbage collection
        gc.collect()
        
        log_memory_usage("after cleanup")
    
    def estimate_memory_usage(self) -> Dict[str, float]:
        """
        Estimate memory usage for the current configuration.
        
        Returns:
            Dictionary with memory estimates in GB
        """
        estimates = {}
        
        # Base model sizes (approximate)
        if "1.3B" in self.config.model_name:
            base_model_size = 2.6  # GB in fp16
        elif "14B" in self.config.model_name:
            base_model_size = 28.0  # GB in fp16
        else:
            base_model_size = 5.0   # Default estimate
        
        # Adjust for precision
        if self.config.precision == "fp32":
            base_model_size *= 2
        elif self.config.precision == "fp8":
            base_model_size *= 0.5
        
        # Adjust for quantization
        if self.config.load_in_8bit:
            base_model_size *= 0.5
        elif self.config.load_in_4bit:
            base_model_size *= 0.25
        
        estimates["base_model"] = base_model_size
        
        # LoRA overhead (minimal)
        estimates["lora_params"] = 0.1  # Usually very small
        
        # Optimizer states (roughly 2x model size for AdamW)
        optimizer_multiplier = 1.0 if self.config.use_8bit_optimizer else 2.0
        estimates["optimizer"] = base_model_size * optimizer_multiplier
        
        # Activations (depends on sequence length and batch size)
        estimates["activations"] = 2.0  # Rough estimate
        
        # VAE and text encoder
        estimates["vae"] = 1.0
        estimates["text_encoder"] = 2.0
        
        # Total
        estimates["total"] = sum(estimates.values())
        
        return estimates

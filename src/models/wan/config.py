"""
WAN 2.1/2.2 Video Model Implementation for LoRA Training.
Supports both T2V and I2V training with memory optimization and mixed precision.
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing import Dict, List, Optional, Union, Tuple, Any
import logging
from pathlib import Path
from enum import Enum

from transformers import (
    T5TokenizerFast, 
    UMT5EncoderModel,
    AutoTokenizer
)
from diffusers import (
    AutoencoderKLWan,
    FlowMatchEulerDiscreteScheduler,
    DDPMScheduler
)
from diffusers.utils import is_compiled_module
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, prepare_model_for_kbit_training
import bitsandbytes as bnb

logger = logging.getLogger(__name__)

class WanModelType(Enum):
    """WAN model types and configurations."""
    WAN_2_1_1_3B = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    WAN_2_1_14B = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
    WAN_2_2_1_3B = "wan-2.2-1.3b"  # Future release
    WAN_2_2_14B = "wan-2.2-14b"    # Future release

class PrecisionType(Enum):
    """Supported precision types for training."""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8 = "fp8"

class WanLoRAConfig:
    """Configuration for WAN LoRA training."""
    
    def __init__(
        self,
        # Model configuration
        model_name: Union[str, WanModelType] = WanModelType.WAN_2_1_1_3B,
        cache_dir: Optional[str] = None,
        
        # LoRA configuration
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        target_modules: Optional[List[str]] = None,
        
        # Training configuration
        precision: PrecisionType = PrecisionType.FP16,
        gradient_checkpointing: bool = True,
        use_8bit_optimizer: bool = True,
        
        # Memory optimization
        enable_cpu_offloading: bool = True,
        enable_sequential_cpu_offload: bool = False,
        enable_model_cpu_offload: bool = True,
        enable_vae_slicing: bool = True,
        enable_vae_tiling: bool = True,
        
        # Chunking for memory efficiency
        enable_attention_chunking: bool = True,
        attention_chunk_size: int = 1024,
        enable_feed_forward_chunking: bool = True,
        ff_chunk_size: int = 2048,
        
        # Advanced optimization
        use_flash_attention: bool = True,
        use_scaled_dot_product_attention: bool = True,
        compile_model: bool = False,
        
        # Training parameters
        max_sequence_length: int = 512,
        resolution: Tuple[int, int] = (512, 512),
        max_frames: int = 16,
        fps: float = 8.0,
        
        # Quantization
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        bnb_4bit_compute_dtype: torch.dtype = torch.float16,
        bnb_4bit_use_double_quant: bool = True,
        bnb_4bit_quant_type: str = "nf4",
        
        # TREAD (Token Routing) for memory optimization
        enable_tread: bool = False,
        tread_selection_ratio: float = 0.1,
        tread_start_layer: int = 2,
        tread_end_layer: int = -2,
    ):
        # Convert enum to string if needed
        if isinstance(model_name, WanModelType):
            self.model_name = model_name.value
        else:
            self.model_name = model_name
            
        if isinstance(precision, PrecisionType):
            self.precision = precision.value
        else:
            self.precision = precision
        
        self.cache_dir = cache_dir
        
        # LoRA settings
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        
        # Default target modules for WAN transformer
        if target_modules is None:
            self.target_modules = [
                "to_q", "to_k", "to_v", "to_out.0",  # Attention layers
                "ff.net.0.proj", "ff.net.2",         # Feed-forward layers
            ]
        else:
            self.target_modules = target_modules
        
        # Training settings
        self.gradient_checkpointing = gradient_checkpointing
        self.use_8bit_optimizer = use_8bit_optimizer
        
        # Memory optimization
        self.enable_cpu_offloading = enable_cpu_offloading
        self.enable_sequential_cpu_offload = enable_sequential_cpu_offload
        self.enable_model_cpu_offload = enable_model_cpu_offload
        self.enable_vae_slicing = enable_vae_slicing
        self.enable_vae_tiling = enable_vae_tiling
        
        # Chunking
        self.enable_attention_chunking = enable_attention_chunking
        self.attention_chunk_size = attention_chunk_size
        self.enable_feed_forward_chunking = enable_feed_forward_chunking
        self.ff_chunk_size = ff_chunk_size
        
        # Advanced optimization
        self.use_flash_attention = use_flash_attention
        self.use_scaled_dot_product_attention = use_scaled_dot_product_attention
        self.compile_model = compile_model
        
        # Training parameters
        self.max_sequence_length = max_sequence_length
        self.resolution = resolution
        self.max_frames = max_frames
        self.fps = fps
        
        # Quantization
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.bnb_4bit_compute_dtype = bnb_4bit_compute_dtype
        self.bnb_4bit_use_double_quant = bnb_4bit_use_double_quant
        self.bnb_4bit_quant_type = bnb_4bit_quant_type
        
        # TREAD settings
        self.enable_tread = enable_tread
        self.tread_selection_ratio = tread_selection_ratio
        self.tread_start_layer = tread_start_layer
        self.tread_end_layer = tread_end_layer
    
    @property
    def torch_dtype(self) -> torch.dtype:
        """Get torch dtype from precision setting."""
        if self.precision == "fp32":
            return torch.float32
        elif self.precision == "fp16":
            return torch.float16
        elif self.precision == "bf16":
            return torch.bfloat16
        elif self.precision == "fp8":
            return torch.float8_e4m3fn if hasattr(torch, 'float8_e4m3fn') else torch.float16
        else:
            return torch.float16
    
    def get_quantization_config(self) -> Optional[Dict[str, Any]]:
        """Get quantization configuration for bitsandbytes."""
        if not (self.load_in_8bit or self.load_in_4bit):
            return None
        
        from transformers import BitsAndBytesConfig
        
        return BitsAndBytesConfig(
            load_in_8bit=self.load_in_8bit,
            load_in_4bit=self.load_in_4bit,
            bnb_4bit_compute_dtype=self.bnb_4bit_compute_dtype,
            bnb_4bit_use_double_quant=self.bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=self.bnb_4bit_quant_type,
        )
    
    def get_lora_config(self) -> LoraConfig:
        """Get PEFT LoRA configuration."""
        return LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.target_modules,
            bias="none",
            task_type="FEATURE_EXTRACTION",  # For transformer models
        )


class MemoryOptimizedAttention(nn.Module):
    """
    Memory-optimized attention mechanism with chunking support.
    Based on gradient checkpointing and attention chunking patterns.
    """
    
    def __init__(
        self, 
        original_attention: nn.Module,
        chunk_size: int = 1024,
        enable_chunking: bool = True
    ):
        super().__init__()
        self.original_attention = original_attention
        self.chunk_size = chunk_size
        self.enable_chunking = enable_chunking
    
    def forward(self, *args, **kwargs):
        """Forward pass with optional chunking and checkpointing."""
        if self.enable_chunking and self.training:
            return self._chunked_forward(*args, **kwargs)
        else:
            if self.training:
                return checkpoint(self.original_attention, *args, **kwargs)
            else:
                return self.original_attention(*args, **kwargs)
    
    def _chunked_forward(self, hidden_states, *args, **kwargs):
        """Chunked attention computation for memory efficiency."""
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        if seq_len <= self.chunk_size:
            # No need to chunk
            return checkpoint(self.original_attention, hidden_states, *args, **kwargs)
        
        # Split into chunks
        chunks = []
        for i in range(0, seq_len, self.chunk_size):
            end_idx = min(i + self.chunk_size, seq_len)
            chunk = hidden_states[:, i:end_idx, :]
            
            # Process chunk with gradient checkpointing
            chunk_output = checkpoint(self.original_attention, chunk, *args, **kwargs)
            chunks.append(chunk_output)
        
        # Concatenate results
        return torch.cat(chunks, dim=1)


class MemoryOptimizedFeedForward(nn.Module):
    """
    Memory-optimized feed-forward network with chunking support.
    """
    
    def __init__(
        self, 
        original_ff: nn.Module,
        chunk_size: int = 2048,
        enable_chunking: bool = True
    ):
        super().__init__()
        self.original_ff = original_ff
        self.chunk_size = chunk_size
        self.enable_chunking = enable_chunking
    
    def forward(self, hidden_states, *args, **kwargs):
        """Forward pass with optional chunking and checkpointing."""
        if self.enable_chunking and self.training:
            return self._chunked_forward(hidden_states, *args, **kwargs)
        else:
            if self.training:
                return checkpoint(self.original_ff, hidden_states, *args, **kwargs)
            else:
                return self.original_ff(hidden_states, *args, **kwargs)
    
    def _chunked_forward(self, hidden_states, *args, **kwargs):
        """Chunked feed-forward computation for memory efficiency."""
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        if seq_len <= self.chunk_size:
            # No need to chunk
            return checkpoint(self.original_ff, hidden_states, *args, **kwargs)
        
        # Split into chunks
        chunks = []
        for i in range(0, seq_len, self.chunk_size):
            end_idx = min(i + self.chunk_size, seq_len)
            chunk = hidden_states[:, i:end_idx, :]
            
            # Process chunk with gradient checkpointing
            chunk_output = checkpoint(self.original_ff, chunk, *args, **kwargs)
            chunks.append(chunk_output)
        
        # Concatenate results
        return torch.cat(chunks, dim=1)


def apply_memory_optimizations(model: nn.Module, config: WanLoRAConfig) -> nn.Module:
    """
    Apply memory optimizations to the model including chunking and checkpointing.
    """
    logger.info("Applying memory optimizations...")
    
    # Enable gradient checkpointing
    if config.gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        logger.info("Enabled gradient checkpointing")
    
    # Apply attention chunking
    if config.enable_attention_chunking:
        for name, module in model.named_modules():
            if 'attn' in name.lower() and hasattr(module, 'forward'):
                # Wrap attention modules with memory optimization
                parent_name = '.'.join(name.split('.')[:-1])
                module_name = name.split('.')[-1]
                
                if parent_name:
                    parent_module = model.get_submodule(parent_name)
                    optimized_attention = MemoryOptimizedAttention(
                        module, 
                        chunk_size=config.attention_chunk_size,
                        enable_chunking=config.enable_attention_chunking
                    )
                    setattr(parent_module, module_name, optimized_attention)
        
        logger.info(f"Applied attention chunking with chunk size {config.attention_chunk_size}")
    
    # Apply feed-forward chunking  
    if config.enable_feed_forward_chunking:
        for name, module in model.named_modules():
            if ('ff' in name.lower() or 'mlp' in name.lower()) and hasattr(module, 'forward'):
                # Wrap feed-forward modules with memory optimization
                parent_name = '.'.join(name.split('.')[:-1])
                module_name = name.split('.')[-1]
                
                if parent_name:
                    parent_module = model.get_submodule(parent_name)
                    optimized_ff = MemoryOptimizedFeedForward(
                        module,
                        chunk_size=config.ff_chunk_size,
                        enable_chunking=config.enable_feed_forward_chunking
                    )
                    setattr(parent_module, module_name, optimized_ff)
        
        logger.info(f"Applied feed-forward chunking with chunk size {config.ff_chunk_size}")
    
    # Enable flash attention if available
    if config.use_flash_attention:
        try:
            # Try to enable flash attention 2
            model = model.to_bettertransformer()
            logger.info("Enabled Flash Attention")
        except Exception as e:
            logger.warning(f"Could not enable Flash Attention: {e}")
    
    # Enable scaled dot product attention
    if config.use_scaled_dot_product_attention:
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            logger.info("Enabled Scaled Dot Product Attention")
        except Exception as e:
            logger.warning(f"Could not enable SDPA: {e}")
    
    return model


def get_memory_stats() -> Dict[str, float]:
    """Get current GPU memory statistics."""
    if not torch.cuda.is_available():
        return {"memory_allocated": 0.0, "memory_reserved": 0.0, "memory_free": 0.0}
    
    device = torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
    reserved = torch.cuda.memory_reserved(device) / 1024**3   # GB
    
    props = torch.cuda.get_device_properties(device)
    total = props.total_memory / 1024**3  # GB
    free = total - allocated
    
    return {
        "memory_allocated": allocated,
        "memory_reserved": reserved, 
        "memory_free": free,
        "memory_total": total
    }


def log_memory_usage(stage: str = ""):
    """Log current memory usage."""
    stats = get_memory_stats()
    logger.info(
        f"Memory Usage {stage}: "
        f"Allocated: {stats['memory_allocated']:.2f}GB, "
        f"Reserved: {stats['memory_reserved']:.2f}GB, "
        f"Free: {stats['memory_free']:.2f}GB, "
        f"Total: {stats['memory_total']:.2f}GB"
    )

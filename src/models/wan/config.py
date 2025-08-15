"""
WAN 2.1/2.2 Model Configuration System
Supports memory optimization, quantization, and chunking for LoRA training
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union, Tuple
from enum import Enum
import torch
import logging

logger = logging.getLogger(__name__)

class ModelPrecision(Enum):
    """Supported model precisions for WAN models."""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8 = "fp8"
    INT8_QUANTO = "int8-quanto"
    INT4_QUANTO = "int4-quanto"
    NF4_BNB = "nf4-bnb"

class WanModelVariant(Enum):
    """WAN model variants and their Hugging Face paths."""
    # WAN 2.1 models
    T2V_1_3B_2_1 = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    T2V_14B_2_1 = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
    # Future WAN 2.2 models (placeholder paths)
    T2V_1_3B_2_2 = "Wan-AI/Wan2.2-T2V-1.3B-Diffusers"  # Placeholder
    T2V_14B_2_2 = "Wan-AI/Wan2.2-T2V-14B-Diffusers"    # Placeholder

class MemoryOptimization(Enum):
    """Memory optimization strategies."""
    NONE = "none"
    SEQUENTIAL_CPU_OFFLOAD = "sequential_cpu_offload"
    MODEL_CPU_OFFLOAD = "model_cpu_offload"
    GROUP_OFFLOADING = "group_offloading"
    LEAF_LEVEL_OFFLOADING = "leaf_level_offloading"
    GRADIENT_CHECKPOINTING = "gradient_checkpointing"
    VAE_SLICING = "vae_slicing"
    VAE_TILING = "vae_tiling"

@dataclass
class QuantizationConfig:
    """Configuration for model quantization."""
    precision: ModelPrecision = ModelPrecision.FP16
    weight_dtype: Optional[torch.dtype] = None
    inference_dtype: Optional[torch.dtype] = None
    
    # Quanto quantization settings
    quanto_backend: str = "quanto"
    quanto_weights: Optional[str] = None  # "qint8", "qfloat8", "qint4"
    quanto_activations: Optional[str] = None  # "qfloat8"
    
    # BitsAndBytes settings
    bnb_4bit: bool = False
    bnb_8bit: bool = False
    bnb_compute_dtype: Optional[torch.dtype] = None
    bnb_quant_type: str = "nf4"  # "fp4", "nf4"
    bnb_use_double_quant: bool = True
    
    def __post_init__(self):
        """Set default dtypes based on precision."""
        if self.weight_dtype is None:
            self.weight_dtype = self._get_default_dtype()
        if self.inference_dtype is None:
            self.inference_dtype = self.weight_dtype
    
    def _get_default_dtype(self) -> torch.dtype:
        """Get default dtype based on precision setting."""
        precision_map = {
            ModelPrecision.FP32: torch.float32,
            ModelPrecision.FP16: torch.float16,
            ModelPrecision.BF16: torch.bfloat16,
            ModelPrecision.FP8: torch.float8_e4m3fn if hasattr(torch, 'float8_e4m3fn') else torch.float16,
            ModelPrecision.INT8_QUANTO: torch.float16,  # Base dtype for quantized models
            ModelPrecision.INT4_QUANTO: torch.float16,
            ModelPrecision.NF4_BNB: torch.float16,
        }
        return precision_map.get(self.precision, torch.float16)

@dataclass 
class MemoryConfig:
    """Configuration for memory optimization."""
    # Memory optimization strategies
    optimizations: List[MemoryOptimization] = field(default_factory=lambda: [
        MemoryOptimization.GRADIENT_CHECKPOINTING,
        MemoryOptimization.VAE_SLICING
    ])
    
    # Group offloading settings (for large models)
    group_offloading_enabled: bool = False
    group_offloading_num_blocks: int = 4
    group_offloading_type: str = "block_level"  # "block_level", "leaf_level"
    
    # CPU offloading settings
    cpu_offload_enabled: bool = False
    cpu_offload_sequential: bool = True  # vs model_cpu_offload
    
    # Memory limits (in GB)
    max_vram_gb: Optional[float] = None
    max_cpu_memory_gb: Optional[float] = None
    
    # Chunk processing for memory efficiency
    chunk_processing_enabled: bool = True
    attention_chunk_size: Optional[int] = None  # Chunked attention
    vae_chunk_size: int = 1  # VAE decode chunk size
    
    # Advanced memory settings
    enable_xformers: bool = True
    enable_flash_attention: bool = True
    enable_torch_compile: bool = False
    
    def estimate_memory_requirements(self, model_variant: WanModelVariant, 
                                   quantization: QuantizationConfig) -> Dict[str, float]:
        """Estimate memory requirements in GB."""
        # Base memory requirements for different models (in GB)
        base_requirements = {
            WanModelVariant.T2V_1_3B_2_1: {
                ModelPrecision.FP32: 6.0,
                ModelPrecision.FP16: 3.0,
                ModelPrecision.BF16: 3.0,
                ModelPrecision.INT8_QUANTO: 2.0,
                ModelPrecision.NF4_BNB: 1.5,
            },
            WanModelVariant.T2V_14B_2_1: {
                ModelPrecision.FP32: 56.0,
                ModelPrecision.FP16: 28.0,
                ModelPrecision.BF16: 28.0,
                ModelPrecision.INT8_QUANTO: 14.0,
                ModelPrecision.NF4_BNB: 8.0,
            }
        }
        
        base_mem = base_requirements.get(model_variant, {}).get(
            quantization.precision, 16.0  # Default fallback
        )
        
        # Add overhead for training
        training_overhead = base_mem * 0.3  # 30% overhead for gradients, optimizer states
        
        # Add VAE memory (roughly 2-3GB for video VAE)
        vae_memory = 2.5
        
        # Adjust for optimizations
        if MemoryOptimization.GRADIENT_CHECKPOINTING in self.optimizations:
            training_overhead *= 0.7  # 30% reduction
        
        if self.group_offloading_enabled:
            base_mem *= 0.5  # Significant reduction with offloading
        
        if MemoryOptimization.VAE_SLICING in self.optimizations:
            vae_memory *= 0.6  # 40% reduction
        
        total_vram = base_mem + training_overhead + vae_memory
        cpu_memory = 0.0
        
        if self.cpu_offload_enabled:
            # Move some memory to CPU
            cpu_memory = total_vram * 0.4
            total_vram *= 0.6
        
        return {
            "vram_gb": total_vram,
            "cpu_memory_gb": cpu_memory,
            "total_gb": total_vram + cpu_memory
        }

@dataclass
class LoRAConfig:
    """Configuration for LoRA training."""
    # LoRA rank and alpha
    rank: int = 16
    alpha: int = 16
    dropout: float = 0.1
    
    # Target modules for LoRA
    target_modules: List[str] = field(default_factory=lambda: [
        "to_k", "to_q", "to_v", "to_out.0"
    ])
    
    # LoRA type
    lora_type: str = "standard"  # "standard", "lycoris", "lokr"
    
    # Advanced LoRA settings
    use_rslora: bool = False
    use_dora: bool = False
    init_lora_weights: Union[bool, str] = True
    
    def get_target_modules_for_model(self, model_variant: WanModelVariant) -> List[str]:
        """Get target modules specific to WAN model architecture."""
        # WAN transformer uses standard attention patterns
        wan_targets = {
            "attention": ["to_k", "to_q", "to_v", "to_out.0"],
            "feedforward": ["ff.net.0.proj", "ff.net.2"],
            "comprehensive": [
                # Self attention
                "attn.to_k", "attn.to_q", "attn.to_v", "attn.to_out.0",
                # Feed forward
                "ff.net.0.proj", "ff.net.2",
                # Layer norms (optional)
                "norm1", "norm2",
            ]
        }
        
        if self.lora_type == "comprehensive":
            return wan_targets["comprehensive"]
        elif self.lora_type == "feedforward":
            return wan_targets["attention"] + wan_targets["feedforward"]
        else:
            return wan_targets["attention"]

@dataclass
class WanModelConfig:
    """Complete configuration for WAN model loading and training."""
    # Model selection
    model_variant: WanModelVariant = WanModelVariant.T2V_1_3B_2_1
    custom_model_path: Optional[str] = None  # Override with custom path
    
    # Model components
    load_transformer: bool = True
    load_vae: bool = True
    load_text_encoder: bool = True
    load_scheduler: bool = True
    
    # Quantization and memory
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig) 
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    
    # Training settings
    resolution: Tuple[int, int] = (512, 512)
    num_frames: int = 16
    max_sequence_length: int = 256
    
    # Device settings
    device: str = "cuda"
    device_map: Optional[Union[str, Dict[str, Any]]] = None
    
    # Cache settings
    cache_dir: Optional[str] = None
    use_safetensors: bool = True
    
    # Advanced settings
    torch_compile: bool = False
    use_auth_token: Optional[str] = None
    
    def __post_init__(self):
        """Validate and adjust configuration."""
        # Auto-adjust memory config based on model size
        if self.model_variant in [WanModelVariant.T2V_14B_2_1, WanModelVariant.T2V_14B_2_2]:
            # Large model - enable aggressive memory optimizations
            if not self.memory.group_offloading_enabled:
                logger.info("Enabling group offloading for 14B model")
                self.memory.group_offloading_enabled = True
            
            if MemoryOptimization.GRADIENT_CHECKPOINTING not in self.memory.optimizations:
                self.memory.optimizations.append(MemoryOptimization.GRADIENT_CHECKPOINTING)
        
        # Adjust quantization for very large models
        if (self.model_variant in [WanModelVariant.T2V_14B_2_1, WanModelVariant.T2V_14B_2_2] 
            and self.quantization.precision == ModelPrecision.FP32):
            logger.warning("FP32 not recommended for 14B model, switching to FP16")
            self.quantization.precision = ModelPrecision.FP16
    
    def get_model_path(self) -> str:
        """Get the model path (custom or from variant)."""
        return self.custom_model_path or self.model_variant.value
    
    def estimate_memory_usage(self) -> Dict[str, float]:
        """Estimate total memory usage."""
        return self.memory.estimate_memory_requirements(self.model_variant, self.quantization)
    
    def is_supported_for_hardware(self, available_vram_gb: float, 
                                 available_cpu_memory_gb: float = 32.0) -> bool:
        """Check if configuration is supported on given hardware."""
        requirements = self.estimate_memory_usage()
        
        # Check VRAM
        if requirements["vram_gb"] > available_vram_gb:
            return False
        
        # Check CPU memory if offloading is used
        if self.memory.cpu_offload_enabled and requirements["cpu_memory_gb"] > available_cpu_memory_gb:
            return False
        
        return True
    
    def get_optimization_suggestions(self, available_vram_gb: float) -> List[str]:
        """Get suggestions for memory optimization."""
        suggestions = []
        requirements = self.estimate_memory_usage()
        
        if requirements["vram_gb"] > available_vram_gb:
            suggestions.append(f"Current config requires {requirements['vram_gb']:.1f}GB VRAM, but only {available_vram_gb}GB available")
            
            # Suggest quantization
            if self.quantization.precision == ModelPrecision.FP16:
                suggestions.append("Consider using INT8 quantization (int8-quanto)")
            elif self.quantization.precision == ModelPrecision.FP32:
                suggestions.append("Consider using FP16 or INT8 quantization")
            
            # Suggest memory optimizations
            if not self.memory.group_offloading_enabled:
                suggestions.append("Enable group offloading for large memory savings")
            
            if not self.memory.cpu_offload_enabled:
                suggestions.append("Enable CPU offloading")
            
            if MemoryOptimization.VAE_SLICING not in self.memory.optimizations:
                suggestions.append("Enable VAE slicing")
            
            if MemoryOptimization.GRADIENT_CHECKPOINTING not in self.memory.optimizations:
                suggestions.append("Enable gradient checkpointing")
        
        return suggestions
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            "model_variant": self.model_variant.value,
            "custom_model_path": self.custom_model_path,
            "quantization": {
                "precision": self.quantization.precision.value,
                "weight_dtype": str(self.quantization.weight_dtype),
                "quanto_weights": self.quantization.quanto_weights,
                "bnb_4bit": self.quantization.bnb_4bit,
                "bnb_8bit": self.quantization.bnb_8bit,
            },
            "memory": {
                "optimizations": [opt.value for opt in self.memory.optimizations],
                "group_offloading_enabled": self.memory.group_offloading_enabled,
                "cpu_offload_enabled": self.memory.cpu_offload_enabled,
                "chunk_processing_enabled": self.memory.chunk_processing_enabled,
                "enable_xformers": self.memory.enable_xformers,
            },
            "lora": {
                "rank": self.lora.rank,
                "alpha": self.lora.alpha,
                "target_modules": self.lora.target_modules,
                "lora_type": self.lora.lora_type,
            },
            "resolution": self.resolution,
            "num_frames": self.num_frames,
            "device": self.device,
        }

# Predefined configurations for common use cases
def get_low_memory_config(model_variant: WanModelVariant = WanModelVariant.T2V_1_3B_2_1) -> WanModelConfig:
    """Get configuration optimized for low VRAM (8-12GB)."""
    config = WanModelConfig(
        model_variant=model_variant,
        quantization=QuantizationConfig(
            precision=ModelPrecision.INT8_QUANTO,
            quanto_weights="qint8"
        ),
        memory=MemoryConfig(
            optimizations=[
                MemoryOptimization.GRADIENT_CHECKPOINTING,
                MemoryOptimization.VAE_SLICING,
                MemoryOptimization.VAE_TILING,
                MemoryOptimization.SEQUENTIAL_CPU_OFFLOAD
            ],
            cpu_offload_enabled=True,
            chunk_processing_enabled=True,
            vae_chunk_size=1,
            enable_xformers=True,
            enable_flash_attention=True
        ),
        lora=LoRAConfig(
            rank=8,
            alpha=8,
            target_modules=["to_k", "to_q", "to_v", "to_out.0"]
        ),
        resolution=(384, 384),  # Lower resolution for memory savings
        num_frames=8  # Fewer frames
    )
    return config

def get_balanced_config(model_variant: WanModelVariant = WanModelVariant.T2V_1_3B_2_1) -> WanModelConfig:
    """Get balanced configuration for mid-range hardware (16-24GB)."""
    config = WanModelConfig(
        model_variant=model_variant,
        quantization=QuantizationConfig(
            precision=ModelPrecision.FP16
        ),
        memory=MemoryConfig(
            optimizations=[
                MemoryOptimization.GRADIENT_CHECKPOINTING,
                MemoryOptimization.VAE_SLICING
            ],
            chunk_processing_enabled=True,
            enable_xformers=True,
            enable_flash_attention=True
        ),
        lora=LoRAConfig(
            rank=16,
            alpha=16
        ),
        resolution=(512, 512),
        num_frames=16
    )
    return config

def get_high_quality_config(model_variant: WanModelVariant = WanModelVariant.T2V_14B_2_1) -> WanModelConfig:
    """Get configuration for high-end hardware (32GB+)."""
    config = WanModelConfig(
        model_variant=model_variant,
        quantization=QuantizationConfig(
            precision=ModelPrecision.BF16
        ),
        memory=MemoryConfig(
            optimizations=[
                MemoryOptimization.GRADIENT_CHECKPOINTING
            ],
            group_offloading_enabled=True,
            enable_xformers=True,
            enable_flash_attention=True,
            enable_torch_compile=True
        ),
        lora=LoRAConfig(
            rank=32,
            alpha=32,
            lora_type="comprehensive"
        ),
        resolution=(768, 768),
        num_frames=24
    )
    return config

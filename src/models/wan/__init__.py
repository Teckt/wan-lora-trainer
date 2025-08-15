"""
WAN package initialization.
Provides access to WAN model components for LoRA training.
"""

from .config import (
    WanLoRAConfig,
    WanModelType,
    PrecisionType,
    MemoryOptimizedAttention,
    MemoryOptimizedFeedForward,
    apply_memory_optimizations,
    log_memory_usage,
)

from .model_loader import WanModelLoader

from .transformer import (
    MemoryOptimizedWanTransformer,
    WanTransformerBlock,
    TREADRouter,
    WanTransformerOutput,
)

from .trainer import (
    WanLoRATrainer,
    TrainingMetrics,
)

__all__ = [
    # Configuration
    "WanLoRAConfig",
    "WanModelType", 
    "PrecisionType",
    "MemoryOptimizedAttention",
    "MemoryOptimizedFeedForward",
    "apply_memory_optimizations",
    "log_memory_usage",
    
    # Model loader
    "WanModelLoader",
    
    # Transformer components
    "MemoryOptimizedWanTransformer",
    "WanTransformerBlock", 
    "TREADRouter",
    "WanTransformerOutput",
    
    # Training
    "WanLoRATrainer",
    "TrainingMetrics",
]

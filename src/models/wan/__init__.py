"""
WAN Models Package
Provides WAN 2.1/2.2 model configurations, loading, and transformer implementations
"""

from .config import (
    WanModelConfig,
    WanModelVariant,
    ModelPrecision,
    MemoryOptimization,
    QuantizationConfig,
    MemoryConfig,
    LoRAConfig,
    get_balanced_config,
    get_memory_optimized_config,
    get_precision_config
)

from .model_loader import (
    WanModelLoader,
    MemoryOptimizer,
    QuantizationManager,
    ChunkedProcessor,
    create_model_loader,
    estimate_model_memory
)

try:
    from .transformer import (
        WanTransformer3DModel,
        WanTransformerConfig,
        WanTransformerBlock,
        ChunkedAttention,
        LoRAIntegration,
        create_wan_transformer,
        count_parameters
    )
    TRANSFORMER_AVAILABLE = True
except ImportError:
    TRANSFORMER_AVAILABLE = False

__all__ = [
    # Configuration
    "WanModelConfig",
    "WanModelVariant", 
    "ModelPrecision",
    "MemoryOptimization",
    "QuantizationConfig",
    "MemoryConfig",
    "LoRAConfig",
    "get_balanced_config",
    "get_memory_optimized_config", 
    "get_precision_config",
    
    # Model loading
    "WanModelLoader",
    "MemoryOptimizer",
    "QuantizationManager",
    "ChunkedProcessor", 
    "create_model_loader",
    "estimate_model_memory",
]

# Add transformer exports if available
if TRANSFORMER_AVAILABLE:
    __all__.extend([
        "WanTransformer3DModel",
        "WanTransformerConfig",
        "WanTransformerBlock",
        "ChunkedAttention",
        "LoRAIntegration",
        "create_wan_transformer",
        "count_parameters"
    ])

def check_dependencies():
    """Check if all required dependencies are available."""
    deps_status = {
        "torch": False,
        "diffusers": False,
        "transformers": False,
        "accelerate": False,
        "optimum": False,
        "peft": False,
        "bitsandbytes": False
    }
    
    try:
        import torch
        deps_status["torch"] = True
    except ImportError:
        pass
    
    try:
        import diffusers
        deps_status["diffusers"] = True
    except ImportError:
        pass
    
    try:
        import transformers
        deps_status["transformers"] = True
    except ImportError:
        pass
    
    try:
        import accelerate
        deps_status["accelerate"] = True
    except ImportError:
        pass
    
    try:
        import optimum
        deps_status["optimum"] = True
    except ImportError:
        pass
    
    try:
        import peft
        deps_status["peft"] = True
    except ImportError:
        pass
    
    try:
        import bitsandbytes
        deps_status["bitsandbytes"] = True
    except ImportError:
        pass
    
    return deps_status

def get_installation_instructions():
    """Get installation instructions for missing dependencies."""
    deps = check_dependencies()
    missing = [name for name, available in deps.items() if not available]
    
    if not missing:
        return "All dependencies are installed!"
    
    instructions = [
        "Missing dependencies detected. To install:",
        "",
        "1. Install PyTorch (if needed):",
        "   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
        "",
        "2. Install other dependencies:",
        "   pip install -r requirements.txt",
        "",
        f"Missing: {', '.join(missing)}"
    ]
    
    return "\n".join(instructions)

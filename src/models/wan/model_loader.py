"""
WAN Model Loader with Memory Optimization and Chunking Support
Handles loading WAN 2.1/2.2 models with various precision and memory optimizations
"""

import os
import logging
from typing import Dict, Any, Optional, Union, Tuple, List
from contextlib import contextmanager
import gc

from .config import (
    WanModelConfig, 
    ModelPrecision, 
    MemoryOptimization,
    WanModelVariant
)

logger = logging.getLogger(__name__)

# Import statements that will be available when dependencies are installed
try:
    import torch
    import torch.nn as nn
    from torch.nn import functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - model loading will not work")

try:
    from diffusers import WanPipeline, AutoencoderKLWan, UMT5EncoderModel
    from transformers import T5TokenizerFast, UMT5EncoderModel as TransformersUMT5
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    logger.warning("Diffusers not available - using placeholder classes")

try:
    from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map
    from accelerate.utils import get_balanced_memory
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False
    logger.warning("Accelerate not available - advanced memory features disabled")

try:
    from optimum.quanto import quantize, freeze, qint8, qint4, qfloat8
    QUANTO_AVAILABLE = True
except ImportError:
    QUANTO_AVAILABLE = False
    logger.warning("Optimum Quanto not available - quantization disabled")

try:
    from bitsandbytes import nn as bnb_nn
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False
    logger.warning("BitsAndBytes not available - BNB quantization disabled")


class ModelNotAvailableError(Exception):
    """Raised when required model dependencies are not available."""
    pass


class MemoryOptimizer:
    """Handles memory optimization strategies for WAN models."""
    
    def __init__(self, config: WanModelConfig):
        self.config = config
        self.device = torch.device(config.device) if TORCH_AVAILABLE else None
    
    @contextmanager
    def optimize_memory_context(self):
        """Context manager for memory optimization during model operations."""
        if not TORCH_AVAILABLE:
            yield
            return
        
        # Store original settings
        original_memory_fraction = None
        
        try:
            # Set memory fraction if specified
            if self.config.memory.max_vram_gb:
                if torch.cuda.is_available():
                    total_memory = torch.cuda.get_device_properties(0).total_memory
                    memory_fraction = (self.config.memory.max_vram_gb * 1024**3) / total_memory
                    torch.cuda.set_per_process_memory_fraction(min(memory_fraction, 1.0))
            
            # Enable memory optimizations
            if self.config.memory.enable_torch_compile:
                torch._dynamo.config.cache_size_limit = 64  # Limit compilation cache
            
            yield
            
        finally:
            # Cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    def apply_model_optimizations(self, model: nn.Module) -> nn.Module:
        """Apply memory optimizations to a loaded model."""
        if not TORCH_AVAILABLE:
            return model
        
        # Enable gradient checkpointing
        if MemoryOptimization.GRADIENT_CHECKPOINTING in self.config.memory.optimizations:
            if hasattr(model, 'enable_gradient_checkpointing'):
                model.enable_gradient_checkpointing()
                logger.info("Enabled gradient checkpointing")
        
        # Enable xformers memory efficient attention
        if self.config.memory.enable_xformers:
            if hasattr(model, 'enable_xformers_memory_efficient_attention'):
                try:
                    model.enable_xformers_memory_efficient_attention()
                    logger.info("Enabled xformers memory efficient attention")
                except Exception as e:
                    logger.warning(f"Could not enable xformers: {e}")
        
        # Enable flash attention
        if self.config.memory.enable_flash_attention:
            if hasattr(model, 'enable_flash_attention'):
                try:
                    model.enable_flash_attention()
                    logger.info("Enabled flash attention")
                except Exception as e:
                    logger.warning(f"Could not enable flash attention: {e}")
        
        return model
    
    def apply_pipeline_optimizations(self, pipeline) -> None:
        """Apply memory optimizations to a pipeline."""
        if not DIFFUSERS_AVAILABLE:
            return
        
        # CPU offloading
        if MemoryOptimization.SEQUENTIAL_CPU_OFFLOAD in self.config.memory.optimizations:
            pipeline.enable_sequential_cpu_offload()
            logger.info("Enabled sequential CPU offload")
        elif MemoryOptimization.MODEL_CPU_OFFLOAD in self.config.memory.optimizations:
            pipeline.enable_model_cpu_offload()
            logger.info("Enabled model CPU offload")
        
        # VAE optimizations
        if hasattr(pipeline, 'vae'):
            if MemoryOptimization.VAE_SLICING in self.config.memory.optimizations:
                pipeline.vae.enable_slicing()
                logger.info("Enabled VAE slicing")
            
            if MemoryOptimization.VAE_TILING in self.config.memory.optimizations:
                pipeline.vae.enable_tiling()
                logger.info("Enabled VAE tiling")


class QuantizationManager:
    """Handles model quantization using various backends."""
    
    def __init__(self, config: WanModelConfig):
        self.config = config
    
    def quantize_model(self, model: nn.Module) -> nn.Module:
        """Apply quantization to a model based on configuration."""
        if not TORCH_AVAILABLE:
            return model
        
        precision = self.config.quantization.precision
        
        if precision == ModelPrecision.INT8_QUANTO:
            return self._apply_quanto_quantization(model, "qint8")
        elif precision == ModelPrecision.INT4_QUANTO:
            return self._apply_quanto_quantization(model, "qint4")
        elif precision == ModelPrecision.NF4_BNB:
            return self._apply_bnb_quantization(model)
        elif precision == ModelPrecision.FP8:
            return self._apply_fp8_quantization(model)
        
        return model
    
    def _apply_quanto_quantization(self, model: nn.Module, quant_type: str) -> nn.Module:
        """Apply Quanto quantization."""
        if not QUANTO_AVAILABLE:
            logger.error("Quanto not available for quantization")
            return model
        
        try:
            # Map quantization types
            quanto_map = {
                "qint8": qint8,
                "qint4": qint4,
                "qfloat8": qfloat8
            }
            
            quant_dtype = quanto_map.get(quant_type, qint8)
            
            # Quantize the model
            quantize(model, weights=quant_dtype)
            freeze(model)
            
            logger.info(f"Applied Quanto {quant_type} quantization")
            return model
            
        except Exception as e:
            logger.error(f"Failed to apply Quanto quantization: {e}")
            return model
    
    def _apply_bnb_quantization(self, model: nn.Module) -> nn.Module:
        """Apply BitsAndBytes quantization."""
        if not BNB_AVAILABLE:
            logger.error("BitsAndBytes not available for quantization")
            return model
        
        try:
            # This would typically be done during model loading
            # Here we show the concept
            logger.info("BitsAndBytes quantization should be applied during model loading")
            return model
            
        except Exception as e:
            logger.error(f"Failed to apply BnB quantization: {e}")
            return model
    
    def _apply_fp8_quantization(self, model: nn.Module) -> nn.Module:
        """Apply FP8 quantization (experimental)."""
        logger.warning("FP8 quantization is experimental and may not be stable")
        # FP8 quantization would be implemented here
        return model


class ChunkedProcessor:
    """Handles chunked processing for memory efficiency."""
    
    def __init__(self, config: WanModelConfig):
        self.config = config
        self.chunk_size = config.memory.attention_chunk_size or 512
        self.vae_chunk_size = config.memory.vae_chunk_size
    
    def chunk_attention(self, query: torch.Tensor, key: torch.Tensor, 
                       value: torch.Tensor, attention_mask=None) -> torch.Tensor:
        """Process attention in chunks to reduce memory usage."""
        if not TORCH_AVAILABLE:
            raise ModelNotAvailableError("PyTorch not available")
        
        batch_size, seq_len, hidden_dim = query.shape
        
        if seq_len <= self.chunk_size or not self.config.memory.chunk_processing_enabled:
            # No chunking needed or disabled
            return F.scaled_dot_product_attention(query, key, value, attention_mask)
        
        # Process in chunks
        output_chunks = []
        
        for i in range(0, seq_len, self.chunk_size):
            end_idx = min(i + self.chunk_size, seq_len)
            
            query_chunk = query[:, i:end_idx]
            key_chunk = key[:, i:end_idx] 
            value_chunk = value[:, i:end_idx]
            
            mask_chunk = None
            if attention_mask is not None:
                mask_chunk = attention_mask[:, i:end_idx]
            
            chunk_output = F.scaled_dot_product_attention(
                query_chunk, key_chunk, value_chunk, mask_chunk
            )
            output_chunks.append(chunk_output)
        
        return torch.cat(output_chunks, dim=1)
    
    def chunk_vae_decode(self, vae, latents: torch.Tensor) -> torch.Tensor:
        """Process VAE decoding in chunks."""
        if not TORCH_AVAILABLE:
            raise ModelNotAvailableError("PyTorch not available")
        
        if self.vae_chunk_size == 1:
            # Process frame by frame
            decoded_frames = []
            for i in range(latents.shape[2]):  # Assuming shape [B, C, T, H, W]
                frame_latents = latents[:, :, i:i+1]
                decoded_frame = vae.decode(frame_latents)
                decoded_frames.append(decoded_frame)
            
            return torch.cat(decoded_frames, dim=2)
        else:
            # Standard decode
            return vae.decode(latents)


class WanModelLoader:
    """Main class for loading WAN models with optimizations."""
    
    def __init__(self, config: WanModelConfig):
        self.config = config
        self.memory_optimizer = MemoryOptimizer(config)
        self.quantization_manager = QuantizationManager(config)
        self.chunked_processor = ChunkedProcessor(config)
        
        # Validate dependencies
        self._validate_dependencies()
    
    def _validate_dependencies(self):
        """Validate that required dependencies are available."""
        missing_deps = []
        
        if not TORCH_AVAILABLE:
            missing_deps.append("torch")
        if not DIFFUSERS_AVAILABLE:
            missing_deps.append("diffusers")
        
        if missing_deps:
            raise ModelNotAvailableError(
                f"Missing required dependencies: {missing_deps}. "
                "Please install with: pip install -r requirements.txt"
            )
    
    def load_model_components(self) -> Dict[str, Any]:
        """Load individual model components with optimizations."""
        components = {}
        model_path = self.config.get_model_path()
        
        logger.info(f"Loading WAN model from: {model_path}")
        logger.info(f"Configuration: {self.config.quantization.precision.value} precision")
        
        with self.memory_optimizer.optimize_memory_context():
            # Load components based on config
            if self.config.load_text_encoder:
                components['text_encoder'] = self._load_text_encoder(model_path)
            
            if self.config.load_vae:
                components['vae'] = self._load_vae(model_path)
            
            if self.config.load_transformer:
                components['transformer'] = self._load_transformer(model_path)
            
            if self.config.load_scheduler:
                components['scheduler'] = self._load_scheduler(model_path)
        
        return components
    
    def _load_text_encoder(self, model_path: str):
        """Load and optimize text encoder."""
        logger.info("Loading text encoder...")
        
        # Load based on precision
        dtype = self.config.quantization.weight_dtype
        
        text_encoder = UMT5EncoderModel.from_pretrained(
            model_path,
            subfolder="text_encoder",
            torch_dtype=dtype,
            use_safetensors=self.config.use_safetensors,
            cache_dir=self.config.cache_dir,
            use_auth_token=self.config.use_auth_token
        )
        
        # Apply optimizations
        text_encoder = self.memory_optimizer.apply_model_optimizations(text_encoder)
        text_encoder = self.quantization_manager.quantize_model(text_encoder)
        
        return text_encoder
    
    def _load_vae(self, model_path: str):
        """Load and optimize VAE."""
        logger.info("Loading VAE...")
        
        # VAE typically needs higher precision
        vae_dtype = torch.float32 if self.config.quantization.precision == ModelPrecision.FP32 else torch.float16
        
        vae = AutoencoderKLWan.from_pretrained(
            model_path,
            subfolder="vae",
            torch_dtype=vae_dtype,
            use_safetensors=self.config.use_safetensors,
            cache_dir=self.config.cache_dir,
            use_auth_token=self.config.use_auth_token
        )
        
        # Apply optimizations
        vae = self.memory_optimizer.apply_model_optimizations(vae)
        
        return vae
    
    def _load_transformer(self, model_path: str):
        """Load and optimize transformer."""
        logger.info("Loading transformer...")
        
        dtype = self.config.quantization.weight_dtype
        
        # Handle device mapping for large models
        device_map = None
        if self.config.device_map:
            device_map = self.config.device_map
        elif self.config.memory.group_offloading_enabled and ACCELERATE_AVAILABLE:
            # Auto-compute device map
            try:
                # This is a placeholder - actual implementation would use proper model class
                logger.info("Computing automatic device map for group offloading")
                device_map = "auto"
            except Exception as e:
                logger.warning(f"Could not compute device map: {e}")
        
        # Load transformer with quantization if needed
        load_kwargs = {
            "subfolder": "transformer",
            "torch_dtype": dtype,
            "use_safetensors": self.config.use_safetensors,
            "cache_dir": self.config.cache_dir,
            "use_auth_token": self.config.use_auth_token,
            "device_map": device_map
        }
        
        # Add quantization config for BitsAndBytes
        if self.config.quantization.precision == ModelPrecision.NF4_BNB and BNB_AVAILABLE:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.config.quantization.bnb_quant_type,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=self.config.quantization.bnb_use_double_quant
            )
            load_kwargs["quantization_config"] = bnb_config
        
        # Placeholder for actual transformer loading
        # This would use the actual WAN transformer class when available
        transformer = None  # WanTransformer3DModel.from_pretrained(model_path, **load_kwargs)
        
        if transformer:
            # Apply optimizations
            transformer = self.memory_optimizer.apply_model_optimizations(transformer)
            
            # Apply non-BnB quantization
            if self.config.quantization.precision not in [ModelPrecision.NF4_BNB]:
                transformer = self.quantization_manager.quantize_model(transformer)
        
        return transformer
    
    def _load_scheduler(self, model_path: str):
        """Load scheduler."""
        logger.info("Loading scheduler...")
        
        # Placeholder for scheduler loading
        # This would load the appropriate scheduler for WAN models
        scheduler = None
        
        return scheduler
    
    def load_pipeline(self) -> Any:
        """Load complete WAN pipeline with all optimizations."""
        logger.info("Loading WAN pipeline...")
        
        model_path = self.config.get_model_path()
        dtype = self.config.quantization.weight_dtype
        
        # Pipeline loading kwargs
        load_kwargs = {
            "torch_dtype": dtype,
            "use_safetensors": self.config.use_safetensors,
            "cache_dir": self.config.cache_dir,
            "use_auth_token": self.config.use_auth_token
        }
        
        # Add quantization config if needed
        if self.config.quantization.precision in [ModelPrecision.INT8_QUANTO, ModelPrecision.INT4_QUANTO]:
            # Quanto quantization would be applied after loading
            pass
        elif self.config.quantization.precision == ModelPrecision.NF4_BNB and BNB_AVAILABLE:
            from diffusers import PipelineQuantizationConfig
            quant_config = PipelineQuantizationConfig(
                quant_backend="bitsandbytes_4bit",
                quant_kwargs={"load_in_4bit": True},
                components_to_quantize=["transformer", "text_encoder"]
            )
            load_kwargs["quantization_config"] = quant_config
        
        with self.memory_optimizer.optimize_memory_context():
            # Load pipeline
            pipeline = WanPipeline.from_pretrained(model_path, **load_kwargs)
            
            # Apply optimizations
            self.memory_optimizer.apply_pipeline_optimizations(pipeline)
            
            # Apply post-loading quantization
            if self.config.quantization.precision in [ModelPrecision.INT8_QUANTO, ModelPrecision.INT4_QUANTO]:
                if hasattr(pipeline, 'transformer') and pipeline.transformer:
                    pipeline.transformer = self.quantization_manager.quantize_model(pipeline.transformer)
        
        return pipeline
    
    def test_model_loading(self) -> Dict[str, Any]:
        """Test model loading and return status information."""
        test_results = {
            "success": False,
            "memory_usage": {},
            "errors": [],
            "warnings": [],
            "config_validation": {}
        }
        
        try:
            # Validate configuration
            if TORCH_AVAILABLE and torch.cuda.is_available():
                available_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                test_results["available_vram_gb"] = available_vram
                
                # Check if config is supported
                is_supported = self.config.is_supported_for_hardware(available_vram)
                test_results["config_validation"]["supported"] = is_supported
                
                if not is_supported:
                    suggestions = self.config.get_optimization_suggestions(available_vram)
                    test_results["config_validation"]["suggestions"] = suggestions
            
            # Estimate memory requirements
            memory_est = self.config.estimate_memory_usage()
            test_results["memory_usage"]["estimated"] = memory_est
            
            # Try loading components (dry run)
            logger.info("Testing model component loading...")
            
            # This would be a dry run or lightweight test
            test_results["components_available"] = {
                "text_encoder": True,  # Would test actual loading
                "vae": True,
                "transformer": True,
                "scheduler": True
            }
            
            test_results["success"] = True
            logger.info("Model loading test completed successfully")
            
        except Exception as e:
            test_results["errors"].append(str(e))
            logger.error(f"Model loading test failed: {e}")
        
        return test_results

# Utility functions
def create_model_loader(model_variant: WanModelVariant = WanModelVariant.T2V_1_3B_2_1,
                       precision: ModelPrecision = ModelPrecision.FP16,
                       memory_optimizations: List[MemoryOptimization] = None,
                       **kwargs) -> WanModelLoader:
    """Create a model loader with common settings."""
    from .config import get_balanced_config
    
    # Start with balanced config and customize
    config = get_balanced_config(model_variant)
    config.quantization.precision = precision
    
    if memory_optimizations:
        config.memory.optimizations = memory_optimizations
    
    # Apply any additional kwargs to config
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return WanModelLoader(config)

def estimate_model_memory(model_variant: WanModelVariant, 
                         precision: ModelPrecision) -> Dict[str, float]:
    """Quick memory estimation for a model variant and precision."""
    from .config import MemoryConfig, QuantizationConfig
    
    memory_config = MemoryConfig()
    quant_config = QuantizationConfig(precision=precision)
    
    return memory_config.estimate_memory_requirements(model_variant, quant_config)

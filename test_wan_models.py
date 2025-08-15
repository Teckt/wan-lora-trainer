"""
Test script for WAN model configuration and loading
Run this to validate the model system works correctly
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from models.wan.config import (
    WanModelConfig,
    WanModelVariant,
    ModelPrecision,
    MemoryOptimization,
    get_balanced_config
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_configuration_system():
    """Test the WAN model configuration system."""
    logger.info("Testing WAN model configuration system...")
    
    try:
        # Test basic configuration
        config = WanModelConfig()
        logger.info(f"Default config created: {config.model_variant.value}")
        
        # Test balanced config
        balanced_config = get_balanced_config(WanModelVariant.T2V_1_3B_2_1)
        logger.info(f"Balanced config: {balanced_config.quantization.precision.value}")
        
        # Test memory estimation
        memory_est = config.estimate_memory_usage()
        logger.info(f"Memory estimation: {memory_est}")
        
        # Test optimization suggestions
        suggestions = config.get_optimization_suggestions(vram_gb=8)
        logger.info(f"Optimization suggestions for 8GB VRAM: {suggestions}")
        
        # Test hardware compatibility
        is_supported = config.is_supported_for_hardware(vram_gb=24)
        logger.info(f"Supported on 24GB hardware: {is_supported}")
        
        return True, "Configuration system tests passed"
        
    except Exception as e:
        return False, f"Configuration test failed: {str(e)}"

def test_model_loader_creation():
    """Test model loader creation without dependencies."""
    logger.info("Testing model loader creation...")
    
    try:
        from models.wan.model_loader import WanModelLoader
        
        # Create configuration
        config = get_balanced_config(WanModelVariant.T2V_1_3B_2_1)
        config.quantization.precision = ModelPrecision.FP16
        config.memory.optimizations = [
            MemoryOptimization.GRADIENT_CHECKPOINTING,
            MemoryOptimization.VAE_SLICING
        ]
        
        # This should work even without PyTorch installed
        # as the class handles missing dependencies gracefully
        try:
            loader = WanModelLoader(config)
            return False, "Expected ModelNotAvailableError but didn't get one"
        except Exception as e:
            if "Missing required dependencies" in str(e):
                logger.info("Correctly detected missing dependencies")
                return True, "Model loader dependency checking works"
            else:
                return False, f"Unexpected error: {str(e)}"
        
    except Exception as e:
        return False, f"Model loader test failed: {str(e)}"

def test_memory_optimization_configs():
    """Test different memory optimization configurations."""
    logger.info("Testing memory optimization configurations...")
    
    try:
        # Test configurations for different VRAM sizes
        vram_configs = {
            "4GB": {
                "precision": ModelPrecision.INT8_QUANTO,
                "optimizations": [
                    MemoryOptimization.SEQUENTIAL_CPU_OFFLOAD,
                    MemoryOptimization.VAE_SLICING,
                    MemoryOptimization.VAE_TILING,
                    MemoryOptimization.GRADIENT_CHECKPOINTING
                ]
            },
            "8GB": {
                "precision": ModelPrecision.INT4_QUANTO,
                "optimizations": [
                    MemoryOptimization.MODEL_CPU_OFFLOAD,
                    MemoryOptimization.VAE_SLICING,
                    MemoryOptimization.GRADIENT_CHECKPOINTING
                ]
            },
            "16GB": {
                "precision": ModelPrecision.FP16,
                "optimizations": [
                    MemoryOptimization.GRADIENT_CHECKPOINTING,
                    MemoryOptimization.VAE_SLICING
                ]
            },
            "24GB": {
                "precision": ModelPrecision.FP16,
                "optimizations": [
                    MemoryOptimization.GRADIENT_CHECKPOINTING
                ]
            }
        }
        
        results = {}
        
        for vram, settings in vram_configs.items():
            config = get_balanced_config(WanModelVariant.T2V_1_3B_2_1)
            config.quantization.precision = settings["precision"]
            config.memory.optimizations = settings["optimizations"]
            
            memory_est = config.estimate_memory_usage()
            results[vram] = {
                "precision": settings["precision"].value,
                "memory_est": memory_est,
                "optimizations": len(settings["optimizations"])
            }
        
        logger.info("Memory optimization results:")
        for vram, result in results.items():
            logger.info(f"  {vram}: {result['precision']}, {result['memory_est']['total_gb']:.1f}GB estimated")
        
        return True, "Memory optimization tests passed"
        
    except Exception as e:
        return False, f"Memory optimization test failed: {str(e)}"

def test_dependency_checking():
    """Test dependency checking functions."""
    logger.info("Testing dependency checking...")
    
    try:
        from models.wan import check_dependencies, get_installation_instructions
        
        deps = check_dependencies()
        logger.info(f"Dependency status: {deps}")
        
        instructions = get_installation_instructions()
        logger.info("Installation instructions available")
        
        # Count available dependencies
        available_deps = sum(1 for available in deps.values() if available)
        total_deps = len(deps)
        
        logger.info(f"Dependencies available: {available_deps}/{total_deps}")
        
        return True, f"Dependency checking works ({available_deps}/{total_deps} available)"
        
    except Exception as e:
        return False, f"Dependency checking failed: {str(e)}"

def test_variant_configurations():
    """Test different model variant configurations."""
    logger.info("Testing model variant configurations...")
    
    try:
        variants = [
            WanModelVariant.T2V_1_3B_2_1,
            WanModelVariant.T2V_14B_2_1,
            WanModelVariant.I2V_1_3B_2_1,
            WanModelVariant.I2V_14B_2_1
        ]
        
        results = {}
        
        for variant in variants:
            config = get_balanced_config(variant)
            memory_est = config.estimate_memory_usage()
            
            results[variant.value] = {
                "model_size": variant.value.split("_")[1],
                "memory_gb": memory_est["total_gb"],
                "supports_8gb": config.is_supported_for_hardware(8),
                "supports_24gb": config.is_supported_for_hardware(24)
            }
        
        logger.info("Variant test results:")
        for variant, result in results.items():
            logger.info(f"  {variant}: {result['memory_gb']:.1f}GB, 8GB:{result['supports_8gb']}, 24GB:{result['supports_24gb']}")
        
        return True, "Variant configuration tests passed"
        
    except Exception as e:
        return False, f"Variant test failed: {str(e)}"

def run_all_tests():
    """Run all tests and report results."""
    logger.info("=" * 60)
    logger.info("WAN Model System Test Suite")
    logger.info("=" * 60)
    
    tests = [
        ("Configuration System", test_configuration_system),
        ("Model Loader Creation", test_model_loader_creation),
        ("Memory Optimizations", test_memory_optimization_configs),
        ("Dependency Checking", test_dependency_checking),
        ("Variant Configurations", test_variant_configurations)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\nRunning: {test_name}")
        logger.info("-" * 40)
        
        try:
            success, message = test_func()
            results.append((test_name, success, message))
            
            if success:
                logger.info(f"✓ PASSED: {message}")
            else:
                logger.error(f"✗ FAILED: {message}")
                
        except Exception as e:
            results.append((test_name, False, f"Exception: {str(e)}"))
            logger.error(f"✗ ERROR: {str(e)}")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for test_name, success, message in results:
        status = "PASS" if success else "FAIL"
        logger.info(f"{status:4} | {test_name}")
    
    logger.info(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! WAN model system is ready.")
        logger.info("\nNext steps:")
        logger.info("1. Install dependencies: pip install -r requirements.txt")
        logger.info("2. Test actual model loading once dependencies are installed")
    else:
        logger.warning("⚠️  Some tests failed. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

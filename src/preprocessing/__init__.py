"""
WAN LoRA Trainer Preprocessing Module

This module provides comprehensive video and image preprocessing capabilities
for training LoRA adapters on WAN 2.1 and 2.2 video models.

Key Features:
- Video frame extraction and processing
- Image preprocessing for I2V training
- Dataset management and validation
- Batch processing with caching
- Support for T2V and I2V training modes

Usage:
    from src.preprocessing import PreprocessingPipeline, DatasetManager
    
    # Create preprocessing pipeline
    pipeline = PreprocessingPipeline(
        dataset_dir="./dataset",
        target_size=(512, 512),
        max_frames=16
    )
    
    # Process a folder of videos/images
    results = pipeline.process_folder(
        folder_path="./raw_data",
        dataset_name="my_dataset",
        default_prompt="A beautiful video"
    )
"""

from .core import VideoProcessor, ImageProcessor, get_supported_files
from .dataset import DatasetManager, TrainingSample
from .pipeline import PreprocessingPipeline

__all__ = [
    "VideoProcessor",
    "ImageProcessor", 
    "DatasetManager",
    "TrainingSample",
    "PreprocessingPipeline",
    "get_supported_files"
]

__version__ = "0.1.0"

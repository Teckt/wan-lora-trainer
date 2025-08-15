"""
Main preprocessing pipeline for WAN LoRA training data preparation.
Orchestrates video/image processing and dataset creation.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from .core import VideoProcessor, ImageProcessor
from .dataset import DatasetManager, TrainingSample

logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """
    Main preprocessing pipeline for WAN LoRA training.
    Handles batch processing of videos and images.
    """
    
    def __init__(
        self,
        dataset_dir: Union[str, Path],
        target_size: Tuple[int, int] = (512, 512),
        max_frames: int = 16,
        fps: Optional[float] = None,
        crop_method: str = "resize",
        max_workers: int = 4
    ):
        self.dataset_dir = Path(dataset_dir)
        self.target_size = target_size
        self.max_frames = max_frames
        self.fps = fps
        self.crop_method = crop_method
        self.max_workers = max_workers
        
        # Initialize processors
        self.video_processor = VideoProcessor(
            target_size=target_size,
            max_frames=max_frames,
            fps=fps
        )
        
        self.image_processor = ImageProcessor(
            target_size=target_size
        )
        
        # Initialize dataset manager
        self.dataset_manager = DatasetManager(dataset_dir)
        
        # Processing cache directory
        self.cache_dir = self.dataset_dir / "processed_cache"
        self.cache_dir.mkdir(exist_ok=True)
    
    def process_single_video(
        self, 
        video_path: Union[str, Path],
        prompt: str = "",
        negative_prompt: str = "",
        sample_id: Optional[str] = None,
        dataset_name: str = "default"
    ) -> Optional[TrainingSample]:
        """Process a single video file."""
        video_path = Path(video_path)
        
        if not video_path.exists():
            logger.error(f"Video file not found: {video_path}")
            return None
        
        try:
            # Check if already processed
            cache_file = self.cache_dir / f"{video_path.stem}_video.npy"
            
            if not cache_file.exists():
                logger.info(f"Processing video: {video_path.name}")
                start_time = time.time()
                
                # Process video
                video_data = self.video_processor.process_video(
                    video_path, 
                    crop_method=self.crop_method
                )
                
                # Save to cache
                import numpy as np
                np.save(cache_file, video_data)
                
                processing_time = time.time() - start_time
                logger.info(f"Processed {video_path.name} in {processing_time:.2f}s")
            else:
                logger.info(f"Using cached data for: {video_path.name}")
            
            # Create training sample
            sample = TrainingSample(
                video_path=str(video_path),
                prompt=prompt,
                negative_prompt=negative_prompt,
                sample_id=sample_id or f"{dataset_name}_{video_path.stem}",
                dataset_name=dataset_name,
                target_size=self.target_size,
                max_frames=self.max_frames,
                fps=self.fps,
                crop_method=self.crop_method
            )
            
            return sample
            
        except Exception as e:
            logger.error(f"Error processing video {video_path}: {e}")
            return None
    
    def process_single_image(
        self,
        image_path: Union[str, Path],
        prompt: str = "",
        negative_prompt: str = "",
        sample_id: Optional[str] = None,
        dataset_name: str = "default"
    ) -> Optional[TrainingSample]:
        """Process a single image file."""
        image_path = Path(image_path)
        
        if not image_path.exists():
            logger.error(f"Image file not found: {image_path}")
            return None
        
        try:
            # Check if already processed
            cache_file = self.cache_dir / f"{image_path.stem}_image.npy"
            
            if not cache_file.exists():
                logger.info(f"Processing image: {image_path.name}")
                start_time = time.time()
                
                # Process image
                image_data = self.image_processor.process_image(
                    image_path,
                    crop_method=self.crop_method
                )
                
                # Save to cache
                import numpy as np
                np.save(cache_file, image_data)
                
                processing_time = time.time() - start_time
                logger.info(f"Processed {image_path.name} in {processing_time:.2f}s")
            else:
                logger.info(f"Using cached data for: {image_path.name}")
            
            # Create training sample
            sample = TrainingSample(
                image_path=str(image_path),
                prompt=prompt,
                negative_prompt=negative_prompt,
                sample_id=sample_id or f"{dataset_name}_{image_path.stem}",
                dataset_name=dataset_name,
                target_size=self.target_size,
                crop_method=self.crop_method
            )
            
            return sample
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return None
    
    def process_folder(
        self,
        folder_path: Union[str, Path],
        dataset_name: str,
        default_prompt: str = "",
        default_negative_prompt: str = "",
        file_types: str = "both",  # "video", "image", "both"
        use_parallel: bool = True
    ) -> Dict[str, Any]:
        """
        Process all supported files in a folder.
        
        Returns:
            Dictionary with processing results and statistics
        """
        folder_path = Path(folder_path)
        
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        from .core import get_supported_files
        
        # Get all supported files
        supported_files = get_supported_files(folder_path, file_types)
        
        results = {
            "processed_samples": [],
            "failed_files": [],
            "total_files": 0,
            "successful": 0,
            "failed": 0,
            "processing_time": 0
        }
        
        start_time = time.time()
        
        # Prepare file list for processing
        files_to_process = []
        
        if "videos" in supported_files:
            for video_path in supported_files["videos"]:
                files_to_process.append(("video", video_path))
                results["total_files"] += 1
        
        if "images" in supported_files:
            for image_path in supported_files["images"]:
                files_to_process.append(("image", image_path))
                results["total_files"] += 1
        
        logger.info(f"Processing {results['total_files']} files from {folder_path}")
        
        if use_parallel and self.max_workers > 1:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_file = {}
                
                for file_type, file_path in files_to_process:
                    if file_type == "video":
                        future = executor.submit(
                            self.process_single_video,
                            file_path,
                            default_prompt,
                            default_negative_prompt,
                            None,
                            dataset_name
                        )
                    else:  # image
                        future = executor.submit(
                            self.process_single_image,
                            file_path,
                            default_prompt,
                            default_negative_prompt,
                            None,
                            dataset_name
                        )
                    
                    future_to_file[future] = (file_type, file_path)
                
                # Collect results
                for future in as_completed(future_to_file):
                    file_type, file_path = future_to_file[future]
                    try:
                        sample = future.result()
                        if sample:
                            results["processed_samples"].append(sample)
                            results["successful"] += 1
                        else:
                            results["failed_files"].append(str(file_path))
                            results["failed"] += 1
                    except Exception as e:
                        logger.error(f"Error processing {file_path}: {e}")
                        results["failed_files"].append(str(file_path))
                        results["failed"] += 1
        
        else:
            # Sequential processing
            for file_type, file_path in files_to_process:
                try:
                    if file_type == "video":
                        sample = self.process_single_video(
                            file_path,
                            default_prompt,
                            default_negative_prompt,
                            None,
                            dataset_name
                        )
                    else:  # image
                        sample = self.process_single_image(
                            file_path,
                            default_prompt,
                            default_negative_prompt,
                            None,
                            dataset_name
                        )
                    
                    if sample:
                        results["processed_samples"].append(sample)
                        results["successful"] += 1
                    else:
                        results["failed_files"].append(str(file_path))
                        results["failed"] += 1
                
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    results["failed_files"].append(str(file_path))
                    results["failed"] += 1
        
        # Add processed samples to dataset
        for sample in results["processed_samples"]:
            self.dataset_manager.add_sample(sample)
        
        # Save dataset
        self.dataset_manager.save_dataset()
        
        results["processing_time"] = time.time() - start_time
        
        logger.info(
            f"Completed processing: {results['successful']} successful, "
            f"{results['failed']} failed, {results['processing_time']:.2f}s total"
        )
        
        return results
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get comprehensive dataset information."""
        return {
            "dataset_path": str(self.dataset_dir),
            "cache_path": str(self.cache_dir),
            "statistics": self.dataset_manager.get_statistics(),
            "validation": self.dataset_manager.validate_dataset(),
            "processing_config": {
                "target_size": self.target_size,
                "max_frames": self.max_frames,
                "fps": self.fps,
                "crop_method": self.crop_method,
                "max_workers": self.max_workers
            }
        }
    
    def clear_cache(self) -> int:
        """Clear processing cache and return number of files removed."""
        removed_count = 0
        
        if self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("*.npy"):
                try:
                    cache_file.unlink()
                    removed_count += 1
                except Exception as e:
                    logger.error(f"Could not remove cache file {cache_file}: {e}")
        
        logger.info(f"Cleared {removed_count} cache files")
        return removed_count

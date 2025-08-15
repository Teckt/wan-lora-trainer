"""
Dataset management for WAN LoRA training.
Handles training samples, metadata, and batch processing.
"""

import json
import jsonlines
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class TrainingSample:
    """
    Represents a single training sample for WAN LoRA training.
    Supports both T2V (text-to-video) and I2V (image-to-video) training.
    """
    # File paths
    video_path: Optional[str] = None
    image_path: Optional[str] = None  # For I2V conditioning
    
    # Training data
    prompt: str = ""
    negative_prompt: str = ""
    
    # Metadata
    sample_id: str = ""
    dataset_name: str = ""
    
    # Processing parameters
    target_size: Tuple[int, int] = (512, 512)
    max_frames: int = 16
    fps: Optional[float] = None
    crop_method: str = "resize"  # "resize" or "center_crop"
    
    # Training parameters
    weight: float = 1.0
    
    @property
    def is_video_sample(self) -> bool:
        """Check if this is a video training sample (T2V)."""
        return self.video_path is not None
    
    @property
    def is_image_sample(self) -> bool:
        """Check if this is an image training sample (I2V)."""
        return self.image_path is not None and self.video_path is None
    
    @property
    def is_i2v_sample(self) -> bool:
        """Check if this is an I2V sample (image conditioning + video)."""
        return self.image_path is not None and self.video_path is not None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingSample':
        """Create from dictionary."""
        return cls(**data)
    
    def validate(self) -> List[str]:
        """Validate the training sample and return any errors."""
        errors = []
        
        if not self.prompt.strip():
            errors.append("Prompt cannot be empty")
        
        if not self.is_video_sample and not self.is_image_sample:
            errors.append("Must have either video_path or image_path")
        
        if self.video_path and not Path(self.video_path).exists():
            errors.append(f"Video file not found: {self.video_path}")
        
        if self.image_path and not Path(self.image_path).exists():
            errors.append(f"Image file not found: {self.image_path}")
        
        if self.weight <= 0:
            errors.append("Weight must be positive")
        
        if self.max_frames <= 0:
            errors.append("Max frames must be positive")
        
        if self.fps is not None and self.fps <= 0:
            errors.append("FPS must be positive")
        
        return errors


class DatasetManager:
    """Manages training datasets for WAN LoRA training."""
    
    def __init__(self, dataset_dir: Union[str, Path]):
        self.dataset_dir = Path(dataset_dir)
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        
        self.samples_file = self.dataset_dir / "samples.jsonl"
        self.metadata_file = self.dataset_dir / "metadata.json"
        
        self.samples: List[TrainingSample] = []
        self.metadata: Dict[str, Any] = {}
        
        self.load_dataset()
    
    def load_dataset(self):
        """Load existing dataset from disk."""
        # Load metadata
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load metadata: {e}")
                self.metadata = {}
        
        # Load samples
        if self.samples_file.exists():
            try:
                with jsonlines.open(self.samples_file, 'r') as reader:
                    for line in reader:
                        sample = TrainingSample.from_dict(line)
                        self.samples.append(sample)
                logger.info(f"Loaded {len(self.samples)} samples from {self.samples_file}")
            except Exception as e:
                logger.error(f"Could not load samples: {e}")
                self.samples = []
    
    def save_dataset(self):
        """Save dataset to disk."""
        # Save metadata
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Could not save metadata: {e}")
        
        # Save samples
        try:
            with jsonlines.open(self.samples_file, 'w') as writer:
                for sample in self.samples:
                    writer.write(sample.to_dict())
            logger.info(f"Saved {len(self.samples)} samples to {self.samples_file}")
        except Exception as e:
            logger.error(f"Could not save samples: {e}")
    
    def add_sample(self, sample: TrainingSample) -> bool:
        """Add a training sample to the dataset."""
        errors = sample.validate()
        if errors:
            logger.error(f"Invalid sample: {errors}")
            return False
        
        # Generate sample ID if not provided
        if not sample.sample_id:
            sample.sample_id = f"sample_{len(self.samples):06d}"
        
        self.samples.append(sample)
        return True
    
    def remove_sample(self, sample_id: str) -> bool:
        """Remove a sample by ID."""
        for i, sample in enumerate(self.samples):
            if sample.sample_id == sample_id:
                del self.samples[i]
                return True
        return False
    
    def get_sample(self, sample_id: str) -> Optional[TrainingSample]:
        """Get a sample by ID."""
        for sample in self.samples:
            if sample.sample_id == sample_id:
                return sample
        return None
    
    def validate_dataset(self) -> Dict[str, List[str]]:
        """Validate all samples in the dataset."""
        validation_results = {
            "valid_samples": [],
            "invalid_samples": [],
            "errors": []
        }
        
        for sample in self.samples:
            errors = sample.validate()
            if errors:
                validation_results["invalid_samples"].append(sample.sample_id)
                validation_results["errors"].extend([f"{sample.sample_id}: {error}" for error in errors])
            else:
                validation_results["valid_samples"].append(sample.sample_id)
        
        return validation_results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        stats = {
            "total_samples": len(self.samples),
            "video_samples": 0,
            "image_samples": 0,
            "i2v_samples": 0,
            "avg_prompt_length": 0,
            "unique_prompts": 0,
            "file_formats": {"videos": {}, "images": {}}
        }
        
        prompts = set()
        total_prompt_length = 0
        
        for sample in self.samples:
            if sample.is_i2v_sample:
                stats["i2v_samples"] += 1
            elif sample.is_video_sample:
                stats["video_samples"] += 1
            elif sample.is_image_sample:
                stats["image_samples"] += 1
            
            prompts.add(sample.prompt)
            total_prompt_length += len(sample.prompt)
            
            # Track file formats
            if sample.video_path:
                ext = Path(sample.video_path).suffix.lower()
                stats["file_formats"]["videos"][ext] = stats["file_formats"]["videos"].get(ext, 0) + 1
            
            if sample.image_path:
                ext = Path(sample.image_path).suffix.lower()
                stats["file_formats"]["images"][ext] = stats["file_formats"]["images"].get(ext, 0) + 1
        
        if len(self.samples) > 0:
            stats["avg_prompt_length"] = total_prompt_length / len(self.samples)
        
        stats["unique_prompts"] = len(prompts)
        
        return stats
    
    def create_from_folder(
        self,
        folder_path: Union[str, Path],
        dataset_name: str,
        default_prompt: str = "",
        auto_caption: bool = False,
        file_types: str = "both"  # "video", "image", "both"
    ) -> int:
        """
        Create dataset samples from a folder of videos/images.
        
        Args:
            folder_path: Path to folder containing media files
            dataset_name: Name for this dataset
            default_prompt: Default prompt for all samples
            auto_caption: Whether to attempt automatic captioning (future feature)
            file_types: Types of files to process
        
        Returns:
            Number of samples created
        """
        from .core import get_supported_files
        
        folder_path = Path(folder_path)
        
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        # Get supported files
        supported_files = get_supported_files(folder_path, file_types)
        
        samples_created = 0
        
        # Process videos
        if "videos" in supported_files:
            for video_path in supported_files["videos"]:
                sample = TrainingSample(
                    video_path=str(video_path),
                    prompt=default_prompt,
                    sample_id=f"{dataset_name}_{video_path.stem}",
                    dataset_name=dataset_name
                )
                
                if self.add_sample(sample):
                    samples_created += 1
                    logger.info(f"Added video sample: {video_path.name}")
        
        # Process images
        if "images" in supported_files:
            for image_path in supported_files["images"]:
                sample = TrainingSample(
                    image_path=str(image_path),
                    prompt=default_prompt,
                    sample_id=f"{dataset_name}_{image_path.stem}",
                    dataset_name=dataset_name
                )
                
                if self.add_sample(sample):
                    samples_created += 1
                    logger.info(f"Added image sample: {image_path.name}")
        
        # Update metadata
        self.metadata.update({
            "dataset_name": dataset_name,
            "created_from_folder": str(folder_path),
            "total_samples": len(self.samples),
            "last_updated": None  # Would use datetime if imported
        })
        
        # Save the dataset
        self.save_dataset()
        
        logger.info(f"Created {samples_created} samples from {folder_path}")
        return samples_created

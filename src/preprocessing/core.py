"""
Core video and image preprocessing utilities for WAN LoRA training.
Supports both T2V (text-to-video) and I2V (image-to-video) training data preparation.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Union, Dict, Any
import torch
from PIL import Image
import imageio
import logging

logger = logging.getLogger(__name__)

class VideoProcessor:
    """Core video preprocessing class for WAN model training data preparation."""
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (512, 512),
        max_frames: int = 16,
        fps: Optional[float] = None,
        video_extensions: List[str] = None
    ):
        self.target_size = target_size
        self.max_frames = max_frames
        self.fps = fps
        self.video_extensions = video_extensions or ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        
    def extract_frames(self, video_path: Union[str, Path]) -> List[np.ndarray]:
        """Extract frames from video file."""
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        if video_path.suffix.lower() not in self.video_extensions:
            raise ValueError(f"Unsupported video format: {video_path.suffix}")
        
        try:
            # Use imageio for better format support
            reader = imageio.get_reader(str(video_path))
            frames = []
            
            # Calculate frame sampling if fps is specified
            if self.fps:
                video_fps = reader.get_meta_data()['fps']
                frame_step = max(1, int(video_fps / self.fps))
            else:
                frame_step = 1
            
            frame_count = 0
            for i, frame in enumerate(reader):
                if i % frame_step == 0:
                    frames.append(frame)
                    frame_count += 1
                    
                    if frame_count >= self.max_frames:
                        break
            
            reader.close()
            
            if len(frames) == 0:
                raise ValueError(f"No frames extracted from video: {video_path}")
                
            logger.info(f"Extracted {len(frames)} frames from {video_path}")
            return frames
            
        except Exception as e:
            logger.error(f"Error extracting frames from {video_path}: {e}")
            raise
    
    def resize_frame(self, frame: np.ndarray, maintain_aspect: bool = True) -> np.ndarray:
        """Resize a single frame to target size."""
        h, w = frame.shape[:2]
        target_h, target_w = self.target_size
        
        if maintain_aspect:
            # Calculate scaling to fit within target size
            scale = min(target_w / w, target_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Resize frame
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
            # Create black canvas and center the resized frame
            canvas = np.zeros((target_h, target_w, 3), dtype=frame.dtype)
            start_y = (target_h - new_h) // 2
            start_x = (target_w - new_w) // 2
            canvas[start_y:start_y + new_h, start_x:start_x + new_w] = resized
            
            return canvas
        else:
            # Direct resize without maintaining aspect ratio
            return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    
    def center_crop_frame(self, frame: np.ndarray) -> np.ndarray:
        """Center crop frame to target size."""
        h, w = frame.shape[:2]
        target_h, target_w = self.target_size
        
        if h < target_h or w < target_w:
            # Pad if frame is smaller than target
            pad_h = max(0, target_h - h)
            pad_w = max(0, target_w - w)
            frame = np.pad(frame, ((pad_h//2, pad_h - pad_h//2), 
                                   (pad_w//2, pad_w - pad_w//2), (0, 0)), 
                          mode='constant', constant_values=0)
            h, w = frame.shape[:2]
        
        # Center crop
        start_y = (h - target_h) // 2
        start_x = (w - target_w) // 2
        return frame[start_y:start_y + target_h, start_x:start_x + target_w]
    
    def process_video(
        self, 
        video_path: Union[str, Path],
        crop_method: str = "resize"  # "resize", "center_crop"
    ) -> np.ndarray:
        """Process a video file into training-ready tensor format."""
        frames = self.extract_frames(video_path)
        processed_frames = []
        
        for frame in frames:
            if crop_method == "resize":
                processed_frame = self.resize_frame(frame, maintain_aspect=True)
            elif crop_method == "center_crop":
                processed_frame = self.center_crop_frame(frame)
            else:
                raise ValueError(f"Unknown crop method: {crop_method}")
            
            # Convert BGR to RGB if needed (OpenCV uses BGR)
            if len(processed_frame.shape) == 3 and processed_frame.shape[2] == 3:
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            
            processed_frames.append(processed_frame)
        
        # Convert to numpy array and normalize to [0, 1]
        video_array = np.stack(processed_frames, axis=0)
        video_array = video_array.astype(np.float32) / 255.0
        
        # Return shape: (frames, height, width, channels)
        return video_array


class ImageProcessor:
    """Image preprocessing class for I2V training data preparation."""
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (512, 512),
        image_extensions: List[str] = None
    ):
        self.target_size = target_size
        self.image_extensions = image_extensions or ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    
    def load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """Load image from file."""
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        if image_path.suffix.lower() not in self.image_extensions:
            raise ValueError(f"Unsupported image format: {image_path.suffix}")
        
        try:
            image = Image.open(image_path)
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return np.array(image)
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            raise
    
    def resize_image(self, image: np.ndarray, maintain_aspect: bool = True) -> np.ndarray:
        """Resize image to target size."""
        h, w = image.shape[:2]
        target_h, target_w = self.target_size
        
        if maintain_aspect:
            # Calculate scaling to fit within target size
            scale = min(target_w / w, target_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Resize image
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
            # Create black canvas and center the resized image
            canvas = np.zeros((target_h, target_w, 3), dtype=image.dtype)
            start_y = (target_h - new_h) // 2
            start_x = (target_w - new_w) // 2
            canvas[start_y:start_y + new_h, start_x:start_x + new_w] = resized
            
            return canvas
        else:
            # Direct resize without maintaining aspect ratio
            return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    
    def center_crop_image(self, image: np.ndarray) -> np.ndarray:
        """Center crop image to target size."""
        h, w = image.shape[:2]
        target_h, target_w = self.target_size
        
        if h < target_h or w < target_w:
            # Pad if image is smaller than target
            pad_h = max(0, target_h - h)
            pad_w = max(0, target_w - w)
            image = np.pad(image, ((pad_h//2, pad_h - pad_h//2), 
                                   (pad_w//2, pad_w - pad_w//2), (0, 0)), 
                          mode='constant', constant_values=0)
            h, w = image.shape[:2]
        
        # Center crop
        start_y = (h - target_h) // 2
        start_x = (w - target_w) // 2
        return image[start_y:start_y + target_h, start_x:start_x + target_w]
    
    def process_image(
        self, 
        image_path: Union[str, Path],
        crop_method: str = "resize"  # "resize", "center_crop"
    ) -> np.ndarray:
        """Process an image file into training-ready format."""
        image = self.load_image(image_path)
        
        if crop_method == "resize":
            processed_image = self.resize_image(image, maintain_aspect=True)
        elif crop_method == "center_crop":
            processed_image = self.center_crop_image(image)
        else:
            raise ValueError(f"Unknown crop method: {crop_method}")
        
        # Normalize to [0, 1]
        processed_image = processed_image.astype(np.float32) / 255.0
        
        return processed_image


def get_supported_files(directory: Union[str, Path], file_types: str = "both") -> Dict[str, List[Path]]:
    """
    Get all supported video and image files from a directory.
    
    Args:
        directory: Path to search for files
        file_types: "video", "image", or "both"
    
    Returns:
        Dictionary with 'videos' and/or 'images' keys containing file paths
    """
    directory = Path(directory)
    
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    video_processor = VideoProcessor()
    image_processor = ImageProcessor()
    
    result = {}
    
    if file_types in ["video", "both"]:
        video_files = []
        for ext in video_processor.video_extensions:
            video_files.extend(directory.glob(f"*{ext}"))
            video_files.extend(directory.glob(f"*{ext.upper()}"))
        result["videos"] = sorted(video_files)
    
    if file_types in ["image", "both"]:
        image_files = []
        for ext in image_processor.image_extensions:
            image_files.extend(directory.glob(f"*{ext}"))
            image_files.extend(directory.glob(f"*{ext.upper()}"))
        result["images"] = sorted(image_files)
    
    return result

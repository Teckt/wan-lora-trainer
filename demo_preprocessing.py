"""
Demo script for WAN LoRA preprocessing modules.
This script demonstrates the key functionality without requiring dependencies.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

def demo_dataset_creation():
    """Demo dataset creation and management."""
    print("=== Dataset Management Demo ===")
    
    try:
        from src.preprocessing.dataset import DatasetManager, TrainingSample
        
        # Create a sample dataset
        dataset_dir = Path("./demo_dataset")
        dataset_manager = DatasetManager(dataset_dir)
        
        # Create some sample training samples
        samples = [
            TrainingSample(
                video_path="./sample_videos/video1.mp4",
                prompt="A person walking in a park",
                sample_id="sample_001",
                dataset_name="demo_dataset"
            ),
            TrainingSample(
                image_path="./sample_images/image1.jpg",
                prompt="A beautiful landscape",
                sample_id="sample_002", 
                dataset_name="demo_dataset"
            ),
            TrainingSample(
                image_path="./sample_images/conditioning.jpg",
                video_path="./sample_videos/video2.mp4",
                prompt="Image to video transformation",
                sample_id="sample_003",
                dataset_name="demo_dataset"
            )
        ]
        
        # Add samples to dataset
        for sample in samples:
            success = dataset_manager.add_sample(sample)
            print(f"Added sample {sample.sample_id}: {'Success' if success else 'Failed'}")
            if sample.is_i2v_sample:
                print(f"  -> I2V sample (image conditioning + video)")
            elif sample.is_video_sample:
                print(f"  -> T2V sample (text to video)")
            elif sample.is_image_sample:
                print(f"  -> Image sample")
        
        # Get statistics
        stats = dataset_manager.get_statistics()
        print(f"\nDataset Statistics:")
        print(f"  Total samples: {stats['total_samples']}")
        print(f"  Video samples: {stats['video_samples']}")
        print(f"  Image samples: {stats['image_samples']}")
        print(f"  I2V samples: {stats['i2v_samples']}")
        print(f"  Unique prompts: {stats['unique_prompts']}")
        
        # Validation
        validation = dataset_manager.validate_dataset()
        print(f"\nValidation Results:")
        print(f"  Valid samples: {len(validation['valid_samples'])}")
        print(f"  Invalid samples: {len(validation['invalid_samples'])}")
        if validation['errors']:
            print(f"  Errors: {validation['errors'][:3]}...")  # Show first 3 errors
        
        # Save dataset
        dataset_manager.save_dataset()
        print(f"\nDataset saved to: {dataset_dir}")
        
    except ImportError as e:
        print(f"Import error (expected without dependencies): {e}")
    except Exception as e:
        print(f"Error: {e}")

def demo_preprocessing_config():
    """Demo preprocessing pipeline configuration."""
    print("\n=== Preprocessing Pipeline Demo ===")
    
    try:
        from src.preprocessing.pipeline import PreprocessingPipeline
        
        # Create preprocessing pipeline with custom settings
        pipeline = PreprocessingPipeline(
            dataset_dir="./demo_dataset",
            target_size=(512, 512),  # WAN model resolution
            max_frames=16,           # Standard for WAN models
            fps=8.0,                 # Target FPS for training
            crop_method="resize",    # Maintain aspect ratio
            max_workers=2            # Parallel processing
        )
        
        # Show configuration
        config_info = pipeline.get_dataset_info()
        print("Pipeline Configuration:")
        print(f"  Target size: {config_info['processing_config']['target_size']}")
        print(f"  Max frames: {config_info['processing_config']['max_frames']}")
        print(f"  FPS: {config_info['processing_config']['fps']}")
        print(f"  Crop method: {config_info['processing_config']['crop_method']}")
        print(f"  Max workers: {config_info['processing_config']['max_workers']}")
        
        print(f"\nDataset path: {config_info['dataset_path']}")
        print(f"Cache path: {config_info['cache_path']}")
        
    except ImportError as e:
        print(f"Import error (expected without dependencies): {e}")
    except Exception as e:
        print(f"Error: {e}")

def demo_supported_formats():
    """Demo supported file format detection."""
    print("\n=== Supported Formats Demo ===")
    
    try:
        from src.preprocessing.core import VideoProcessor, ImageProcessor
        
        # Show supported formats
        video_processor = VideoProcessor()
        image_processor = ImageProcessor()
        
        print("Supported Video Formats:")
        for ext in video_processor.video_extensions:
            print(f"  {ext}")
        
        print("\nSupported Image Formats:")
        for ext in image_processor.image_extensions:
            print(f"  {ext}")
        
    except ImportError as e:
        print(f"Import error (expected without dependencies): {e}")
    except Exception as e:
        print(f"Error: {e}")

def demo_training_sample_features():
    """Demo training sample features and validation."""
    print("\n=== Training Sample Features Demo ===")
    
    try:
        from src.preprocessing.dataset import TrainingSample
        
        # Different types of training samples
        samples = [
            # T2V (Text to Video)
            TrainingSample(
                video_path="./videos/dance.mp4",
                prompt="A person dancing energetically",
                sample_id="t2v_001",
                target_size=(512, 512),
                max_frames=16,
                fps=8.0
            ),
            
            # I2V (Image to Video) 
            TrainingSample(
                image_path="./images/portrait.jpg",
                video_path="./videos/talking.mp4",
                prompt="A person speaking from the portrait",
                sample_id="i2v_001",
                target_size=(512, 512),
                max_frames=16
            ),
            
            # Image only (for I2V conditioning)
            TrainingSample(
                image_path="./images/landscape.jpg",
                prompt="A serene mountain landscape",
                sample_id="img_001",
                weight=0.8  # Lower weight for this sample
            )
        ]
        
        print("Sample Types and Features:")
        for sample in samples:
            print(f"\nSample: {sample.sample_id}")
            print(f"  Type: ", end="")
            if sample.is_i2v_sample:
                print("I2V (Image-to-Video)")
            elif sample.is_video_sample:
                print("T2V (Text-to-Video)")
            elif sample.is_image_sample:
                print("Image Only")
            
            print(f"  Prompt: {sample.prompt}")
            print(f"  Target size: {sample.target_size}")
            print(f"  Max frames: {sample.max_frames}")
            print(f"  Weight: {sample.weight}")
            
            # Show validation (will show file not found errors for demo files)
            errors = sample.validate()
            if errors:
                print(f"  Validation errors: {len(errors)} (expected for demo)")
        
    except ImportError as e:
        print(f"Import error (expected without dependencies): {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("WAN LoRA Trainer - Preprocessing Modules Demo")
    print("=" * 50)
    
    demo_dataset_creation()
    demo_preprocessing_config()
    demo_supported_formats()
    demo_training_sample_features()
    
    print("\n" + "=" * 50)
    print("Demo completed!")
    print("\nNext steps:")
    print("1. Install requirements: pip install -r requirements.txt")
    print("2. Place videos/images in folders for processing")
    print("3. Use the preprocessing pipeline to prepare training data")
    print("4. Build the Gradio interface for easy use")

#!/usr/bin/env python3
"""
WAN 2.1 Dataset Preparation Helper
Helps prepare video datasets for LoRA training with automatic caption generation
"""

import os
import argparse
import subprocess
import json
from pathlib import Path
from typing import List, Tuple
import cv2

def get_video_info(video_path: str) -> dict:
    """Get video information using OpenCV."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    
    cap.release()
    
    return {
        'fps': fps,
        'frame_count': frame_count,
        'width': width,
        'height': height,
        'duration': duration,
        'resolution': f"{width}x{height}"
    }

def get_video_files(directory: str) -> List[str]:
    """Get all video files in directory."""
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
    video_files = []
    
    for file_path in Path(directory).rglob('*'):
        if file_path.suffix.lower() in video_extensions:
            video_files.append(str(file_path))
    
    return sorted(video_files)

def create_caption_file(video_path: str, caption: str):
    """Create caption file for video."""
    caption_path = Path(video_path).with_suffix('.txt')
    with open(caption_path, 'w', encoding='utf-8') as f:
        f.write(caption)
    print(f"Created caption: {caption_path}")

def generate_basic_caption(video_path: str, video_info: dict) -> str:
    """Generate a basic caption based on filename and video info."""
    filename = Path(video_path).stem
    
    # Basic processing of filename
    caption_parts = []
    
    # Remove common video prefixes/suffixes
    clean_name = filename.lower()
    for remove in ['video', 'clip', 'sample', 'test', '_', '-']:
        clean_name = clean_name.replace(remove, ' ')
    
    clean_name = ' '.join(clean_name.split())  # Clean extra spaces
    
    if clean_name:
        caption_parts.append(clean_name)
    
    # Add quality indicators based on resolution
    if video_info:
        width, height = video_info['width'], video_info['height']
        if width >= 1920 or height >= 1080:
            caption_parts.append("high quality")
        elif width >= 1280 or height >= 720:
            caption_parts.append("good quality")
        
        # Add duration context
        duration = video_info.get('duration', 0)
        if duration > 10:
            caption_parts.append("long sequence")
        elif duration < 3:
            caption_parts.append("short clip")
    
    caption = ', '.join(caption_parts) if caption_parts else "video content"
    return caption.capitalize()

def validate_dataset(dataset_dir: str) -> Tuple[List[str], List[str]]:
    """Validate dataset - return (valid_videos, missing_captions)."""
    video_files = get_video_files(dataset_dir)
    valid_videos = []
    missing_captions = []
    
    for video_path in video_files:
        caption_path = Path(video_path).with_suffix('.txt')
        if caption_path.exists():
            valid_videos.append(video_path)
        else:
            missing_captions.append(video_path)
    
    return valid_videos, missing_captions

def create_dataset_report(dataset_dir: str, output_file: str = None):
    """Create a detailed dataset report."""
    video_files = get_video_files(dataset_dir)
    
    if not video_files:
        print(f"No video files found in {dataset_dir}")
        return
    
    report = {
        'dataset_info': {
            'total_videos': len(video_files),
            'dataset_directory': dataset_dir
        },
        'videos': []
    }
    
    print(f"Analyzing {len(video_files)} videos...")
    
    resolutions = {}
    total_duration = 0
    
    for i, video_path in enumerate(video_files, 1):
        print(f"Processing {i}/{len(video_files)}: {Path(video_path).name}")
        
        video_info = get_video_info(video_path)
        caption_path = Path(video_path).with_suffix('.txt')
        has_caption = caption_path.exists()
        
        caption_text = ""
        if has_caption:
            try:
                with open(caption_path, 'r', encoding='utf-8') as f:
                    caption_text = f.read().strip()
            except:
                has_caption = False
        
        video_data = {
            'path': video_path,
            'filename': Path(video_path).name,
            'has_caption': has_caption,
            'caption': caption_text,
            'info': video_info
        }
        
        if video_info:
            resolution = video_info['resolution']
            resolutions[resolution] = resolutions.get(resolution, 0) + 1
            total_duration += video_info['duration']
        
        report['videos'].append(video_data)
    
    # Add summary statistics
    report['dataset_info'].update({
        'total_duration_seconds': total_duration,
        'total_duration_minutes': total_duration / 60,
        'average_duration_seconds': total_duration / len(video_files),
        'resolution_distribution': resolutions,
        'videos_with_captions': sum(1 for v in report['videos'] if v['has_caption']),
        'videos_without_captions': sum(1 for v in report['videos'] if not v['has_caption'])
    })
    
    # Print summary
    print("\n" + "="*50)
    print("DATASET ANALYSIS SUMMARY")
    print("="*50)
    print(f"Total videos: {report['dataset_info']['total_videos']}")
    print(f"Total duration: {report['dataset_info']['total_duration_minutes']:.1f} minutes")
    print(f"Average duration: {report['dataset_info']['average_duration_seconds']:.1f} seconds")
    print(f"Videos with captions: {report['dataset_info']['videos_with_captions']}")
    print(f"Videos without captions: {report['dataset_info']['videos_without_captions']}")
    
    print("\nResolution distribution:")
    for resolution, count in sorted(resolutions.items()):
        print(f"  {resolution}: {count} videos")
    
    # Save report if requested
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed report saved to: {output_file}")
    
    return report

def main():
    parser = argparse.ArgumentParser(description="WAN 2.1 Dataset Preparation Helper")
    parser.add_argument("dataset_dir", help="Directory containing video files")
    parser.add_argument("--mode", choices=['analyze', 'generate-captions', 'validate'], 
                       default='analyze', help="Operation mode")
    parser.add_argument("--output-report", help="Save detailed report to JSON file")
    parser.add_argument("--default-caption", help="Default caption for videos without captions")
    parser.add_argument("--auto-caption", action='store_true', 
                       help="Generate basic captions from filenames")
    parser.add_argument("--overwrite", action='store_true', 
                       help="Overwrite existing caption files")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset_dir):
        print(f"Error: Directory {args.dataset_dir} does not exist")
        return
    
    if args.mode == 'analyze':
        print(f"Analyzing dataset in: {args.dataset_dir}")
        create_dataset_report(args.dataset_dir, args.output_report)
        
    elif args.mode == 'validate':
        print(f"Validating dataset in: {args.dataset_dir}")
        valid_videos, missing_captions = validate_dataset(args.dataset_dir)
        
        print(f"\nValidation Results:")
        print(f"Valid videos (with captions): {len(valid_videos)}")
        print(f"Videos missing captions: {len(missing_captions)}")
        
        if missing_captions:
            print("\nVideos without captions:")
            for video_path in missing_captions:
                print(f"  {Path(video_path).name}")
        
    elif args.mode == 'generate-captions':
        video_files = get_video_files(args.dataset_dir)
        
        if not video_files:
            print(f"No video files found in {args.dataset_dir}")
            return
        
        generated_count = 0
        skipped_count = 0
        
        for video_path in video_files:
            caption_path = Path(video_path).with_suffix('.txt')
            
            # Skip if caption exists and not overwriting
            if caption_path.exists() and not args.overwrite:
                skipped_count += 1
                continue
            
            # Generate caption
            if args.default_caption:
                caption = args.default_caption
            elif args.auto_caption:
                video_info = get_video_info(video_path)
                caption = generate_basic_caption(video_path, video_info)
            else:
                # Interactive mode
                print(f"\nVideo: {Path(video_path).name}")
                video_info = get_video_info(video_path)
                if video_info:
                    print(f"Resolution: {video_info['resolution']}")
                    print(f"Duration: {video_info['duration']:.1f}s")
                    print(f"FPS: {video_info['fps']:.1f}")
                
                caption = input("Enter caption (or press Enter to skip): ").strip()
                if not caption:
                    continue
            
            create_caption_file(video_path, caption)
            generated_count += 1
        
        print(f"\nCaption generation complete:")
        print(f"Generated: {generated_count}")
        print(f"Skipped: {skipped_count}")

if __name__ == "__main__":
    main()

#!/bin/bash

# Example Dataset Creation Script for WAN 2.1 LoRA Training
# This script creates sample training data to demonstrate the workflow
# Replace with your own videos and captions for actual training

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== WAN 2.1 Example Dataset Creation ===${NC}"
echo -e "${YELLOW}This script creates sample training data for demonstration${NC}"
echo -e "${YELLOW}Replace with your own videos and captions for actual training${NC}"

# Check if we're in the musubi-tuner directory
if [ ! -f "train_wan21_lora.sh" ]; then
    echo -e "${RED}Error: Please run this script from the musubi-tuner installation directory${NC}"
    echo -e "${YELLOW}Expected to find train_wan21_lora.sh in current directory${NC}"
    exit 1
fi

# Create sample videos directory if it doesn't exist
SAMPLE_DIR="datasets/training_data/videos"
mkdir -p "$SAMPLE_DIR"

echo -e "\n${CYAN}=== Creating Sample Training Data ===${NC}"

# Function to create a sample video using FFmpeg
create_sample_video() {
    local video_name="$1"
    local caption="$2"
    local video_path="$SAMPLE_DIR/${video_name}.mp4"
    local caption_path="$SAMPLE_DIR/${video_name}.txt"
    
    echo -e "${YELLOW}Creating sample video: ${video_name}.mp4${NC}"
    
    # Create a simple animated video using FFmpeg
    # This creates a 5-second video with animated colored background
    case "$video_name" in
        "gradient_flow")
            # Animated gradient flow
            ffmpeg -f lavfi -i "color=c=blue:size=720x480:duration=5" \
                   -f lavfi -i "color=c=red:size=720x480:duration=5" \
                   -filter_complex "[0][1]blend=all_mode=multiply:all_opacity=0.5,hue=H=2*PI*t:s=sin(t)+1" \
                   -t 5 -r 30 -pix_fmt yuv420p -y "$video_path" >/dev/null 2>&1
            ;;
        "noise_pattern")
            # Animated noise pattern
            ffmpeg -f lavfi -i "noise=c0s=64:c1s=0:c2s=0:c3s=0:size=720x480:duration=5" \
                   -filter_complex "hue=H=t*2*PI,eq=brightness=0.1:contrast=2" \
                   -t 5 -r 30 -pix_fmt yuv420p -y "$video_path" >/dev/null 2>&1
            ;;
        "rotating_pattern")
            # Rotating geometric pattern
            ffmpeg -f lavfi -i "color=c=black:size=720x480:duration=5" \
                   -vf "drawgrid=w=60:h=60:t=2:c=white@0.5,rotate=t*2*PI/5:c=black:ow=720:oh=480" \
                   -t 5 -r 30 -pix_fmt yuv420p -y "$video_path" >/dev/null 2>&1
            ;;
        "wave_motion")
            # Wave motion pattern
            ffmpeg -f lavfi -i "color=c=navy:size=720x480:duration=5" \
                   -vf "geq=r='255*sin(X/20+T*2)':g='255*sin(Y/20+T*2)':b='255*cos((X+Y)/20+T*2)'" \
                   -t 5 -r 30 -pix_fmt yuv420p -y "$video_path" >/dev/null 2>&1
            ;;
        "particle_system")
            # Particle-like movement
            ffmpeg -f lavfi -i "color=c=black:size=720x480:duration=5" \
                   -vf "noise=alls=20:allf=t+1,eq=brightness=-0.8" \
                   -t 5 -r 30 -pix_fmt yuv420p -y "$video_path" >/dev/null 2>&1
            ;;
        *)
            # Default: simple color fade
            ffmpeg -f lavfi -i "color=c=blue:size=720x480:duration=5" \
                   -vf "fade=t=in:st=0:d=2.5,fade=t=out:st=2.5:d=2.5" \
                   -t 5 -r 30 -pix_fmt yuv420p -y "$video_path" >/dev/null 2>&1
            ;;
    esac
    
    # Create caption file
    echo "$caption" > "$caption_path"
    echo -e "${GREEN}  ✓ Created: ${video_name}.mp4 and ${video_name}.txt${NC}"
}

# Check if FFmpeg is available
if ! command -v ffmpeg &> /dev/null; then
    echo -e "${RED}Error: FFmpeg is required to create sample videos${NC}"
    echo -e "${YELLOW}Install FFmpeg:${NC}"
    echo -e "${NC}  Ubuntu/Debian: sudo apt install ffmpeg${NC}"
    echo -e "${NC}  CentOS/RHEL: sudo yum install ffmpeg${NC}"
    echo -e "${NC}  Arch: sudo pacman -S ffmpeg${NC}"
    echo ""
    echo -e "${CYAN}Alternative: Place your own MP4 videos in $SAMPLE_DIR${NC}"
    echo -e "${CYAN}and create corresponding .txt caption files${NC}"
    exit 1
fi

# Create sample videos with captions
echo -e "${CYAN}Creating 5 sample training videos...${NC}"

create_sample_video "gradient_flow" "Smooth flowing gradient with blue and red colors, abstract animation, high quality"

create_sample_video "noise_pattern" "Dynamic noise pattern with shifting colors, textural movement, digital art style"

create_sample_video "rotating_pattern" "Geometric grid pattern rotating slowly, minimalist design, clean lines"

create_sample_video "wave_motion" "Colorful wave motion with mathematical patterns, smooth transitions, vibrant colors"

create_sample_video "particle_system" "Particle-like noise movement on dark background, atmospheric effect, subtle animation"

# Create additional example captions to show variety
echo -e "\n${CYAN}=== Creating Additional Caption Examples ===${NC}"

cat > "$SAMPLE_DIR/example_captions.txt" << 'EOF'
# Example Caption Formats for WAN 2.1 Training

# Good captions are:
# - Descriptive but concise
# - Include visual elements, style, and mood
# - Mention lighting, colors, and movement
# - Consistent in style across dataset

# Examples:

"A serene mountain landscape at sunrise, golden hour lighting, misty atmosphere"
"Urban cityscape at night, neon lights reflecting on wet streets, cinematic mood"
"Ocean waves crashing on rocky shore, dramatic lighting, slow motion effect"
"Abstract geometric shapes moving rhythmically, bright colors, modern art style"
"Forest path with dappled sunlight, peaceful atmosphere, natural lighting"
"Busy marketplace scene, vibrant colors, documentary style, natural movement"
"Minimalist interior with soft shadows, clean design, warm lighting"
"Fireworks exploding in night sky, colorful bursts, celebration atmosphere"
"Time-lapse of clouds moving across blue sky, natural motion, peaceful scene"
"Close-up of water droplets on glass, macro photography, soft focus background"

# Avoid:
# - Very short captions: "water"
# - Overly long descriptions
# - Inconsistent terminology
# - Missing visual details
EOF

# Create a dataset analysis script
echo -e "\n${CYAN}=== Creating Dataset Analysis Script ===${NC}"

cat > analyze_sample_dataset.sh << 'EOF'
#!/bin/bash

# Quick dataset analysis for the sample data
echo "=== Sample Dataset Analysis ==="
echo ""

VIDEOS_DIR="datasets/training_data/videos"

if [ ! -d "$VIDEOS_DIR" ]; then
    echo "Error: Videos directory not found at $VIDEOS_DIR"
    exit 1
fi

VIDEO_COUNT=$(find "$VIDEOS_DIR" -name "*.mp4" | wc -l)
CAPTION_COUNT=$(find "$VIDEOS_DIR" -name "*.txt" | wc -l)

echo "Video files: $VIDEO_COUNT"
echo "Caption files: $CAPTION_COUNT"
echo ""

if [ $VIDEO_COUNT -eq $CAPTION_COUNT ] && [ $VIDEO_COUNT -gt 0 ]; then
    echo "✓ Dataset is properly paired (each video has a caption)"
else
    echo "⚠ Warning: Mismatch between videos and captions"
fi

echo ""
echo "Video file details:"
for video in "$VIDEOS_DIR"/*.mp4; do
    if [ -f "$video" ]; then
        filename=$(basename "$video")
        size=$(du -h "$video" | cut -f1)
        echo "  $filename ($size)"
    fi
done

echo ""
echo "Total dataset size: $(du -sh "$VIDEOS_DIR" | cut -f1)"
echo ""
echo "To analyze in detail, run:"
echo "python dataset_helper.py $VIDEOS_DIR --mode analyze --output-report sample_dataset_report.json"
EOF

chmod +x analyze_sample_dataset.sh

# Run dataset analysis
echo -e "\n${CYAN}=== Running Dataset Analysis ===${NC}"
./analyze_sample_dataset.sh

echo -e "\n${GREEN}=== Sample Dataset Creation Complete! ===${NC}"
echo ""
echo -e "${YELLOW}What was created:${NC}"
echo -e "${NC}  📂 datasets/training_data/videos/ - Sample videos and captions${NC}"
echo -e "${NC}  📊 analyze_sample_dataset.sh - Quick dataset analysis script${NC}"
echo -e "${NC}  📝 example_captions.txt - Caption format examples${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo -e "${NC}1. Review the sample videos in datasets/training_data/videos/${NC}"
echo -e "${NC}2. Replace with your own training videos and captions${NC}"
echo -e "${NC}3. Run: python dataset_helper.py datasets/training_data/videos --mode analyze${NC}"
echo -e "${NC}4. Train with: ./train_wan21_lora.sh${NC}"
echo ""
echo -e "${CYAN}Note: These are synthetic sample videos for demonstration only.${NC}"
echo -e "${CYAN}For best results, use real video content relevant to your training goals.${NC}"
echo ""
echo -e "${GREEN}Ready to proceed with WAN 2.1 LoRA training! 🚀${NC}"
EOF

chmod +x create_sample_dataset.sh

# Update the main installation script to include the sample dataset creation
echo -e "\n${CYAN}=== Creating sample dataset creation helper ===${NC}"

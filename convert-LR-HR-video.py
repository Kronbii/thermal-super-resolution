import os
import sys
import time
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# Add model path
sys.path.append('model')
sys.path.append('.')
from model.architecture import IMDN

# Configuration constants
INPUT_FRAMES_DIR = "/home/kronbii/repos/thermal-super-resolution/datasets/video1"  # Directory containing video frames
SCALE = 3  # Super-resolution scale factor (2, 3, or 4)
MODEL_DIR = f"./checkpoints/_x{SCALE}/thermal_best.pth"
OUTPUT_DIR = "./results/video_processing"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FPS = 30.0  # Output video frame rate (adjust as needed)

def load_model(model_path, scale, device):
    """Load trained thermal model"""
    print(f"🔄 Loading model from {model_path}")
    
    # Create model
    model = IMDN(upscale=scale, in_nc=1, out_nc=1)  # Thermal model (1 channel)
    
    # Load checkpoint
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Extract state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Remove 'module.' prefix if present
        if any(key.startswith('module.') for key in state_dict.keys()):
            state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
        
        model.load_state_dict(state_dict, strict=True)
        print("✅ Model loaded successfully")
    else:
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    model = model.to(device)
    model.eval()
    return model

def preprocess_frame(frame):
    """Preprocess frame for model inference"""
    # Convert to grayscale if not already
    if len(frame.shape) == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Convert to float32 and normalize to [0, 1]
    frame = frame.astype(np.float32) / 255.0
    
    # Add batch and channel dimensions
    frame_tensor = torch.from_numpy(frame).unsqueeze(0).unsqueeze(0)
    return frame_tensor

def postprocess_frame(frame_tensor):
    """Convert tensor back to OpenCV format"""
    # Remove batch and channel dimensions, clamp values, and convert to uint8
    frame = torch.clamp(frame_tensor.squeeze(), 0, 1)
    frame_np = (frame.cpu().numpy() * 255).astype(np.uint8)
    return frame_np

def create_lr_frame(frame, scale):
    """Create low-resolution version of frame"""
    h, w = frame.shape[:2]
    lr_h, lr_w = h // scale, w // scale
    
    # Downsample using area interpolation for better quality
    lr_frame = cv2.resize(frame, (lr_w, lr_h), interpolation=cv2.INTER_AREA)
    return lr_frame

def create_side_by_side_frame(lr_frame, hr_frame):
    """Create side-by-side comparison frame"""
    # Resize LR frame to match HR frame height for comparison
    lr_height, lr_width = lr_frame.shape
    hr_height, hr_width = hr_frame.shape
    
    # Scale up LR frame using nearest neighbor to maintain pixelated look
    lr_upscaled = cv2.resize(lr_frame, (hr_width, hr_height), interpolation=cv2.INTER_NEAREST)
    
    # Create side-by-side frame
    comparison_frame = np.hstack((lr_upscaled, hr_frame))
    
    # Add text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2
    text_color = (255, 255, 255)  # White text
    
    # Add "LR" label on left side
    cv2.putText(comparison_frame, f"Original {lr_frame.shape}", (10, 30), font, font_scale, text_color, font_thickness)
    
    # Add "HR" label on right side
    cv2.putText(comparison_frame, f"SR {hr_frame.shape}", (hr_width + 10, 30), font, font_scale, text_color, font_thickness)
    
    return comparison_frame

def get_frame_files(frames_dir):
    """Get sorted list of frame files"""
    frame_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']:
        frame_files.extend(Path(frames_dir).glob(ext))
    
    # Sort files naturally (frame_001.png, frame_002.png, etc.)
    frame_files.sort(key=lambda x: x.name)
    return frame_files

def process_thermal_video():
    """Main video processing function"""
    print("🎬 Starting thermal video super-resolution processing...")
    print(f"📁 Input frames directory: {INPUT_FRAMES_DIR}")
    print(f"📊 Scale factor: {SCALE}x")
    print(f"🖥️  Device: {DEVICE}")
    
    # Check if input directory exists
    if not os.path.exists(INPUT_FRAMES_DIR):
        raise FileNotFoundError(f"Input frames directory not found: {INPUT_FRAMES_DIR}")
    
    # Get all frame files
    frame_files = get_frame_files(INPUT_FRAMES_DIR)
    if not frame_files:
        raise ValueError(f"No image files found in: {INPUT_FRAMES_DIR}")
    
    total_frames = len(frame_files)
    print(f"📹 Found {total_frames} frames")
    
    # Load model
    model = load_model(MODEL_DIR, SCALE, DEVICE)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{OUTPUT_DIR}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(str(frame_files[0]), cv2.IMREAD_GRAYSCALE)
    if first_frame is None:
        raise ValueError(f"Cannot read first frame: {frame_files[0]}")
    
    height, width = first_frame.shape
    print(f"📹 Frame info: {width}x{height}, {FPS} FPS, {total_frames} frames")
    
    # Calculate output dimensions
    lr_width, lr_height = width // SCALE, height // SCALE
    hr_width, hr_height = lr_width * SCALE, lr_height * SCALE
    
    print(f"🔽 LR output: {lr_width}x{lr_height}")
    print(f"🔼 HR output: {hr_width}x{hr_height}")
    print(f"🔄 Comparison output: {hr_width*2}x{hr_height}")
    
    # Define codec and create video writers
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Output paths
    lr_output_path = os.path.join(output_dir, f"thermal_LR_x{SCALE}.mp4")
    hr_output_path = os.path.join(output_dir, f"thermal_HR_x{SCALE}_AI.mp4")
    comparison_output_path = os.path.join(output_dir, f"thermal_comparison_x{SCALE}.mp4")
    
    # Create video writers
    lr_writer = cv2.VideoWriter(lr_output_path, fourcc, FPS, (lr_width, lr_height), False)
    hr_writer = cv2.VideoWriter(hr_output_path, fourcc, FPS, (hr_width, hr_height), False)
    comparison_writer = cv2.VideoWriter(comparison_output_path, fourcc, FPS, (hr_width*2, hr_height), False)
    
    if not lr_writer.isOpened() or not hr_writer.isOpened() or not comparison_writer.isOpened():
        raise ValueError("Failed to open video writers")
    
    print("🚀 Processing frames...")
    
    # Processing statistics
    total_inference_time = 0
    processed_frames = 0
    
    # Process each frame
    with tqdm(total=total_frames, desc="Processing frames") as pbar:
        for frame_file in frame_files:
            try:
                # Read frame
                frame = cv2.imread(str(frame_file), cv2.IMREAD_GRAYSCALE)
                if frame is None:
                    print(f"⚠️  Skipping unreadable frame: {frame_file}")
                    pbar.update(1)
                    continue
                
                # Create LR version of the frame
                lr_frame = create_lr_frame(frame, SCALE)
                
                # Preprocess LR frame for model
                lr_tensor = preprocess_frame(lr_frame).to(DEVICE)
                
                # Super-resolution inference
                start_time = time.time()
                with torch.no_grad():
                    hr_tensor = model(lr_tensor)
                inference_time = time.time() - start_time
                total_inference_time += inference_time
                
                # Postprocess HR frame
                hr_frame = postprocess_frame(hr_tensor)
                
                # Create side-by-side comparison frame
                comparison_frame = create_side_by_side_frame(lr_frame, hr_frame)
                
                # Write frames to output videos
                lr_writer.write(lr_frame)
                hr_writer.write(hr_frame)
                comparison_writer.write(comparison_frame)
                
                processed_frames += 1
                pbar.update(1)
                
            except Exception as e:
                print(f"❌ Error processing frame {frame_file}: {e}")
                pbar.update(1)
                continue
    
    # Clean up
    lr_writer.release()
    hr_writer.release()
    comparison_writer.release()
    
    # Print results
    avg_fps = processed_frames / total_inference_time if total_inference_time > 0 else 0
    
    print("\n" + "="*60)
    print("🎯 VIDEO PROCESSING COMPLETE")
    print("="*60)
    print(f"📊 Processed frames: {processed_frames}/{total_frames}")
    print(f"🚀 Average processing speed: {avg_fps:.1f} FPS")
    print(f"⏱️  Total inference time: {total_inference_time:.1f}s")
    print(f"\n💾 Output files:")
    print(f"   📹 LR Video: {lr_output_path}")
    print(f"   📹 HR Video (AI Enhanced): {hr_output_path}")
    print(f"   📹 Side-by-Side Comparison: {comparison_output_path}")
    print("="*60)

if __name__ == "__main__":
    try:
        process_thermal_video()
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
#!/usr/bin/env python3
"""
Thermal Super-Resolution Performance Test
=========================================

Tests model performance with real LR frames at specified FPS.
Evaluates super-resolution models with actual low-resolution thermal data.
"""

import torch
import time
import os
import cv2
import numpy as np
import glob
from pathlib import Path
from model.architecture import IMDN

# 🔧 CONFIGURATION
LR_FRAME_DIRECTORY = "datasets/flir_thermal_x2/val/LR_bicubic/X2"  # Directory containing LR frames
TARGET_FPS = 60  # Target FPS for evaluation (simulates video playback speed)
SCALE_FACTOR = 2  # Super-resolution scale (2x, 3x, or 4x)
LOG_FILE = "thermal_performance.log"  # Log file for results
MODEL_PATH = f"checkpoints/_x{SCALE_FACTOR}/thermal_best.pth"

# Alternative directories for different scales:
# LR_FRAME_DIRECTORY = "datasets/flir_thermal_x3/val/LR_bicubic"  # For 3x testing
# LR_FRAME_DIRECTORY = "datasets/flir_thermal_x4/val/LR_bicubic"  # For 4x testing
# 
# Or use your own LR frames:
# LR_FRAME_DIRECTORY = "/path/to/your/lr_frames"
# 
# Supported formats: .png, .jpg, .jpeg, .tiff, .bmp, .npy

def load_frame_from_file(filepath):
    """Load a thermal frame from file - optimized for speed"""
    if filepath.endswith('.npy'):
        # Load numpy array directly
        frame = np.load(filepath)
        if len(frame.shape) == 2:
            frame = np.expand_dims(frame, axis=2)
    else:
        # Load image using OpenCV with optimizations
        frame = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if frame is None:
            raise ValueError(f"Cannot load image: {filepath}")
        frame = np.expand_dims(frame, axis=2)
    
    # Normalize to 0-1 range (optimized)
    if frame.dtype == np.uint16:
        frame = (frame * (1.0 / 65535.0)).astype(np.float32)
    elif frame.dtype == np.uint8:
        frame = (frame * (1.0 / 255.0)).astype(np.float32)
    
    return frame

def preprocess_frame(frame, device):
    """Convert frame to tensor format - optimized for speed"""
    # Convert to tensor and move to device (blocking for accurate timing)
    tensor = torch.from_numpy(frame.transpose(2, 0, 1)).unsqueeze(0).to(device)
    return tensor

def log_message(message, log_file):
    """Write message to log file"""
    with open(log_file, 'a') as f:
        f.write(message + '\n')

def get_frame_files(directory):
    """Get all supported frame files from directory"""
    if not os.path.exists(directory):
        raise ValueError(f"Directory does not exist: {directory}")
    
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.tiff', '*.tif', '*.bmp', '*.npy']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(directory, ext)))
        files.extend(glob.glob(os.path.join(directory, ext.upper())))
    
    if not files:
        raise ValueError(f"No supported image files found in: {directory}")
    
    return sorted(files)

def evaluate_thermal_model():
    """Evaluate thermal super-resolution model with real LR frames"""
    # Clear previous log
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_message("Thermal Super-Resolution Performance Evaluation", LOG_FILE)
    log_message(f"Device: {device}", LOG_FILE)
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        log_message(f"GPU: {gpu_name}", LOG_FILE)
        log_message(f"CUDA Memory: {gpu_memory:.1f} GB", LOG_FILE)
        
        # Enable optimizations for faster processing
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    log_message("", LOG_FILE)
    log_message("Configuration:", LOG_FILE)
    log_message(f"Scale Factor: {SCALE_FACTOR}x", LOG_FILE)
    log_message(f"Target FPS: {TARGET_FPS}", LOG_FILE)
    log_message(f"LR Frame Directory: {LR_FRAME_DIRECTORY}", LOG_FILE)
    
    # Validate configuration
    if not os.path.exists(LR_FRAME_DIRECTORY):
        log_message(f"ERROR: LR frame directory not found: {LR_FRAME_DIRECTORY}", LOG_FILE)
        print(f"❌ ERROR: Check log file {LOG_FILE}")
        return
    
    if SCALE_FACTOR not in [2, 3, 4]:
        log_message(f"ERROR: Invalid scale factor: {SCALE_FACTOR}", LOG_FILE)
        print(f"❌ ERROR: Check log file {LOG_FILE}")
        return
    
    # Get LR frame files
    try:
        lr_frame_files = get_frame_files(LR_FRAME_DIRECTORY)
        log_message(f"Found {len(lr_frame_files)} LR frames", LOG_FILE)
    except Exception as e:
        log_message(f"ERROR loading LR frames: {e}", LOG_FILE)
        print(f"❌ ERROR: Check log file {LOG_FILE}")
        return
    
    # Load thermal model
    
    if os.path.exists(MODEL_PATH):
        log_message(f"Using trained thermal model: {MODEL_PATH}", LOG_FILE)
    else:
        log_message(f"ERROR: No model found for {SCALE_FACTOR}x super-resolution", LOG_FILE)
        print(f"❌ ERROR: Check log file {LOG_FILE}")
        return
    
    # Initialize model
    try:
        model = IMDN(upscale=SCALE_FACTOR, in_nc=1, out_nc=1).to(device)
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        log_message("Model loaded successfully", LOG_FILE)
    except Exception as e:
        log_message(f"ERROR loading model: {e}", LOG_FILE)
        print(f"❌ ERROR: Check log file {LOG_FILE}")
        return
    
    log_message("", LOG_FILE)
    log_message("Running Performance Evaluation...", LOG_FILE)
    
    # Run evaluation with detailed timing
    evaluation_success = evaluate_with_detailed_timing(model, device, lr_frame_files, SCALE_FACTOR, TARGET_FPS)
    
    if evaluation_success:
        log_message("", LOG_FILE)
        log_message("Evaluation completed successfully!", LOG_FILE)
        log_message(f"Summary: {SCALE_FACTOR}x super-resolution at {TARGET_FPS} FPS target", LOG_FILE)
        print(f"✅ Performance evaluation completed!")
        print(f"📊 Results saved to: {LOG_FILE}")
    else:
        log_message("Evaluation failed!", LOG_FILE)
        print(f"❌ Evaluation failed! Check log file {LOG_FILE}")

def evaluate_with_detailed_timing(model, device, lr_frame_files, scale, target_fps):
    """Evaluate model performance with detailed timing breakdown"""
    
    expected_frame_time = 1000.0 / target_fps  # ms per frame
    total_frames = min(100, len(lr_frame_files))  # Limit for testing
    
    log_message(f"Testing {total_frames} frames at {target_fps} FPS", LOG_FILE)
    log_message(f"Expected frame interval: {expected_frame_time:.1f}ms", LOG_FILE)
    log_message("", LOG_FILE)
    
    # Performance tracking arrays
    loading_times = []
    preprocessing_times = []
    inference_times = []
    total_times = []
    successful_frames = 0
    
    # Load a sample frame to get correct dimensions for warmup
    sample_frame = load_frame_from_file(lr_frame_files[0])
    h, w = sample_frame.shape[:2]
    
    # Warmup GPU with correct input dimensions
    dummy_input = torch.randn(1, 1, h, w).to(device)
    with torch.no_grad():
        for _ in range(10):  # More warmup iterations
            _ = model(dummy_input)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    torch.cuda.empty_cache()  # Clear cache after warmup
    
    # Process frames with detailed timing
    with torch.no_grad():
        for i, frame_path in enumerate(lr_frame_files[:total_frames]):
            try:
                overall_start = time.perf_counter()
                
                # Time loading
                load_start = time.perf_counter()
                lr_frame = load_frame_from_file(frame_path)
                load_end = time.perf_counter()
                loading_time = (load_end - load_start) * 1000
                
                # Time preprocessing
                preprocess_start = time.perf_counter()
                lr_tensor = preprocess_frame(lr_frame, device)
                preprocess_end = time.perf_counter()
                preprocessing_time = (preprocess_end - preprocess_start) * 1000
                
                # Time inference (minimize synchronization overhead)
                inference_start = time.perf_counter()
                sr_output = model(lr_tensor)
                inference_end = time.perf_counter()
                inference_time = (inference_end - inference_start) * 1000
                
                overall_end = time.perf_counter()
                total_time = (overall_end - overall_start) * 1000
                
                # Record times
                loading_times.append(loading_time)
                preprocessing_times.append(preprocessing_time)
                inference_times.append(inference_time)
                total_times.append(total_time)
                successful_frames += 1
                
                # Get dimensions
                h, w = lr_frame.shape[:2]
                sr_h, sr_w = sr_output.shape[2], sr_output.shape[3]
                filename = os.path.basename(frame_path)[:35]  # Truncate for readability
                
                # Log detailed timing for every frame (first 10, every 20th, last 10)
                if i < 10 or (i + 1) % 20 == 0 or i >= total_frames - 10:
                    log_message(f"Frame {i+1:3d} | {w}x{h} → {sr_w}x{sr_h} | "
                               f"Load: {loading_time:.2f}ms | Prep: {preprocessing_time:.2f}ms | "
                               f"Inference: {inference_time:.2f}ms | Total: {total_time:.2f}ms", LOG_FILE)
                
            except Exception as e:
                log_message(f"ERROR Frame {i+1}: {os.path.basename(frame_path)} - {e}", LOG_FILE)
                continue
    
    # Calculate statistics
    if successful_frames > 0:
        avg_loading = np.mean(loading_times)
        avg_preprocessing = np.mean(preprocessing_times)
        avg_inference = np.mean(inference_times)
        avg_total = np.mean(total_times)
        
        inference_fps = 1000.0 / avg_inference
        total_fps = 1000.0 / avg_total
        
        # Honest assessment of target achievement
        can_meet_target = avg_total <= expected_frame_time
        fps_status = "CAN" if can_meet_target else "CANNOT"
        
        log_message("", LOG_FILE)
        log_message("Performance Results:", LOG_FILE)
        log_message(f"  Frames Processed: {successful_frames}/{total_frames}", LOG_FILE)
        log_message(f"  Average Loading Time: {avg_loading:.2f}ms", LOG_FILE)
        log_message(f"  Average Preprocessing Time: {avg_preprocessing:.2f}ms", LOG_FILE)
        log_message(f"  Average Inference Time: {avg_inference:.2f}ms ({inference_fps:.1f} FPS)", LOG_FILE)
        log_message(f"  Average Total Processing Time: {avg_total:.2f}ms ({total_fps:.1f} FPS)", LOG_FILE)
        log_message(f"  Target FPS: {target_fps} ({expected_frame_time:.1f}ms per frame)", LOG_FILE)
        log_message(f"  Target Achievement: {fps_status} meet {target_fps} FPS target", LOG_FILE)
        
        if device.type == 'cuda':
            peak_memory = torch.cuda.max_memory_allocated() / (1024**2)
            log_message(f"  Peak GPU Memory: {peak_memory:.1f} MB", LOG_FILE)
        
        # Print summary to terminal
        print(f"🎯 Target: {target_fps} FPS | Achieved: {total_fps:.1f} FPS | Status: {fps_status}")
        
        return True
    
    return False

if __name__ == "__main__":
    evaluate_thermal_model()
#!/usr/bin/env python3
"""
Simple Demo Launcher with Error Handling
========================================

A more robust launcher that handles common PyQt5/OpenCV issues.
"""

import sys
import os
import subprocess
from pathlib import Path

def check_and_fix_environment():
    """Check and fix common environment issues"""
    print("🔍 Checking environment...")
    
    # Check display
    if 'DISPLAY' not in os.environ:
        print("❌ No DISPLAY variable found")
        if 'SSH_CLIENT' in os.environ or 'SSH_TTY' in os.environ:
            print("💡 You're using SSH. Try: ssh -X username@hostname")
        return False
    
    # Set Qt platform
    os.environ['QT_QPA_PLATFORM'] = 'xcb'
    
    # Check if we can import PyQt5
    try:
        from PyQt5.QtWidgets import QApplication
        print("✅ PyQt5 available")
    except ImportError:
        print("❌ PyQt5 not available")
        print("💡 Install with: pip install PyQt5")
        return False
    
    # Check OpenCV
    try:
        import cv2
        # Check for Qt backend conflicts
        if hasattr(cv2, 'qt'):
            print("⚠️ OpenCV has Qt backend - this may cause conflicts")
            print("💡 Consider using: pip uninstall opencv-python && pip install opencv-python-headless")
        else:
            print("✅ OpenCV available (headless)")
    except ImportError:
        print("❌ OpenCV not available")
        return False
    
    return True

def run_simple_demo():
    """Run a simpler demo without PyQt5 if needed"""
    print("🚀 Starting simple terminal demo...")
    
    try:
        import torch
        import numpy as np
        import cv2
        from model.architecture import IMDN
        
        print("✅ All dependencies available")
        print("📊 Running basic inference test...")
        
        # Load model
        model_path = "checkpoints/_x2/thermal_best.pth"
        if not os.path.exists(model_path):
            print(f"⚠️ Model not found: {model_path}")
            print("Creating dummy test instead...")
            
            # Create dummy test
            model = IMDN(upscale=2, in_nc=1, out_nc=1)
            input_tensor = torch.randn(1, 1, 64, 64)
            
            start_time = time.time()
            with torch.no_grad():
                output = model(input_tensor)
            inference_time = time.time() - start_time
            
            print(f"✅ Inference test successful:")
            print(f"   Input shape: {input_tensor.shape}")
            print(f"   Output shape: {output.shape}")
            print(f"   Inference time: {inference_time*1000:.2f} ms")
            print(f"   Estimated FPS: {1/inference_time:.1f}")
            
        else:
            # Load real model and test
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = IMDN(upscale=2, in_nc=1, out_nc=1).to(device)
            
            checkpoint = torch.load(model_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.eval()
            print(f"✅ Model loaded: {model_path}")
            
            # Test inference
            test_input = torch.randn(1, 1, 160, 120).to(device)
            
            import time
            start_time = time.time()
            with torch.no_grad():
                output = model(test_input)
            inference_time = time.time() - start_time
            
            print(f"📊 Performance Test Results:")
            print(f"   Device: {device}")
            print(f"   Input: {test_input.shape} -> Output: {output.shape}")
            print(f"   Inference time: {inference_time*1000:.2f} ms")
            print(f"   Estimated FPS: {1/inference_time:.1f}")
            
            if torch.cuda.is_available():
                print(f"   GPU memory: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Simple demo failed: {e}")
        return False

def main():
    """Main launcher with fallback options"""
    print("=" * 60)
    print("🔥 Thermal Super-Resolution Demo Launcher")
    print("=" * 60)
    
    # Try to fix environment
    if not check_and_fix_environment():
        print("\n💡 GUI demo not available. Trying simple terminal demo...")
        if run_simple_demo():
            print("\n✅ Simple demo completed successfully!")
        else:
            print("\n❌ All demos failed. Please check your environment.")
        return
    
    # Try to run full GUI demo
    try:
        print("🚀 Starting GUI demo application...")
        from demo_app import main as demo_main
        demo_main()
    except Exception as e:
        print(f"❌ GUI demo failed: {e}")
        print("\n💡 Falling back to simple terminal demo...")
        if run_simple_demo():
            print("\n✅ Simple demo completed!")
        else:
            print("\n❌ All demos failed.")

if __name__ == "__main__":
    main()
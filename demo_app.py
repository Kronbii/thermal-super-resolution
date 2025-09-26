#!/usr/bin/env python3
"""
Thermal Super-Resolution Demo Application
========================================

Professional demo system for showcasing thermal super-resolution performance.
Features side-by-side video comparison and real-time FPS benchmarking.

Author: Kronbii
Date: 2025
"""

import sys
import os
import time
import threading
from pathlib import Path
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
import json

# Fix environment issues before importing Qt
os.environ['QT_QPA_PLATFORM'] = 'xcb'
if 'DISPLAY' not in os.environ:
    os.environ['DISPLAY'] = ':0'

# Import Qt before OpenCV to avoid conflicts
try:
    from PyQt5.QtWidgets import *
    from PyQt5.QtCore import *
    from PyQt5.QtGui import *
    print("✅ PyQt5 imported successfully")
except ImportError as e:
    print(f"❌ PyQt5 import failed: {e}")
    print("💡 Run: python3 fix_demo_environment.py")
    sys.exit(1)

# Import OpenCV after Qt
try:
    import cv2
    print("✅ OpenCV imported successfully")
except ImportError as e:
    print(f"❌ OpenCV import failed: {e}")
    print("💡 Run: pip install opencv-python-headless")
    sys.exit(1)

import numpy as np
import torch
import torch.nn.functional as F

# Import your model
try:
    from model.architecture import IMDN
    print("✅ IMDN model imported successfully")
except ImportError:
    print("⚠️ Could not import IMDN model. Using dummy model for demo.")
    
    class IMDN(torch.nn.Module):
        """Dummy IMDN model for demo purposes"""
        def __init__(self, upscale=2, in_nc=1, out_nc=1):
            super().__init__()
            self.upscale = upscale
            self.conv = torch.nn.Conv2d(in_nc, out_nc * (upscale ** 2), 3, padding=1)
            self.pixel_shuffle = torch.nn.PixelShuffle(upscale)
            
        def forward(self, x):
            return self.pixel_shuffle(self.conv(x))

@dataclass
class PerformanceMetrics:
    """Real-time performance metrics tracking"""
    inference_time: float = 0.0
    preprocessing_time: float = 0.0
    postprocessing_time: float = 0.0
    total_time: float = 0.0
    current_fps: float = 0.0
    average_fps: float = 0.0
    target_fps: float = 0.0
    frames_processed: int = 0
    dropped_frames: int = 0
    gpu_memory_used: float = 0.0
    efficiency_score: float = 0.0

class ThermalSRModel:
    """Optimized model wrapper for real-time inference"""
    
    def __init__(self, model_path: str, scale: int, device: str = 'auto'):
        self.scale = scale
        self.device = torch.device('cuda' if torch.cuda.is_available() and device != 'cpu' else 'cpu')
        self.model = self._load_model(model_path)
        self._optimize_model()
        
    def _load_model(self, model_path: str) -> torch.nn.Module:
        """Load and initialize the thermal super-resolution model"""
        model = IMDN(upscale=self.scale, in_nc=1, out_nc=1)
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"✅ Model loaded successfully: {model_path}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
            
        return model.to(self.device).eval()
    
    def _optimize_model(self):
        """Apply optimization techniques for maximum performance"""
        try:
            # Enable optimized execution
            torch.backends.cudnn.benchmark = True
            if hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
                torch.backends.cuda.matmul.allow_tf32 = True
            if hasattr(torch.backends.cudnn, 'allow_tf32'):
                torch.backends.cudnn.allow_tf32 = True
            
            # Use channels_last memory format for better performance
            if self.device.type == 'cuda':
                self.model = self.model.to(memory_format=torch.channels_last)
            
            # Optional: Compile model for extra speed (PyTorch 2.0+)
            if hasattr(torch, 'compile'):
                try:
                    self.model = torch.compile(self.model, mode='default')
                    print("🚀 Model compiled for better performance")
                except Exception as e:
                    print(f"⚠️ Model compilation failed: {e}")
        except Exception as e:
            print(f"⚠️ Optimization failed: {e}")
            # Continue without optimizations
    
    def preprocess(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess thermal frame for model inference"""
        # Ensure single channel
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Normalize to [0, 1]
        frame = frame.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        tensor = torch.from_numpy(frame).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Use channels_last for better performance
        if self.device.type == 'cuda':
            tensor = tensor.to(memory_format=torch.channels_last)
            
        return tensor
    
    def postprocess(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert model output back to displayable image"""
        # Move to CPU and convert to numpy
        output = tensor.squeeze().cpu().numpy()
        
        # Clip and convert to uint8
        output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
        
        return output
    
    def inference(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """Perform complete inference with timing"""
        times = {}
        
        # Preprocessing
        start_time = time.perf_counter()
        input_tensor = self.preprocess(frame)
        times['preprocessing'] = time.perf_counter() - start_time
        
        # Model inference
        start_time = time.perf_counter()
        with torch.no_grad():
            if self.device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    output_tensor = self.model(input_tensor)
            else:
                output_tensor = self.model(input_tensor)
        
        # Synchronize CUDA if needed
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        times['inference'] = time.perf_counter() - start_time
        
        # Postprocessing
        start_time = time.perf_counter()
        enhanced_frame = self.postprocess(output_tensor)
        times['postprocessing'] = time.perf_counter() - start_time
        
        times['total'] = sum(times.values())
        
        return enhanced_frame, times

class VideoComparisonWidget(QWidget):
    """Side-by-side video comparison widget"""
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.video_path = None
        self.current_frame = 0
        self.total_frames = 0
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.fps = 30
        
        self.setup_ui()
        self.apply_styles()
    
    def setup_ui(self):
        """Setup the elegant UI layout"""
        layout = QVBoxLayout(self)
        
        # Header
        header = QLabel("Video Comparison Demo")
        header.setAlignment(Qt.AlignCenter)
        header.setObjectName("header")
        layout.addWidget(header)
        
        # Controls
        controls = QHBoxLayout()
        
        # Video selection
        self.select_video_btn = QPushButton("Select Video")
        self.select_video_btn.clicked.connect(self.select_video)
        controls.addWidget(self.select_video_btn)
        
        # Scale selection
        scale_label = QLabel("Scale:")
        controls.addWidget(scale_label)
        
        self.scale_combo = QComboBox()
        self.scale_combo.addItems(["2x", "3x", "4x"])
        self.scale_combo.currentTextChanged.connect(self.scale_changed)
        controls.addWidget(self.scale_combo)
        
        # Play/Pause
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.toggle_playback)
        self.play_btn.setEnabled(False)
        controls.addWidget(self.play_btn)
        
        # FPS control
        fps_label = QLabel("FPS:")
        controls.addWidget(fps_label)
        
        self.fps_slider = QSlider(Qt.Horizontal)
        self.fps_slider.setRange(1, 60)
        self.fps_slider.setValue(30)
        self.fps_slider.valueChanged.connect(self.fps_changed)
        controls.addWidget(self.fps_slider)
        
        self.fps_display = QLabel("30")
        controls.addWidget(self.fps_display)
        
        controls.addStretch()
        layout.addLayout(controls)
        
        # Video displays
        video_layout = QHBoxLayout()
        
        # Original video
        original_group = QGroupBox("Original (Low Resolution)")
        original_layout = QVBoxLayout(original_group)
        self.original_label = QLabel()
        self.original_label.setFixedSize(400, 300)
        self.original_label.setStyleSheet("border: 2px solid #333; background: black;")
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setText("No video loaded")
        original_layout.addWidget(self.original_label)
        video_layout.addWidget(original_group)
        
        # Enhanced video
        enhanced_group = QGroupBox("AI Enhanced (High Resolution)")
        enhanced_layout = QVBoxLayout(enhanced_group)
        self.enhanced_label = QLabel()
        self.enhanced_label.setFixedSize(400, 300)
        self.enhanced_label.setStyleSheet("border: 2px solid #333; background: black;")
        self.enhanced_label.setAlignment(Qt.AlignCenter)
        self.enhanced_label.setText("No video loaded")
        enhanced_layout.addWidget(self.enhanced_label)
        video_layout.addWidget(enhanced_group)
        
        layout.addLayout(video_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Status
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
    
    def apply_styles(self):
        """Apply elegant styling"""
        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e1e;
                color: #ffffff;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            
            QLabel#header {
                font-size: 24px;
                font-weight: bold;
                color: #00d4aa;
                margin: 20px 0;
            }
            
            QPushButton {
                background-color: #0084ff;
                border: none;
                color: white;
                padding: 8px 16px;
                border-radius: 6px;
                font-weight: 500;
            }
            
            QPushButton:hover {
                background-color: #0066cc;
            }
            
            QPushButton:pressed {
                background-color: #004499;
            }
            
            QPushButton:disabled {
                background-color: #333;
                color: #666;
            }
            
            QGroupBox {
                font-weight: bold;
                border: 2px solid #333;
                border-radius: 8px;
                margin: 10px 0;
                padding: 10px;
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            
            QSlider::groove:horizontal {
                border: 1px solid #333;
                height: 8px;
                background: #444;
                border-radius: 4px;
            }
            
            QSlider::handle:horizontal {
                background: #00d4aa;
                border: 1px solid #00d4aa;
                width: 18px;
                border-radius: 9px;
                margin: -5px 0;
            }
            
            QProgressBar {
                border: 1px solid #333;
                border-radius: 4px;
                background: #444;
            }
            
            QProgressBar::chunk {
                background: #00d4aa;
                border-radius: 4px;
            }
            
            QComboBox {
                border: 1px solid #333;
                border-radius: 4px;
                padding: 4px;
                background: #444;
            }
            
            QComboBox::drop-down {
                border: none;
            }
            
            QComboBox::down-arrow {
                image: none;
                border-style: solid;
                border-width: 3px;
                border-color: transparent transparent #00d4aa transparent;
            }
        """)
    
    def select_video(self):
        """Select video file for comparison"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "", 
            "Video Files (*.mp4 *.avi *.mov *.mkv *.flv);;All Files (*)"
        )
        
        if file_path:
            self.load_video(file_path)
    
    def load_video(self, video_path: str):
        """Load video file and initialize capture"""
        try:
            self.cap = cv2.VideoCapture(video_path)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            self.video_path = video_path
            self.current_frame = 0
            self.progress_bar.setMaximum(self.total_frames)
            self.progress_bar.setVisible(True)
            self.play_btn.setEnabled(True)
            
            self.status_label.setText(f"Video loaded: {Path(video_path).name} ({self.total_frames} frames)")
            
            # Load model if not already loaded
            self.load_model()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load video: {str(e)}")
    
    def load_model(self):
        """Load the appropriate model for the selected scale"""
        scale_map = {"2x": 2, "3x": 3, "4x": 4}
        scale = scale_map[self.scale_combo.currentText()]
        
        model_path = f"checkpoints/_x{scale}/thermal_best.pth"
        if not os.path.exists(model_path):
            model_path = f"checkpoints/thermal_epoch_best.pth"
        
        if not os.path.exists(model_path):
            QMessageBox.warning(self, "Warning", f"Model not found: {model_path}")
            return
        
        try:
            self.model = ThermalSRModel(model_path, scale)
            self.status_label.setText(f"Model loaded: {scale}x enhancement ready")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
    
    def scale_changed(self):
        """Handle scale selection change"""
        if self.video_path:
            self.load_model()
    
    def fps_changed(self, value):
        """Handle FPS slider change"""
        self.fps = value
        self.fps_display.setText(str(value))
        if self.timer.isActive():
            self.timer.setInterval(1000 // self.fps)
    
    def toggle_playback(self):
        """Toggle video playback"""
        if self.timer.isActive():
            self.timer.stop()
            self.play_btn.setText("Play")
        else:
            self.timer.start(1000 // self.fps)
            self.play_btn.setText("Pause")
    
    def update_frame(self):
        """Update video frames"""
        if not self.cap or not self.model:
            return
        
        ret, frame = self.cap.read()
        if not ret:
            # Loop video
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.current_frame = 0
            ret, frame = self.cap.read()
        
        if ret:
            # Display original frame
            self.display_frame(frame, self.original_label, "Original")
            
            # Enhance frame
            try:
                enhanced_frame, _ = self.model.inference(frame)
                self.display_frame(enhanced_frame, self.enhanced_label, "Enhanced")
            except Exception as e:
                print(f"Enhancement error: {e}")
            
            self.current_frame += 1
            self.progress_bar.setValue(self.current_frame)
    
    def display_frame(self, frame: np.ndarray, label: QLabel, title: str):
        """Display frame in label widget"""
        if len(frame.shape) == 2:
            # Grayscale
            height, width = frame.shape
            bytes_per_line = width
            q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        else:
            # Color (convert BGR to RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = frame_rgb.shape
            bytes_per_line = 3 * width
            q_image = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        # Scale to fit label
        pixmap = QPixmap.fromImage(q_image).scaled(
            label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        label.setPixmap(pixmap)

class PerformanceBenchmarkWidget(QWidget):
    """Real-time FPS benchmarking widget"""
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.frames_dir = None
        self.frame_files = []
        self.current_frame_idx = 0
        self.benchmark_thread = None
        self.is_running = False
        self.metrics = PerformanceMetrics()
        
        self.setup_ui()
        self.apply_styles()
    
    def setup_ui(self):
        """Setup benchmark UI"""
        layout = QVBoxLayout(self)
        
        # Header
        header = QLabel("Real-time Performance Benchmark")
        header.setAlignment(Qt.AlignCenter)
        header.setObjectName("header")
        layout.addWidget(header)
        
        # Controls
        controls = QHBoxLayout()
        
        # Directory selection
        self.select_dir_btn = QPushButton("Select Frames Directory")
        self.select_dir_btn.clicked.connect(self.select_frames_directory)
        controls.addWidget(self.select_dir_btn)
        
        # Scale selection
        scale_label = QLabel("Scale:")
        controls.addWidget(scale_label)
        
        self.scale_combo = QComboBox()
        self.scale_combo.addItems(["2x", "3x", "4x"])
        self.scale_combo.currentTextChanged.connect(self.scale_changed)
        controls.addWidget(self.scale_combo)
        
        # Target FPS
        fps_label = QLabel("Target FPS:")
        controls.addWidget(fps_label)
        
        self.fps_spinbox = QSpinBox()
        self.fps_spinbox.setRange(1, 120)
        self.fps_spinbox.setValue(30)
        controls.addWidget(self.fps_spinbox)
        
        # Start/Stop
        self.benchmark_btn = QPushButton("Start Benchmark")
        self.benchmark_btn.clicked.connect(self.toggle_benchmark)
        self.benchmark_btn.setEnabled(False)
        controls.addWidget(self.benchmark_btn)
        
        controls.addStretch()
        layout.addLayout(controls)
        
        # Metrics display
        metrics_layout = QGridLayout()
        
        # Current frame display
        frame_group = QGroupBox("Current Frame")
        frame_layout = QVBoxLayout(frame_group)
        self.frame_label = QLabel()
        self.frame_label.setFixedSize(300, 225)
        self.frame_label.setStyleSheet("border: 2px solid #333; background: black;")
        self.frame_label.setAlignment(Qt.AlignCenter)
        self.frame_label.setText("No frames loaded")
        frame_layout.addWidget(self.frame_label)
        metrics_layout.addWidget(frame_group, 0, 0, 2, 1)
        
        # Performance metrics
        perf_group = QGroupBox("Performance Metrics")
        perf_layout = QGridLayout(perf_group)
        
        # Metric labels
        self.metric_labels = {}
        metrics = [
            ("Target FPS", "target_fps"),
            ("Current FPS", "current_fps"),
            ("Average FPS", "average_fps"),
            ("Inference Time", "inference_time"),
            ("Total Time", "total_time"),
            ("Frames Processed", "frames_processed"),
            ("Dropped Frames", "dropped_frames"),
            ("GPU Memory", "gpu_memory_used"),
            ("Efficiency Score", "efficiency_score")
        ]
        
        for i, (name, key) in enumerate(metrics):
            label = QLabel(f"{name}:")
            value_label = QLabel("0")
            value_label.setObjectName("metric_value")
            
            perf_layout.addWidget(label, i, 0)
            perf_layout.addWidget(value_label, i, 1)
            self.metric_labels[key] = value_label
        
        metrics_layout.addWidget(perf_group, 0, 1)
        
        # Real-time graph placeholder
        graph_group = QGroupBox("Performance Graph")
        graph_layout = QVBoxLayout(graph_group)
        self.graph_label = QLabel("Performance visualization would go here")
        self.graph_label.setFixedSize(300, 150)
        self.graph_label.setStyleSheet("border: 2px solid #333; background: #2a2a2a;")
        self.graph_label.setAlignment(Qt.AlignCenter)
        graph_layout.addWidget(self.graph_label)
        metrics_layout.addWidget(graph_group, 1, 1)
        
        layout.addLayout(metrics_layout)
        
        # Status
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        # Update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_metrics_display)
        self.update_timer.start(100)  # Update UI every 100ms
    
    def apply_styles(self):
        """Apply elegant styling to benchmark widget"""
        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e1e;
                color: #ffffff;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            
            QLabel#header {
                font-size: 24px;
                font-weight: bold;
                color: #ff6b35;
                margin: 20px 0;
            }
            
            QLabel#metric_value {
                font-weight: bold;
                color: #00d4aa;
                font-size: 14px;
            }
            
            QPushButton {
                background-color: #ff6b35;
                border: none;
                color: white;
                padding: 8px 16px;
                border-radius: 6px;
                font-weight: 500;
            }
            
            QPushButton:hover {
                background-color: #e55a2b;
            }
            
            QPushButton:pressed {
                background-color: #cc4a21;
            }
            
            QPushButton:disabled {
                background-color: #333;
                color: #666;
            }
            
            QGroupBox {
                font-weight: bold;
                border: 2px solid #333;
                border-radius: 8px;
                margin: 10px 0;
                padding: 10px;
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            
            QSpinBox {
                border: 1px solid #333;
                border-radius: 4px;
                padding: 4px;
                background: #444;
            }
            
            QComboBox {
                border: 1px solid #333;
                border-radius: 4px;
                padding: 4px;
                background: #444;
            }
        """)
    
    def select_frames_directory(self):
        """Select directory containing test frames"""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Frames Directory")
        
        if dir_path:
            self.load_frames(dir_path)
    
    def load_frames(self, frames_dir: str):
        """Load frames from directory"""
        self.frames_dir = Path(frames_dir)
        
        # Find image files
        extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        self.frame_files = []
        for ext in extensions:
            self.frame_files.extend(self.frames_dir.glob(f'*{ext}'))
            self.frame_files.extend(self.frames_dir.glob(f'*{ext.upper()}'))
        
        self.frame_files.sort()
        
        if self.frame_files:
            self.status_label.setText(f"Loaded {len(self.frame_files)} frames from {self.frames_dir.name}")
            self.benchmark_btn.setEnabled(True)
            self.load_model()
        else:
            QMessageBox.warning(self, "Warning", "No image files found in selected directory")
    
    def load_model(self):
        """Load model for benchmarking"""
        scale_map = {"2x": 2, "3x": 3, "4x": 4}
        scale = scale_map[self.scale_combo.currentText()]
        
        model_path = f"checkpoints/_x{scale}/thermal_best.pth"
        if not os.path.exists(model_path):
            model_path = f"checkpoints/thermal_epoch_best.pth"
        
        if not os.path.exists(model_path):
            QMessageBox.warning(self, "Warning", f"Model not found: {model_path}")
            return
        
        try:
            self.model = ThermalSRModel(model_path, scale)
            self.status_label.setText(f"Model loaded: {scale}x enhancement ready")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
    
    def scale_changed(self):
        """Handle scale selection change"""
        if self.frames_dir:
            self.load_model()
    
    def toggle_benchmark(self):
        """Start/stop benchmark"""
        if not self.is_running:
            self.start_benchmark()
        else:
            self.stop_benchmark()
    
    def start_benchmark(self):
        """Start the benchmarking process"""
        if not self.model or not self.frame_files:
            return
        
        self.is_running = True
        self.benchmark_btn.setText("Stop Benchmark")
        self.metrics = PerformanceMetrics()
        self.metrics.target_fps = self.fps_spinbox.value()
        
        # Start benchmark thread
        self.benchmark_thread = BenchmarkThread(
            self.model, self.frame_files, self.metrics.target_fps
        )
        self.benchmark_thread.metrics_updated.connect(self.update_metrics)
        self.benchmark_thread.frame_processed.connect(self.display_current_frame)
        self.benchmark_thread.finished.connect(self.benchmark_finished)
        self.benchmark_thread.start()
    
    def stop_benchmark(self):
        """Stop the benchmarking process"""
        self.is_running = False
        if self.benchmark_thread:
            self.benchmark_thread.stop()
        self.benchmark_btn.setText("Start Benchmark")
        self.status_label.setText("Benchmark stopped")
    
    def benchmark_finished(self):
        """Handle benchmark completion"""
        self.is_running = False
        self.benchmark_btn.setText("Start Benchmark")
        self.status_label.setText("Benchmark completed")
    
    @pyqtSlot(object)
    def update_metrics(self, metrics: PerformanceMetrics):
        """Update metrics from benchmark thread"""
        self.metrics = metrics
    
    @pyqtSlot(np.ndarray)
    def display_current_frame(self, frame: np.ndarray):
        """Display current frame being processed"""
        if len(frame.shape) == 2:
            # Grayscale
            height, width = frame.shape
            bytes_per_line = width
            q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        else:
            # Color
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = frame_rgb.shape
            bytes_per_line = 3 * width
            q_image = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        pixmap = QPixmap.fromImage(q_image).scaled(
            self.frame_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.frame_label.setPixmap(pixmap)
    
    def update_metrics_display(self):
        """Update metrics display in UI"""
        self.metric_labels["target_fps"].setText(f"{self.metrics.target_fps:.1f}")
        self.metric_labels["current_fps"].setText(f"{self.metrics.current_fps:.1f}")
        self.metric_labels["average_fps"].setText(f"{self.metrics.average_fps:.1f}")
        self.metric_labels["inference_time"].setText(f"{self.metrics.inference_time*1000:.1f} ms")
        self.metric_labels["total_time"].setText(f"{self.metrics.total_time*1000:.1f} ms")
        self.metric_labels["frames_processed"].setText(f"{self.metrics.frames_processed}")
        self.metric_labels["dropped_frames"].setText(f"{self.metrics.dropped_frames}")
        self.metric_labels["gpu_memory_used"].setText(f"{self.metrics.gpu_memory_used:.1f} MB")
        self.metric_labels["efficiency_score"].setText(f"{self.metrics.efficiency_score:.1f}%")

class BenchmarkThread(QThread):
    """Background thread for running performance benchmark"""
    
    metrics_updated = pyqtSignal(object)  # PerformanceMetrics
    frame_processed = pyqtSignal(np.ndarray)  # Current frame
    
    def __init__(self, model: ThermalSRModel, frame_files: List[Path], target_fps: float):
        super().__init__()
        self.model = model
        self.frame_files = frame_files
        self.target_fps = target_fps
        self.running = True
        self.metrics = PerformanceMetrics()
        self.metrics.target_fps = target_fps
    
    def stop(self):
        """Stop the benchmark"""
        self.running = False
        self.wait()
    
    def run(self):
        """Run the benchmark"""
        frame_interval = 1.0 / self.target_fps
        start_time = time.perf_counter()
        fps_calc_start = start_time
        fps_frame_count = 0
        
        for i, frame_file in enumerate(self.frame_files):
            if not self.running:
                break
            
            loop_start = time.perf_counter()
            
            try:
                # Load frame
                frame = cv2.imread(str(frame_file), cv2.IMREAD_GRAYSCALE)
                if frame is None:
                    continue
                
                # Process frame
                enhanced_frame, times = self.model.inference(frame)
                
                # Update metrics
                self.metrics.inference_time = times['inference']
                self.metrics.preprocessing_time = times['preprocessing']
                self.metrics.postprocessing_time = times['postprocessing']
                self.metrics.total_time = times['total']
                self.metrics.frames_processed += 1
                
                # Calculate FPS
                current_time = time.perf_counter()
                self.metrics.current_fps = 1.0 / (current_time - loop_start)
                
                fps_frame_count += 1
                if current_time - fps_calc_start >= 1.0:  # Update average every second
                    self.metrics.average_fps = fps_frame_count / (current_time - fps_calc_start)
                    fps_calc_start = current_time
                    fps_frame_count = 0
                
                # Calculate efficiency score
                if self.metrics.current_fps > 0:
                    self.metrics.efficiency_score = min(100, (self.metrics.current_fps / self.target_fps) * 100)
                
                # GPU memory usage
                if torch.cuda.is_available():
                    self.metrics.gpu_memory_used = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                
                # Emit updates
                self.metrics_updated.emit(self.metrics)
                self.frame_processed.emit(enhanced_frame)
                
                # Frame pacing
                elapsed = time.perf_counter() - loop_start
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)
                else:
                    self.metrics.dropped_frames += 1
                
            except Exception as e:
                print(f"Error processing frame {frame_file}: {e}")
                continue

class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Thermal Super-Resolution Demo")
        self.setMinimumSize(1200, 800)
        
        self.setup_ui()
        self.apply_styles()
    
    def setup_ui(self):
        """Setup main window UI"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        
        # Title
        title = QLabel("Thermal Super-Resolution Demo")
        title.setAlignment(Qt.AlignCenter)
        title.setObjectName("main_title")
        layout.addWidget(title)
        
        # Tab widget
        tab_widget = QTabWidget()
        
        # Video comparison tab
        self.video_tab = VideoComparisonWidget()
        tab_widget.addTab(self.video_tab, "Video Comparison")
        
        # Performance benchmark tab
        self.benchmark_tab = PerformanceBenchmarkWidget()
        tab_widget.addTab(self.benchmark_tab, "Performance Benchmark")
        
        layout.addWidget(tab_widget)
        
        # Status bar
        self.statusBar().showMessage("Ready")
    
    def apply_styles(self):
        """Apply main window styling"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            
            QLabel#main_title {
                font-size: 28px;
                font-weight: bold;
                color: #00d4aa;
                margin: 20px 0;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #00d4aa, stop:1 #0084ff);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }
            
            QTabWidget::pane {
                border: 1px solid #333;
                background-color: #1e1e1e;
            }
            
            QTabBar::tab {
                background-color: #333;
                color: #ffffff;
                padding: 8px 20px;
                margin: 2px;
                border-radius: 4px;
            }
            
            QTabBar::tab:selected {
                background-color: #0084ff;
                color: #ffffff;
            }
            
            QTabBar::tab:hover {
                background-color: #444;
            }
            
            QStatusBar {
                background-color: #2a2a2a;
                color: #ffffff;
                border-top: 1px solid #333;
            }
        """)

def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("Thermal Super-Resolution Demo")
    app.setApplicationVersion("1.0")
    
    # Set application icon if available
    icon_path = "assets/icon.png"
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
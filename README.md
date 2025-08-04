# YOLOv7 Enhanced Color Tracking for JeVois Pro

Advanced object tracking system combining YOLOv7 detection with dynamic color analysis, position prediction, and adaptive context detection for JeVois Pro.

## Features

### 🎯 Core Tracking
- **YOLOv7-tiny detection** with high-resolution support (1024x576)
- **Persistent object IDs** that remain stable across frames
- **Color-based tracking enhancement** using HSV analysis
- **Multi-modal cost function**: Distance (70%) + Color (30%) + Class (20%)

### 🚀 Enhanced Features
- **Position Prediction**: Simple velocity-based prediction for better association
- **Auto-Context Detection**: Automatic switching between INDOOR/CITY/HIGHWAY modes
- **Adaptive Parameters**: Dynamic adjustment based on detected context
- **Hysteresis**: Prevents frequent mode switching for stability

### 🏙️ Context Modes
- **INDOOR**: Small distances (120px), stable tracking for indoor scenes
- **CITY**: Medium distances (150px), optimized for urban environments  
- **HIGHWAY**: Large distances (200px), handles fast-moving vehicles

## Files

- `PyPostYolov7ColorTracker_Final.py` - Enhanced tracker with prediction and context detection
- `DOCUMENTATION_TRACKING_COULEUR.md` - Complete technical documentation
- `connect_jevois_script.sh` - SSH connection script for JeVois Pro

## Installation

1. Copy `PyPostYolov7ColorTracker_Final.py` to `/jevoispro/share/pydnn/post/PyPostYolov7ColorTracker.py`
2. Select the `NPU:Detect:yolov7-hires-tracking` pipeline in JeVois interface

## Serial Output

Enhanced messages with context and prediction info:
```
ENHANCED id:1 class:person color:blue hsv:120,180,200 conf:0.85 quality:0.92 speed:2.3 context:INDOOR x:640 y:360
```

## Technical Details

- **Performance**: ~60 FPS on JeVois Pro NPU
- **Classes**: 80 COCO classes supported
- **Tracking Quality**: Real-time quality scoring (EXCELLENT/GOOD/OK/POOR)
- **Color Analysis**: 9 main colors with reliability validation

Perfect for applications requiring stable object identification and tracking across different environments.
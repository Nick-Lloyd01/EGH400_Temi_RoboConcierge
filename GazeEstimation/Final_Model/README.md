# Temi 5-Layer Subsumption Architecture
## Face Detection & Gaze Tracking System

**Author:** Nicholas Lloyd  
**Date:** October 2025  
**Project:** EGH400

---

## 📋 Overview

This is a sophisticated face detection and gaze tracking system designed for the Temi robot, implementing a **5-layer subsumption architecture** with face-first detection capabilities. The system can operate in two modes: real-time camera processing or offline video analysis.

### Key Features

- ✅ **Face-First Detection Pipeline**: YOLO face detection → MediaPipe landmarks → 3D pose estimation
- ✅ **Gaze Direction Tracking**: Real-time head pose analysis (pitch, yaw, roll)
- ✅ **Distance Estimation**: Adaptive proximity detection at multiple ranges
- ✅ **Behavior Management**: Sustained gaze detection and interaction triggers
- ✅ **Subsumption Architecture**: 5-layer priority-based behavior system
- ✅ **Dual Mode Operation**: Live camera or video file processing
- ✅ **Comprehensive Visualization**: All detections overlayed on output

---

## 🏗️ System Architecture

### 5-Layer Subsumption Hierarchy (Priority Order)

1. **Layer 1: Battery Management** (Highest Priority)
   - Critical battery monitoring
   - Emergency charging behavior

2. **Layer 2: Sensor Safety Systems**
   - Obstacle detection
   - Emergency stop protocols

3. **Layer 3: Human Interaction Management**
   - Gaze-based interaction
   - Sustained attention detection

4. **Layer 4: Face-First Gaze Detection**
   - YOLO face detection
   - MediaPipe facial landmarks (468 points)
   - 3D head pose estimation

5. **Layer 5: Base Robot Behaviors** (Lowest Priority)
   - Patrol mode
   - Idle behaviors

Higher-priority layers can **suppress** lower-priority layers when active.

---

## 📂 Project Structure

```
Final_Model/
├── main.py                      # Main application entry point
├── config.py                    # All configuration parameters
├── detection_layers.py          # Face/person detection classes
├── behavior_manager.py          # Gaze & behavior management
├── subsumption_layers.py        # 5-layer subsumption system
├── utils.py                     # Utility functions
├── test_video_mode.py          # Interactive testing script
├── README.md                    # This file
├── VIDEO_GUIDE.md              # Video processing documentation
├── yolov8n-face.pt             # Face detection model
├── yolov8s.pt                  # Person detection model (backup)
├── InputVideos/                # Place input videos here
└── ResultsVideos/              # Processed videos saved here
```

---

## 🚀 Quick Start

### Installation

1. **Install required packages:**
```bash
pip install ultralytics opencv-python mediapipe numpy
```

2. **Ensure model files exist:**
   - `yolov8n-face.pt` (primary face detection)
   - `yolov8s.pt` (backup person detection)

### Running Live Camera Mode

```python
# In config.py
VIDEO_MODE = False
```

```bash
python main.py
```

### Processing a Video File

```python
# In config.py
VIDEO_MODE = True
INPUT_VIDEO_PATH = 'InputVideos/my_video.MOV'
OUTPUT_VIDEO_PATH = 'ResultsVideos/output.mp4'
```

```bash
python main.py
```

---

## ⚙️ Configuration

All settings are in **`config.py`**. Key parameters:

### Mode Selection
```python
VIDEO_MODE = False              # False = live camera, True = video file
```

### Camera Settings (Live Mode)
```python
CAMERA_INDEX = 0                # Primary camera (try 1 if 0 fails)
CAMERA_RESOLUTION = (640, 480)  # Camera resolution
CAMERA_FPS = 30                 # Frame rate
```

### Video Processing Settings
```python
INPUT_VIDEO_PATH = 'InputVideos/video.MOV'
OUTPUT_VIDEO_PATH = 'ResultsVideos/output.mp4'
SHOW_DISPLAY_WINDOW = True      # False = faster processing
```

### Detection Thresholds
```python
MIN_FACE_CONF = 0.05            # Face detection confidence
YAW_THRESH_DEG = 20.0           # Gaze detection threshold (degrees)
PITCH_THRESH_DEG = 15.0         # Vertical head rotation threshold
```

### Behavior Parameters
```python
SUSTAINED_GAZE_SEC = 3.0        # Seconds of sustained gaze to trigger
APPROACH_DISTANCE = 1.5         # Meters for interaction distance
```

---

## 🎮 Controls

### Keyboard Shortcuts (During Execution)
- **'q'** - Quit program (saves video if in video mode)
- **'r'** - Reset behavior manager state
- **'s'** - Print status summary to console
- **'p'** - Toggle performance logging

---

## 📊 Output Information

### Visual Overlays (Live Display & Video Output)
- **Face bounding boxes** (cyan) with confidence scores
- **468 facial landmarks** (green dots)
- **3D head pose axes** (RGB arrows showing orientation)
- **Gaze direction arrow** (indicates where person is looking)
- **Gaze status** - "LOOKING" or "NOT LOOKING"
- **Person bounding box** (yellow, backup detection)

### Text Information Display

**Top Right Corner:**
- Face detection confidence
- Head pose angles (Yaw, Pitch, Roll)
- Estimated distance in meters

**Top Left Corner:**
- Gaze status
- Sustained gaze timer
- Approach behavior status

**Bottom Left Corner:**
- Active subsumption layer
- Layer status for all 5 layers
- Suppressed layer indicators

---

## 🎯 Detection Pipeline

```
Camera/Video Input
    ↓
YOLO Face Detection (Primary)
    ↓
MediaPipe Face Mesh (468 landmarks)
    ↓
3D Head Pose Estimation (PnP solver)
    ↓
Gaze Direction Analysis
    ↓
Behavior Management
    ↓
Subsumption Layer Arbitration
    ↓
Output Display & Video Write
```

---

## 📏 Distance Estimation

The system estimates distance using face area percentage:

- **Close Range (<1.5m)**: Face area > 4% of frame
- **Medium Range (1.5-3m)**: Face area > 1.5% of frame
- **Far Range (3-5m)**: Face area > 0.8% of frame
- **Too Far (>5m)**: Face area < 0.8% of frame

Distance calculation uses camera focal length and reference face size.

---

## 🔬 Advanced Features

### Adaptive Image Enhancement
Progressive enhancement for distant faces:
- **Light enhancement**: 1.2x contrast, +8 brightness
- **Medium enhancement**: 1.4x contrast, +15 brightness, noise reduction
- **Aggressive enhancement**: 1.6x contrast, +25 brightness, gamma correction

### Proximity Gate System
Multi-range proximity detection with adaptive thresholds based on estimated distance.

### Sustained Gaze Detection
Tracks continuous eye contact over time:
- Requires sustained gaze for configurable duration (default: 3s)
- Triggers approach behavior when threshold met
- Resets if gaze breaks

---

## 🧪 Testing

### Interactive Test Menu
```bash
python test_video_mode.py
```

Provides menu to:
1. Test live camera mode
2. Test video processing mode
3. Run both tests sequentially
4. Show current configuration

### Manual Testing
```bash
# Live camera
python main.py  # with VIDEO_MODE = False

# Video processing
python main.py  # with VIDEO_MODE = True
```

---

## 🐛 Troubleshooting

### Camera Issues
**Problem:** Camera not opening  
**Solution:** Try changing `CAMERA_INDEX` from 0 to 1 in config.py

**Problem:** Low FPS  
**Solution:** Reduce `CAMERA_RESOLUTION` or close other applications

### Video Processing Issues
**Problem:** Video file not found  
**Solution:** Check `INPUT_VIDEO_PATH` is correct, use absolute path if needed

**Problem:** Slow processing  
**Solution:** Set `SHOW_DISPLAY_WINDOW = False` for faster processing

**Problem:** No output video  
**Solution:** Check output directory exists and has write permissions

### Detection Issues
**Problem:** Face not detected at distance  
**Solution:** Lower `MIN_FACE_CONF` threshold (try 0.03)

**Problem:** Gaze detection too sensitive  
**Solution:** Increase `YAW_THRESH_DEG` and `PITCH_THRESH_DEG`

**Problem:** False detections  
**Solution:** Increase `MIN_FACE_CONF` for stricter detection

---

## 📈 Performance Optimization

### For Faster Processing
1. Set `SHOW_DISPLAY_WINDOW = False` (video mode only)
2. Reduce input video resolution before processing
3. Use `OUTPUT_VIDEO_CODEC = 'XVID'` for faster encoding
4. Close other applications to free resources

### For Better Detection
1. Ensure good lighting conditions
2. Position camera at face height
3. Maintain appropriate distance (1-3 meters optimal)
4. Use higher resolution camera/video

---

## 🔄 Common Workflows

### Analyze Pre-Recorded Videos
1. Place video in `InputVideos/` folder
2. Edit `config.py`:
   ```python
   VIDEO_MODE = True
   INPUT_VIDEO_PATH = 'InputVideos/your_video.MOV'
   OUTPUT_VIDEO_PATH = 'ResultsVideos/analyzed_video.mp4'
   SHOW_DISPLAY_WINDOW = False  # Optional: faster processing
   ```
3. Run: `python main.py`
4. Find output in `ResultsVideos/analyzed_video.mp4`

### Real-Time Interaction Testing
1. Edit `config.py`:
   ```python
   VIDEO_MODE = False
   ```
2. Run: `python main.py`
3. Test different head poses and distances
4. Monitor gaze detection and behavior triggers

### Batch Processing Multiple Videos
See `VIDEO_GUIDE.md` for batch processing examples.

---

## 📝 Module Documentation

### `main.py`
- `TemiControlCenter`: Main application class
- `run()`: Main execution loop
- `_process_frame()`: Frame processing pipeline
- `_render_frame()`: Visualization rendering

### `detection_layers.py`
- `FaceDetector`: YOLO-based face detection
- `PersonDetector`: Backup person detection
- `FacePoseEstimator`: Complete pose estimation pipeline

### `behavior_manager.py`
- `GazeClassifier`: Determines if looking at camera
- `ProximityGate`: Distance-based interaction readiness
- `BehaviorManager`: Sustained gaze tracking
- `BehaviorCoordinator`: High-level behavior orchestration

### `subsumption_layers.py`
- `SubsumptionLayer`: Base layer class
- `SubsumptionCoordinator`: Layer arbitration system
- Individual layer implementations (L1-L5)

### `utils.py`
- Geometric calculations (rotation, distance)
- Image enhancement functions
- Coordinate transformations
- Face landmark utilities

### `config.py`
- All system parameters
- Detection thresholds
- Behavior settings
- Visualization colors

---

## 📚 Additional Documentation

- **`VIDEO_GUIDE.md`** - Detailed video processing guide with examples
- **Code comments** - Comprehensive inline documentation
- **Docstrings** - All functions documented with parameters and returns

---

## 🎓 Technical Details

### Models Used
- **YOLOv8n-face**: Custom face detection model (primary)
- **YOLOv8s**: Standard person detection (backup)
- **MediaPipe Face Mesh**: 468-point facial landmark detection

### Coordinate Systems
- **Image coordinates**: OpenCV standard (origin top-left)
- **3D pose**: Right-handed coordinate system
- **Angles**: Euler angles (pitch, yaw, roll) in degrees

### Detection Confidence
- Face detection: 0.05-1.0 (lower = more sensitive)
- MediaPipe landmarks: 0.15 minimum confidence
- Person detection: 0.3 minimum confidence

---

## 🔐 License & Credits

**Author:** Nicholas Lloyd  
**Project:** EGH400  
**Institution:** [Your Institution]  
**Date:** October 2025

### Dependencies
- **Ultralytics YOLOv8**: Object detection
- **OpenCV**: Computer vision operations
- **MediaPipe**: Facial landmark detection
- **NumPy**: Numerical computations

---

## 📞 Support

For issues, questions, or contributions:
1. Check this README and VIDEO_GUIDE.md
2. Review code comments and docstrings
3. Test with `test_video_mode.py`
4. Check console output for error messages

---

## 🚧 Future Enhancements

Potential improvements:
- [ ] Multi-person tracking
- [ ] Eye gaze estimation (beyond head pose)
- [ ] Emotion recognition
- [ ] Person re-identification
- [ ] GPU acceleration support
- [ ] Real-time performance metrics dashboard
- [ ] Web-based configuration interface

---

**Last Updated:** October 2025  
**Version:** 1.0

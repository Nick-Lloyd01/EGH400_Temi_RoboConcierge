# Temi 5-Layer Subsumption Architecture# Temi 5-Layer Subsumption Architecture

## Face Detection & Gaze Tracking System## Face Detection & Gaze Tracking System



**Author:** Nicholas Lloyd  **Author:** Nicholas Lloyd  

**Date:** October 2025  **Date:** October 2025  

**Project:** EGH400**Project:** EGH400



------



## 🚀 Quick Start## 📋 Overview



### Basic UsageThis is a sophisticated face detection and gaze tracking system designed for the Temi robot, implementing a **5-layer subsumption architecture** with face-first detection capabilities. The system can operate in two modes: real-time camera processing or offline video analysis.

```bash

# Run with live camera### Key Features

python main.py

- ✅ **Face-First Detection Pipeline**: YOLO face detection → MediaPipe landmarks → 3D pose estimation

# Run with video file (configure in config.py first)- ✅ **Gaze Direction Tracking**: Real-time head pose analysis (pitch, yaw, roll)

python main.py- ✅ **Distance Estimation**: Adaptive proximity detection at multiple ranges

```- ✅ **Behavior Management**: Sustained gaze detection and interaction triggers

- ✅ **Subsumption Architecture**: 5-layer priority-based behavior system

### Configuration- ✅ **Dual Mode Operation**: Live camera or video file processing

Edit `config.py` to set:- ✅ **Comprehensive Visualization**: All detections overlayed on output

- `VIDEO_MODE = True/False` - Switch between camera and video input- ✅ **Data Logging & Analysis**: Complete CSV logging with 4-way classification

- `INPUT_VIDEO_PATH` - Path to input video- ✅ **Report Generation Tools**: Automatic tables, graphs, and LaTeX output

- `OUTPUT_VIDEO_PATH` - Path to save processed video

- `ENABLE_CSV_LOGGING = True/False` - Enable data logging---

- `CSV_OUTPUT_PATH` - Path to save CSV results

## 🏗️ System Architecture

### Keyboard Controls (During Execution)

- **'q'** - Quit program### 5-Layer Subsumption Hierarchy (Priority Order)

- **'p'** - Mark frame as False Positive (data logging mode)

- **'n'** - Mark frame as False Negative (data logging mode)1. **Layer 1: Battery Management** (Highest Priority)

- **'r'** - Reset behavior state   - Critical battery monitoring

- **'s'** - Print status summary   - Emergency charging behavior



---2. **Layer 2: Sensor Safety Systems**

   - Obstacle detection

## 📊 Data Analysis & Report Generation   - Emergency stop protocols



### Step 1: Collect Data3. **Layer 3: Human Interaction Management**

```bash   - Gaze-based interaction

python main.py   - Sustained attention detection

```

- Set `ENABLE_CSV_LOGGING = True` in config.py4. **Layer 4: Face-First Gaze Detection**

- Process your test videos   - YOLO face detection

- Press 'p'/'n' to annotate false positives/negatives   - MediaPipe facial landmarks (468 points)

   - 3D head pose estimation

### Step 2: Analyze Results

```bash5. **Layer 5: Base Robot Behaviors** (Lowest Priority)

python analyze_results.py "Control Tests/Control Result CSV/results_1324.csv"   - Patrol mode

```   - Idle behaviors

Generates:

- Confusion Matrix (TP, FP, FN, TN)Higher-priority layers can **suppress** lower-priority layers when active.

- Performance Metrics (Accuracy, Precision, Recall)

- Accuracy by Distance tables---

- LaTeX tables for reports

## 📂 Project Structure

### Step 3: Create Graphs

```bash```

python create_graphs.py "Control Tests/Control Result CSV/results_1324.csv"Final_Model/

```├── main.py                      # Main application entry point

Creates 6 publication-ready graphs:├── config.py                    # All configuration parameters

- FPS performance├── detection_layers.py          # Face/person detection classes

- Confidence vs distance├── behavior_manager.py          # Gaze & behavior management

- Confusion matrix├── subsumption_layers.py        # 5-layer subsumption system

- Accuracy by distance├── utils.py                     # Utility functions

- Gaze distribution├── test_video_mode.py          # Interactive testing script

- Detection method comparison├── README.md                    # This file

├── VIDEO_GUIDE.md              # Video processing documentation

**Note:** Requires matplotlib, pandas, numpy├── yolov8n-face.pt             # Face detection model

```bash├── yolov8s.pt                  # Person detection model (backup)

pip install matplotlib pandas numpy├── InputVideos/                # Place input videos here

```└── ResultsVideos/              # Processed videos saved here

```

---

---

## 📁 Project Structure

## 🚀 Quick Start

```

Final_Model/### Installation

├── main.py                    # Main application

├── config.py                  # Configuration settings1. **Install required packages:**

├── detection_layers.py        # Face detection & pose estimation```bash

├── subsumption_layers.py      # Subsumption architecturepip install ultralytics opencv-python mediapipe numpy

├── behavior_manager.py        # Behavior coordination```

├── data_logger.py            # CSV data logging

├── utils.py                  # Utility functions2. **Ensure model files exist:**

├── analyze_results.py        # Analysis tool for reports   - `yolov8n-face.pt` (primary face detection)

├── create_graphs.py          # Graph generation tool   - `yolov8s.pt` (backup person detection)

├── Documentation/            # All documentation files

│   ├── README.md             # Complete documentation### Running Live Camera Mode

│   ├── WORKFLOW.md           # Step-by-step guide

│   ├── CSV_LOGGING_GUIDE.md  # Data logging details```python

│   ├── REPORT_TOOLS.md       # Analysis tools reference# In config.py

│   ├── QUICKREF.txt          # Quick reference cardVIDEO_MODE = False

│   └── VIDEO_GUIDE.md        # Video processing guide```

├── Control Tests/            # Test videos and results

│   ├── IMG_1324.MOV          # Full test video```bash

│   ├── Move_5to1.mov         # 5m→1m approach testpython main.py

│   ├── Move_1to5.mov         # 1m→5m retreat test```

│   ├── Control Result CSV/   # CSV output files

│   └── Control Results Videos/ # Processed videos### Processing a Video File

└── yolov8*.pt               # YOLO model files

``````python

# In config.py

---VIDEO_MODE = True

INPUT_VIDEO_PATH = 'InputVideos/my_video.MOV'

## 🎯 Key FeaturesOUTPUT_VIDEO_PATH = 'ResultsVideos/output.mp4'

```

- ✅ **Double-Layer Detection**: YOLO + MediaPipe for robust face detection

- ✅ **3D Pose Estimation**: Real-time head orientation (yaw, pitch, roll)```bash

- ✅ **Distance Estimation**: Adaptive proximity detectionpython main.py

- ✅ **Gaze Classification**: 5-layer subsumption architecture```

- ✅ **Data Logging**: Complete CSV export with 4-way classification

- ✅ **Analysis Tools**: Automatic table and graph generation---

- ✅ **Dual Mode**: Live camera or video file processing

- ✅ **Frontal Face Filtering**: Rejects back-of-head false detections## ⚙️ Configuration



---All settings are in **`config.py`**. Key parameters:



## 📖 Documentation### Mode Selection

```python

For complete documentation, see the `Documentation/` folder:VIDEO_MODE = False              # False = live camera, True = video file

```

- **README.md** - Complete system documentation

- **WORKFLOW.md** - Step-by-step workflow guide### Camera Settings (Live Mode)

- **CSV_LOGGING_GUIDE.md** - Data collection and CSV format```python

- **REPORT_TOOLS.md** - Analysis tools and metricsCAMERA_INDEX = 0                # Primary camera (try 1 if 0 fails)

- **QUICKREF.txt** - Quick command referenceCAMERA_RESOLUTION = (640, 480)  # Camera resolution

- **VIDEO_GUIDE.md** - Video processing guideCAMERA_FPS = 30                 # Frame rate

```

---

### Video Processing Settings

## 🔧 Requirements```python

INPUT_VIDEO_PATH = 'InputVideos/video.MOV'

```bashOUTPUT_VIDEO_PATH = 'ResultsVideos/output.mp4'

pip install ultralytics opencv-python mediapipe numpy pandas matplotlibSHOW_DISPLAY_WINDOW = True      # False = faster processing

``````



**Required Files:**### Detection Thresholds

- `yolov8n-face.pt` - Face detection model```python

- `yolov8s.pt` - General detection model (backup)MIN_FACE_CONF = 0.05            # Face detection confidence

YAW_THRESH_DEG = 20.0           # Gaze detection threshold (degrees)

---PITCH_THRESH_DEG = 15.0         # Vertical head rotation threshold

```

## 📊 Report Metrics

### Behavior Parameters

The system provides comprehensive metrics for academic reports:```python

SUSTAINED_GAZE_SEC = 3.0        # Seconds of sustained gaze to trigger

**Classification:**APPROACH_DISTANCE = 1.5         # Meters for interaction distance

- True Positives, False Positives, False Negatives, True Negatives```



**Performance Metrics:**---

- Accuracy, Precision, Recall, Specificity, F1 Score

## 🎮 Controls

**Analysis:**

- Detection accuracy by distance range### Keyboard Shortcuts (During Execution)

- Processing performance (FPS)- **'q'** - Quit program (saves video if in video mode)

- Gaze classification statistics- **'r'** - Reset behavior manager state

- Detection method comparison- **'s'** - Print status summary to console

- **'p'** - Mark current frame as **False Positive** (data logging)

---- **'n'** - Mark current frame as **False Negative** (data logging)



## 🐛 Troubleshooting---



**Camera not opening:**## 📊 Output Information

- Try changing `CAMERA_INDEX` from 0 to 1 in config.py

### Visual Overlays (Live Display & Video Output)

**Video file not found:**- **Face bounding boxes** (cyan) with confidence scores

- Use absolute path in `INPUT_VIDEO_PATH`- **468 facial landmarks** (green dots)

- **3D head pose axes** (RGB arrows showing orientation)

**Slow processing:**- **Gaze direction arrow** (indicates where person is looking)

- Set `SHOW_DISPLAY_WINDOW = False` for faster video processing- **Gaze status** - "LOOKING" or "NOT LOOKING"

- **Person bounding box** (yellow, backup detection)

**Low detection accuracy:**

- Adjust `MIN_FACE_CONF` threshold in config.py### Text Information Display

- Check lighting conditions

**Top Right Corner:**

For detailed troubleshooting, see `Documentation/README.md`- Face detection confidence

- Head pose angles (Yaw, Pitch, Roll)

---- Estimated distance in meters



## 📞 Contact**Top Left Corner:**

- Gaze status

**Author:** Nicholas Lloyd  - Sustained gaze timer

**Project:** EGH400 - Temi Robot Navigation System  - Approach behavior status

**Date:** October 2025

**Bottom Left Corner:**

---- Active subsumption layer

- Layer status for all 5 layers

## 📝 License- Suppressed layer indicators



Academic project for EGH400.---


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

## 📊 Data Logging & Report Generation

### Quick Start - Generate Report Data

**Step 1: Run Control Tests**
```bash
python run_control_tests.py
```
- Select test video from menu
- Watch playback and annotate:
  - Press **'p'** for False Positives (wrong detections)
  - Press **'n'** for False Negatives (missed faces)
  - System auto-marks True Positives and True Negatives
- CSV results saved automatically

**Step 2: Analyze Results**
```bash
python analyze_results.py "Control Tests/Control Result CSV/results_1324.csv"
```
Generates:
- Confusion Matrix (TP, FP, FN, TN)
- Performance Metrics (Accuracy, Precision, Recall, F1)
- Accuracy by Distance tables
- LaTeX tables for reports

**Step 3: Create Graphs**
```bash
python create_graphs.py "Control Tests/Control Result CSV/results_1324.csv"
```
Creates 6 publication-ready graphs:
- FPS performance over time
- Confidence vs distance scatter plot
- Confusion matrix visualization
- Accuracy by distance bar chart
- Gaze distribution pie chart
- Detection method comparison

### Data Logging Features

**4-Way Classification System:**
- **True Positive (TP)**: Face detected correctly (automatic)
- **False Positive (FP)**: Wrong detection - press 'p' key
- **False Negative (FN)**: Missed face - press 'n' key
- **True Negative (TN)**: No face correctly identified (automatic)

**Metrics Logged Per Frame:**
- Detection confidence and method
- Pose angles (yaw, pitch, roll)
- Distance estimation
- Gaze classification
- Processing performance (FPS, frame time)
- Manual annotations

**CSV Output Includes:**
- Per-frame detailed metrics
- Confusion matrix summary
- Performance statistics
- Accuracy/Precision/Recall calculations

### Documentation
- **WORKFLOW.md** - Complete step-by-step guide
- **CSV_LOGGING_GUIDE.md** - Data collection details
- **REPORT_TOOLS.md** - Analysis tools reference
- **QUICKREF.txt** - Quick command reference

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

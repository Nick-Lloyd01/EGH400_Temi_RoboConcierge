"""
Configuration Module for Temi 5-Layer Subsumption Architecture

This module contains all configuration constants, thresholds, and parameters
used throughout the Temi gaze detection and behavior system.
"""

# =============================================================================
# FILE PATHS - CHANGE THESE FOR DIFFERENT TEST VIDEOS
# =============================================================================
# 
# QUICK START: To process a different video, just change TEST_NAME below
# Example: TEST_NAME = '1324'  → processes IMG_1324.MOV
#          TEST_NAME = '5to1'  → processes Move_5to1.mov
#
# All output files (video, CSV, graphs) will automatically use this name
# =============================================================================

# Test video identifier (used for naming outputs)
TEST_NAME = 'Distance_Test'                     # Change this for different test runs (e.g., '1324', '5to1', '1to5')

#   python analyze_results.py "Control Tests/Control Result CSV/Distance_Test/output_Distance_Test_CSV_results.csv"
#   python create_graphs.py "Control Tests/Control Result CSV/Distance_Test/output_Distance_Test_CSV_results.csv"

# Input/Output Paths - All paths use TEST_NAME for consistency
INPUT_VIDEO_PATH = f'Control Tests/{TEST_NAME}.MOV'
OUTPUT_VIDEO_PATH = f'Control Tests/Control Results Videos/output_{TEST_NAME}_Result.mp4'
CSV_OUTPUT_PATH = f'Control Tests/Control Result CSV/{TEST_NAME}/output_{TEST_NAME}_CSV_results.csv'
GRAPH_OUTPUT_DIR = f'Control Tests/Control Result CSV/{TEST_NAME}/graphs'  # Graphs saved here

# NOTE: When you run analyze_results.py or create_graphs.py, use CSV_OUTPUT_PATH:
#   python analyze_results.py "Control Tests/Control Result CSV/{TEST_NAME}/output_{TEST_NAME}_CSV_results.csv"
#   python create_graphs.py "Control Tests/Control Result CSV/{TEST_NAME}/output_{TEST_NAME}_CSV_results.csv"



# Model file paths
YOLO_MODEL = 'yolov8s.pt'              # General purpose detection (backup)
FACE_YOLO_MODEL = 'yolov8n-face.pt'    # Primary face detection model


# =============================================================================
# DETECTION CONFIDENCE THRESHOLDS
# =============================================================================

# Person detection thresholds (used as backup)
MIN_PERSON_CONF = 0.5                  # Minimum confidence for person detection

# Face detection thresholds (primary detection method)
MIN_FACE_CONF = 0.5                  # Higher confidence for more reliable face detection
MAX_NUM_FACES = 2                      # Maximum number of faces to process

# Frontal face filtering (prevents back-of-head false detections)
REQUIRE_FRONTAL_FACE = True            # Only accept frontal faces (reject back of head)
MAX_YAW_FOR_FRONTAL = 90.0             # Maximum yaw angle for frontal face (degrees)
                                        # 90° = profile view, 180° = back of head
                                        # Recommended: 70-100° to filter back-of-head

# MediaPipe face mesh confidence settings (only runs if face detected by YOLO)
MIN_DETECTION_CONFIDENCE = 0.15        # Low confidence for landmarks - double layer check

# =============================================================================
# MODE CONFIGURATION
# =============================================================================

# Video processing mode
VIDEO_MODE = True                      # True = process video file, False = live camera

# Display settings
SHOW_DISPLAY_WINDOW = True             # False = faster processing without display window

# Data logging
ENABLE_CSV_LOGGING = True              # True = log all metrics to CSV for analysis

# Video output settings
OUTPUT_VIDEO_CODEC = 'mp4v'            # Codec for output video ('mp4v', 'XVID', 'H264', etc.)
OUTPUT_VIDEO_FPS = None                # FPS for output (None = match input video fps)

# =============================================================================
# DATA LOGGING CONFIGURATION
# =============================================================================

# Logging configuration
LOG_FRAME_METRICS = True                # Log per-frame detection data
LOG_PERFORMANCE = True                  # Log FPS and processing time
LOG_DETECTION_ACCURACY = True           # Log face/landmark detection success
LOG_DISTANCE_ESTIMATES = True           # Log distance estimation data
LOG_GAZE_CLASSIFICATION = True          # Log gaze detection results
LOG_FALSE_POSITIVES = True              # Enable manual false positive/negative logging (press 'p' or 'n')

# Additional report metrics
LOG_CONFIDENCE_SCORES = True            # Log YOLO and MediaPipe confidence scores
LOG_POSE_ANGLES_DETAIL = True           # Log detailed yaw/pitch/roll measurements
LOG_ENVIRONMENTAL_FACTORS = True        # Log lighting, motion blur indicators (future)
LOG_PROCESSING_STAGES = True            # Log time spent in each detection stage (future)


# =============================================================================
# GAZE DETECTION PARAMETERS
# =============================================================================

# Head pose thresholds for determining if someone is looking at camera
YAW_THRESH_DEG = 20.0                  # Horizontal head rotation: -20° to +20° 
PITCH_THRESH_DEG = 15.0                # Vertical head rotation (will be overridden by range)
PITCH_MIN_DEG = -180.0                 # Minimum pitch for looking (full range)
PITCH_MAX_DEG = 180.0                  # Maximum pitch for looking (full range)

# Gaze direction adjustment
FLIP_GAZE_DIRECTION = False            # Set to True if arrow points wrong direction

# =============================================================================
# DISTANCE ESTIMATION AND PROXIMITY DETECTION
# =============================================================================

# Distance-adaptive proximity detection thresholds
FACE_AREA_FRAC_CLOSE = 0.04           # Close range (<1.5m): 4% of frame
FACE_AREA_FRAC_MEDIUM = 0.015         # Medium range (1.5-3m): 1.5% of frame  
FACE_AREA_FRAC_FAR = 0.008            # Far range (3-5m): 0.8% of frame
MIN_FACE_PIXELS = 20                  # Minimum face size in pixels (reduced for distant)

# Distance estimation parameters for camera calculations
CLOSE_DISTANCE_THRESHOLD = 1.5        # meters - boundary for close range
MEDIUM_DISTANCE_THRESHOLD = 3.0       # meters - boundary for medium range
REFERENCE_FACE_SIZE_MM = 150          # Average human face width in millimeters
CAMERA_FOCAL_LENGTH_MM = 4.0          # Typical webcam focal length

# =============================================================================
# BEHAVIOR AND TIMING PARAMETERS
# =============================================================================

# Sustained gaze detection for behavior triggering
SUSTAINED_GAZE_SEC = 3.0              # Seconds of sustained gaze to trigger behavior

# =============================================================================
# 3D FACIAL LANDMARK MODEL FOR POSE ESTIMATION
# =============================================================================

# 3D model points for solvePnP head pose estimation (in millimeters)
# These correspond to specific MediaPipe landmark indices
MODEL_POINTS = [
    (   0.0,    0.0,    0.0),          # Nose tip        (landmark 1)
    (   0.0, -63.6, -12.5),            # Chin            (landmark 199)
    (-43.3,  32.7, -26.0),             # Left eye left   (landmark 33)
    ( 43.3,  32.7, -26.0),             # Right eye right (landmark 263)
    (-28.9, -28.9, -24.1),             # Mouth left      (landmark 61)
    ( 28.9, -28.9, -24.1)              # Mouth right     (landmark 291)
]

# MediaPipe landmark indices corresponding to the 3D model points above
LANDMARK_IDS = [1, 199, 33, 263, 61, 291]

# =============================================================================
# TEMI ROBOT SUBSUMPTION LAYER PARAMETERS
# =============================================================================

# Layer 1: Battery Management
CRITICAL_BATTERY_LEVEL = 15.0         # % - Emergency charging required
LOW_BATTERY_LEVEL = 25.0              # % - Start seeking charging station

# Layer 2: Sensor Safety Parameters  
OBSTACLE_SAFETY_DISTANCE = 0.3        # meters - Emergency stop distance
SENSOR_UPDATE_RATE = 10.0             # Hz - Sensor reading frequency

# Layer 3: Human Interaction Parameters
APPROACH_DISTANCE = 1.5               # meters - Distance to initiate approach
INTERACTION_TIMEOUT = 30.0            # seconds - Maximum interaction duration
MIN_SUSTAINED_GAZE_FOR_HELP = 2.0     # seconds - Gaze time indicating need for help

# Layer 4: Gaze Detection Enhancement Levels
MAX_ENHANCEMENT_LEVELS = 3            # Number of progressive enhancement attempts

# Layer 5: Base Behavior Parameters
PATROL_BATTERY_THRESHOLD = 50.0       # % - Minimum battery for patrol mode
IDLE_TIMEOUT = 60.0                   # seconds - Time before switching to standby

# =============================================================================
# IMAGE PREPROCESSING PARAMETERS
# =============================================================================

# Enhancement parameters for distant face detection
ENHANCEMENT_PARAMS = {
    'light': {
        'alpha': 1.2,                  # Contrast multiplier
        'beta': 8,                     # Brightness addition
        'blur_reduce': False,          # Apply noise reduction
        'gamma': 1.0                   # Gamma correction (1.0 = no change)
    },
    'medium': {
        'alpha': 1.4,
        'beta': 15,
        'blur_reduce': True,
        'gamma': 0.9
    },
    'aggressive': {
        'alpha': 1.6,
        'beta': 25,
        'blur_reduce': True,
        'gamma': 0.8
    }
}

# Face region expansion for MediaPipe processing
FACE_REGION_MARGIN = 0.2              # 20% margin around detected face

# =============================================================================
# DISPLAY AND UI PARAMETERS
# =============================================================================

# Visualization colors (BGR format for OpenCV)
COLORS = {
    'person_box': (255, 255, 0),       # Yellow for person detection
    'face_box': (0, 255, 255),         # Cyan for face detection  
    'landmarks': (0, 255, 0),          # Green for facial landmarks
    'pose_axes': {
        'x': (0, 0, 255),              # Red for X-axis (pitch)
        'y': (0, 255, 0),              # Green for Y-axis (yaw)  
        'z': (255, 0, 0)               # Blue for Z-axis (roll)
    },
    'status_text': (255, 255, 255),    # White for status text
    'warning': (0, 165, 255),          # Orange for warnings
    'error': (0, 0, 255),              # Red for errors
    'success': (0, 255, 0)             # Green for success
}

# Text display parameters
TEXT_FONT = 0                          # cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 0.8                       # Base text scale
TEXT_THICKNESS = 2                     # Default text thickness

# Text size multiplier for all detection overlays
# Increase this value to make ALL text larger (1.0 = normal, 1.5 = 50% larger, 2.0 = double size)
TEXT_MULTIPLIER = 1.0                  # Global multiplier for all detection text

# =============================================================================
# DEBUGGING AND LOGGING
# =============================================================================

# Debug output control
ENABLE_DEBUG_PRINTS = True            # Enable/disable console debug output
ENABLE_PERFORMANCE_LOGGING = False    # Log timing information
SAVE_DEBUG_FRAMES = False             # Save frames for debugging

# Frame rate limiting
TARGET_FPS = 30                       # Target processing frame rate
MAX_PROCESSING_TIME = 0.1             # Maximum time per frame (seconds)

# =============================================================================
# CAMERA PARAMETERS
# =============================================================================

# Camera selection and settings
CAMERA_INDEX = 0                      # Primary camera index (try 1 if 0 fails)
CAMERA_BACKUP_INDEX = 1               # Backup camera index
CAMERA_RESOLUTION = (640, 480)        # Default camera resolution (width, height)
CAMERA_FPS = 30                       # Camera capture frame rate


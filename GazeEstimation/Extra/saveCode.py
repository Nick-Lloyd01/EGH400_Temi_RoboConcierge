"""

source ~/venvs/TemiGaze/bin/activate



python temi_5layer_subsumption.py







gaze_offline.py - Subsumption Architecture Version
Converted from monolithic structure to subsumption-based gaze detection system.

Pipeline (every frame):
  YOLOv8 (person gate) -> MediaPipe FaceMesh -> solvePnP -> (yaw, pitch, roll)
  -> GazeClassifier (+ optional refine_gaze hook) -> BehaviorManager

Design:
  - Subsumption layers with priority handling
  - Modular components for easy extension
  - Configurable thresholds and parameters
  - Behavior triggers for sustained gaze detection
"""

import cv2
import numpy as np
import math
import time
from ultralytics import YOLO
import mediapipe as mp
from dataclasses import dataclass
from typing import List, Optional, Tuple

# -----------------------
# Configuration Constants
# -----------------------
# YOLO_MODEL = 'yolov8n-face.pt'
YOLO_MODEL = 'yolov8s.pt'
MIN_PERSON_CONF = 0.3  # Lowered for better distant detection
MAX_NUM_FACES = 1
MIN_DETECTION_CONFIDENCE = 0.3  # Lowered for distant faces

# Gaze direction settings
FLIP_GAZE_DIRECTION = False  # Set to True if the arrow still points the wrong way

# Head-pose thresholds for gaze detection
YAW_THRESH_DEG = 20.0  # -20 to +20 degrees
PITCH_THRESH_DEG = 15.0  # Will be overridden by range check
PITCH_MIN_DEG = -180.0  # Minimum pitch for looking (extended to handle -175°)
PITCH_MAX_DEG = 180.0   # Maximum pitch for looking (full range)

# Distance-adaptive proximity detection
FACE_AREA_FRAC_CLOSE = 0.04     # Close range (< 1.5m): 4% of frame
FACE_AREA_FRAC_MEDIUM = 0.015   # Medium range (1.5-3m): 1.5% of frame  
FACE_AREA_FRAC_FAR = 0.008      # Far range (3-5m): 0.8% of frame
MIN_FACE_PIXELS = 30            # Minimum face size in pixels (width or height)

# Distance estimation parameters
CLOSE_DISTANCE_THRESHOLD = 1.5   # meters
MEDIUM_DISTANCE_THRESHOLD = 3.0  # meters
REFERENCE_FACE_SIZE_MM = 150     # Average face width in mm
CAMERA_FOCAL_LENGTH_MM = 4.0     # Typical webcam focal length

# Sustained gaze time to trigger behavior
SUSTAINED_GAZE_SEC = 3.0

# 3D model points for solvePnP (same as original)
MODEL_POINTS = np.array([
    (   0.0,    0.0,    0.0),    # Nose tip        (landmark 1)
    (   0.0, -63.6, -12.5),      # Chin            (landmark 199)
    (-43.3,  32.7, -26.0),       # Left eye left   (landmark 33)
    ( 43.3,  32.7, -26.0),       # Right eye right (landmark 263)
    (-28.9, -28.9, -24.1),       # Mouth left      (landmark 61)
    ( 28.9, -28.9, -24.1)        # Mouth right     (landmark 291)
], dtype=np.float64)
LANDMARK_IDS = [1, 199, 33, 263, 61, 291]

# -----------------------
# Utility Functions
# -----------------------

def rotation_to_euler(rvec: np.ndarray) -> Tuple[float, float, float]:
    """Convert OpenCV rotation vector to Euler angles in degrees."""
    R, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2,1], R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.degrees(x), np.degrees(y), np.degrees(z)

def face_box_from_pts(pts2d: np.ndarray) -> Tuple[int,int,int,int]:
    """Calculate bounding box (x1,y1,x2,y2) from a set of 2D points."""
    xs = pts2d[:,0]
    ys = pts2d[:,1]
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

def face_area_fraction(box, frame_shape) -> float:
    """Calculate face box area as fraction of total frame area."""
    x1, y1, x2, y2 = box
    area = max(0, x2-x1) * max(0, y2-y1)
    H, W = frame_shape[:2]
    return area / float(W*H + 1e-6)

def estimate_distance_from_face(face_width_pixels: float, frame_width: int) -> float:
    """
    Estimate distance to face using pinhole camera model.
    Returns distance in meters.
    """
    if face_width_pixels <= 0:
        return float('inf')
    
    # Convert camera specs to appropriate units
    # Typical webcam: focal length ~4mm, sensor width ~4.8mm
    sensor_width_mm = 4.8
    focal_length_pixels = (CAMERA_FOCAL_LENGTH_MM * frame_width) / sensor_width_mm
    
    # Distance = (real_object_size * focal_length_pixels) / object_size_pixels
    distance_mm = (REFERENCE_FACE_SIZE_MM * focal_length_pixels) / face_width_pixels
    return distance_mm / 1000.0  # Convert to meters

def get_adaptive_proximity_threshold(distance_meters: float) -> float:
    """
    Return appropriate proximity threshold based on estimated distance.
    """
    if distance_meters < CLOSE_DISTANCE_THRESHOLD:
        return FACE_AREA_FRAC_CLOSE
    elif distance_meters < MEDIUM_DISTANCE_THRESHOLD:
        return FACE_AREA_FRAC_MEDIUM
    else:
        return FACE_AREA_FRAC_FAR

def preprocess_for_distant_detection(frame, estimated_distance: Optional[float] = None) -> np.ndarray:
    """
    Preprocess frame to improve distant face detection.
    Returns enhanced frame for MediaPipe processing.
    """
    if estimated_distance is None or estimated_distance < MEDIUM_DISTANCE_THRESHOLD:
        return frame  # No preprocessing needed for close/medium range
    
    # For distant detection, apply enhancement techniques
    enhanced = frame.copy()
    
    # 1. Increase contrast and brightness for better feature visibility
    alpha = 1.3  # Contrast control (1.0-3.0)
    beta = 10    # Brightness control (0-100)
    enhanced = cv2.convertScaleAbs(enhanced, alpha=alpha, beta=beta)
    
    # 2. Apply sharpening filter to enhance edges
    kernel = np.array([[-1,-1,-1],
                       [-1, 9,-1],
                       [-1,-1,-1]])
    enhanced = cv2.filter2D(enhanced, -1, kernel)
    
    # 3. Optional: Apply histogram equalization to face region if we have it
    # (This would require face region detection first, so skip for now)
    
    return enhanced

def get_adaptive_detection_confidence(distance_meters: float) -> float:
    """
    Return appropriate MediaPipe detection confidence based on distance.
    """
    if distance_meters < CLOSE_DISTANCE_THRESHOLD:
        return 0.5  # Standard confidence for close range
    elif distance_meters < MEDIUM_DISTANCE_THRESHOLD:
        return 0.3  # Lower confidence for medium range
    else:
        return 0.2  # Very low confidence for far range

# -----------------------
# Subsumption Architecture Classes
# -----------------------

class PersonDetector:
    """YOLOv8-based person detection layer with distance-adaptive confidence."""
    
    def __init__(self, model_path: str = YOLO_MODEL, min_conf: float = MIN_PERSON_CONF):
        self.model = YOLO(model_path)
        self.base_min_conf = min_conf

    def detect(self, frame, adaptive_conf: bool = True) -> List[Tuple[int,int,int,int,float]]:
        """Return person boxes [(x1,y1,x2,y2,conf), ...] with optional adaptive confidence."""
        res = self.model(frame, verbose=False)[0]
        out = []
        frame_area = frame.shape[0] * frame.shape[1]
        
        for det in res.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = det
            if int(cls) == 0:  # class 0 = person
                # Calculate person box size for distance estimation
                person_area = (x2 - x1) * (y2 - y1)
                person_area_fraction = person_area / frame_area
                
                # Adaptive confidence: lower threshold for smaller (distant) persons
                if adaptive_conf:
                    if person_area_fraction > 0.15:  # Large person (close)
                        min_conf = self.base_min_conf
                    elif person_area_fraction > 0.05:  # Medium person
                        min_conf = max(0.2, self.base_min_conf - 0.2)
                    else:  # Small person (distant)
                        min_conf = max(0.15, self.base_min_conf - 0.35)
                else:
                    min_conf = self.base_min_conf
                
                if conf >= min_conf:
                    out.append((int(x1), int(y1), int(x2), int(y2), float(conf)))
        return out

class FacePoseEstimator:
    """MediaPipe FaceMesh + solvePnP head pose estimation layer with adaptive confidence."""
    
    def __init__(self, max_faces: int = MAX_NUM_FACES):
        self.max_faces = max_faces
        self.mp_face_mesh = mp.solutions.face_mesh
        # Initialize with default confidence, will be updated adaptively
        self.mesh = None
        self.current_confidence = MIN_DETECTION_CONFIDENCE
        self._init_mesh(MIN_DETECTION_CONFIDENCE)

    def _init_mesh(self, detection_confidence: float):
        """Initialize MediaPipe mesh with given confidence."""
        self.current_confidence = detection_confidence
        self.mesh = self.mp_face_mesh.FaceMesh(
            refine_landmarks=True,
            static_image_mode=False,
            max_num_faces=self.max_faces,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=max(0.1, detection_confidence - 0.2)  # Tracking can be lower
        )

    def estimate(self, frame, estimated_distance: Optional[float] = None) -> Optional[dict]:
        """Return dict with pitch, yaw, roll, pts2d, face_kps, distance and ok status."""
        
        # Adapt confidence based on distance
        if estimated_distance is not None:
            new_confidence = get_adaptive_detection_confidence(estimated_distance)
            if abs(new_confidence - self.current_confidence) > 0.05:  # Only reinit if significant change
                self._init_mesh(new_confidence)
        
        # Apply preprocessing for distant detection
        processed_frame = preprocess_for_distant_detection(frame, estimated_distance)
        
        H, W = frame.shape[:2]  # Use original frame dimensions
        rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        results = self.mesh.process(rgb)
        
        if not results.multi_face_landmarks:
            return None

        lm = results.multi_face_landmarks[0].landmark
        pts2d = []
        face_kps = []
        
        for idx in LANDMARK_IDS:
            x = int(lm[idx].x * W)
            y = int(lm[idx].y * H)
            pts2d.append([x, y])
            face_kps.append({'x': float(x), 'y': float(y)})
            
        pts2d = np.array(pts2d, dtype=np.float64)

        # Calculate face width for distance estimation
        xs = [pt[0] for pt in pts2d]
        face_width_pixels = max(xs) - min(xs)
        
        # Estimate distance if not provided
        if estimated_distance is None:
            estimated_distance = estimate_distance_from_face(face_width_pixels, W)

        # Camera intrinsics (pinhole approximation)
        focal = W
        cam_matrix = np.array([[focal, 0, W/2],
                               [0, focal, H/2],
                               [0,     0,   1]], dtype=np.float64)
        dist_coeffs = np.zeros((4,1), dtype=np.float64)

        ok, rvec, tvec = cv2.solvePnP(
            MODEL_POINTS, pts2d, cam_matrix, dist_coeffs, 
            flags=cv2.SOLVEPNP_ITERATIVE)
            
        if not ok:
            return {'ok': False}

        pitch, yaw, roll = rotation_to_euler(rvec)
        return {
            'ok': True,
            'pitch': float(pitch),
            'yaw': float(yaw),
            'roll': float(roll),
            'pts2d': pts2d,
            'face_kps': face_kps,
            'rvec': rvec,
            'tvec': tvec,
            'cam_matrix': cam_matrix,
            'dist_coeffs': dist_coeffs,
            'estimated_distance': float(estimated_distance),
            'face_width_pixels': float(face_width_pixels),
            'detection_confidence': float(self.current_confidence)
        }

@dataclass
class GazeClassifier:
    """Gaze classification layer based on head pose thresholds."""
    yaw_thresh: float = YAW_THRESH_DEG
    pitch_min: float = PITCH_MIN_DEG
    pitch_max: float = PITCH_MAX_DEG

    def is_looking(self, pitch: float, yaw: float) -> bool:
        """Determine if person is looking based on head pose ranges."""
        yaw_ok = abs(yaw) <= self.yaw_thresh  # -20 to +20 degrees
        
        # Handle pitch with full range since -175° should be valid for looking at camera
        # Since you're getting -175° when looking at camera, we'll accept the full range
        pitch_ok = True  # Accept all pitch values for now
        
        return yaw_ok and pitch_ok

def refine_gaze(is_looking: bool, pitch: float, yaw: float, roll: float, 
                face_crop, landmarks_2d: np.ndarray) -> bool:
    """
    Accuracy refinement hook (placeholder for future enhancements).
    Later: integrate Roboflow eye-gaze, iris tracking, temporal smoothing, etc.
    For now: return the coarse decision unchanged.
    """
    # TODO: integrate eye model here and override is_looking if needed
    return is_looking

class ProximityGate:
    """Proximity detection layer - adaptive thresholds for monitoring (not gating detection)."""
    
    def __init__(self, min_frac: float = FACE_AREA_FRAC_CLOSE):
        self.default_min_frac = float(min_frac)

    def is_close(self, box, frame_shape, estimated_distance: Optional[float] = None) -> Tuple[bool, float, float]:
        """
        Check if face meets proximity thresholds for monitoring purposes.
        NOTE: This no longer gates gaze detection - used for display/debugging only.
        Returns (is_close, area_fraction, threshold_used)
        """
        area_frac = face_area_fraction(box, frame_shape)
        
        # Use adaptive threshold if distance is estimated
        if estimated_distance is not None:
            threshold = get_adaptive_proximity_threshold(estimated_distance)
        else:
            threshold = self.default_min_frac
        
        # Also check minimum pixel size to avoid tiny detections
        x1, y1, x2, y2 = box
        face_width = max(0, x2-x1)
        face_height = max(0, y2-y1)
        min_size_ok = face_width >= MIN_FACE_PIXELS and face_height >= MIN_FACE_PIXELS
        
        is_close = area_frac >= threshold and min_size_ok
        return is_close, area_frac, threshold

class SensorGate:
    """Sensor interrupt layer (highest priority in subsumption)."""
    
    def __init__(self):
        self.state = {
            'emergency_stop': False,
            'obstacle_close': False,
            'override_hold': False
        }
    
    def update_from_sensors(self, sensor_state: dict):
        """Update sensor state from external inputs."""
        self.state.update(sensor_state or {})
    
    def should_interrupt(self) -> bool:
        """Check if sensors require behavior interruption."""
        s = self.state
        return s.get('emergency_stop') or s.get('obstacle_close') or s.get('override_hold')

# -----------------------
# TEMI ROBOT 5-LAYER SUBSUMPTION ARCHITECTURE
# -----------------------

class LowBatteryLayer:
    """Layer 1: Low battery detection and emergency power management (HIGHEST PRIORITY)."""
    
    def __init__(self, critical_battery_level: float = 15.0, low_battery_level: float = 25.0):
        self.critical_battery_level = critical_battery_level
        self.low_battery_level = low_battery_level
        self.battery_status = {'level': 100.0, 'charging': False, 'critical': False}
        
    def update_battery_status(self, battery_level: float, charging: bool = False) -> dict:
        """Update battery status from Temi system."""
        self.battery_status = {
            'level': battery_level,
            'charging': charging,
            'critical': battery_level <= self.critical_battery_level,
            'low': battery_level <= self.low_battery_level
        }
        return self.battery_status
    
    def should_interrupt(self) -> bool:
        """Check if low battery requires immediate action."""
        return self.battery_status['critical'] and not self.battery_status['charging']
    
    def get_battery_behavior(self) -> str:
        """Return required battery behavior."""
        if self.battery_status['critical']:
            return 'EMERGENCY_CHARGE'
        elif self.battery_status['low']:
            return 'SEEK_CHARGING'
        else:
            return 'NORMAL'

class TemiSensorLayer:
    """Layer 2: Temi robot sensor readings and safety systems."""
    
    def __init__(self):
        self.sensor_data = {
            'collision_detected': False,
            'proximity_warning': False,
            'virtual_wall_detected': False,
            'mapping_active': False,
            'obstacle_distance': float('inf'),
            'temi_moving': False,
            'manual_control': False
        }
        
    def update_sensors(self, **sensor_readings) -> dict:
        """Update sensor readings from Temi robot systems."""
        self.sensor_data.update(sensor_readings)
        return self.sensor_data
    
    def should_interrupt(self) -> bool:
        """Check if sensors require immediate behavior interruption."""
        return (self.sensor_data['collision_detected'] or 
                self.sensor_data['virtual_wall_detected'] or
                self.sensor_data['manual_control'] or
                self.sensor_data['obstacle_distance'] < 0.3)  # 30cm safety distance
    
    def get_sensor_status(self) -> str:
        """Return current sensor status for behavior selection."""
        if self.sensor_data['collision_detected']:
            return 'COLLISION'
        elif self.sensor_data['virtual_wall_detected']:
            return 'VIRTUAL_WALL'
        elif self.sensor_data['manual_control']:
            return 'MANUAL_OVERRIDE'
        elif self.sensor_data['proximity_warning']:
            return 'PROXIMITY_WARNING'
        else:
            return 'CLEAR'

class ApproachingLayer:
    """Layer 3: Human approach and interaction management."""
    
    def __init__(self, approach_distance: float = 1.5, interaction_timeout: float = 30.0):
        self.approach_distance = approach_distance
        self.interaction_timeout = interaction_timeout
        self.interaction_state = {
            'person_identified': False,
            'approach_initiated': False,
            'interaction_active': False,
            'interaction_start_time': None,
            'target_distance': None,
            'help_requested': False
        }
        
    def update_interaction(self, person_detected: bool, estimated_distance: Optional[float], 
                          sustained_gaze: bool, timestamp: float) -> dict:
        """Update interaction state based on person detection and gaze."""
        
        # Check if person needs help (sustained gaze indicates need for assistance)
        if person_detected and sustained_gaze and estimated_distance:
            if not self.interaction_state['person_identified']:
                self.interaction_state['person_identified'] = True
                self.interaction_state['target_distance'] = estimated_distance
                print(f">>> PERSON IDENTIFIED for assistance at {estimated_distance:.1f}m")
            
            # Initiate approach if person is far enough to warrant approaching
            if (estimated_distance > self.approach_distance and 
                not self.interaction_state['approach_initiated']):
                self.interaction_state['approach_initiated'] = True
                self.interaction_state['interaction_start_time'] = timestamp
                print(f">>> APPROACHING: Moving closer to person at {estimated_distance:.1f}m")
            
            # Start interaction if close enough
            elif (estimated_distance <= self.approach_distance and 
                  self.interaction_state['approach_initiated']):
                self.interaction_state['interaction_active'] = True
                print(f">>> INTERACTION STARTED: Person reached at {estimated_distance:.1f}m")
        
        # Reset if person is no longer detected or timeout
        elif not person_detected or not sustained_gaze:
            if self.interaction_state['interaction_active']:
                print(">>> INTERACTION ENDED: Person no longer detected or looking away")
            self.interaction_state = {
                'person_identified': False,
                'approach_initiated': False,
                'interaction_active': False,
                'interaction_start_time': None,
                'target_distance': None,
                'help_requested': False
            }
        
        # Check for timeout
        if (self.interaction_state['interaction_start_time'] and 
            timestamp - self.interaction_state['interaction_start_time'] > self.interaction_timeout):
            print(">>> INTERACTION TIMEOUT: Returning to patrol")
            self.interaction_state['person_identified'] = False
            self.interaction_state['approach_initiated'] = False
            self.interaction_state['interaction_active'] = False
        
        return self.interaction_state
    
    def should_interrupt(self) -> bool:
        """Check if approaching layer should control behavior."""
        return (self.interaction_state['approach_initiated'] or 
                self.interaction_state['interaction_active'])
    
    def get_approach_behavior(self) -> str:
        """Return current approach behavior requirement."""
        if self.interaction_state['interaction_active']:
            return 'INTERACTING'
        elif self.interaction_state['approach_initiated']:
            return 'APPROACHING'
        elif self.interaction_state['person_identified']:
            return 'PERSON_IDENTIFIED'
        else:
            return 'PATROL'

class GazeDetectionLayer:
    """Layer 4: Combined gaze detection using MediaPipe+OpenCV and Roboflow."""
    
    def __init__(self):
        self.mediapipe_active = True
        self.roboflow_active = False  # Placeholder for future Roboflow integration
        self.gaze_data = {
            'mediapipe_result': None,
            'roboflow_result': None,
            'combined_result': None,
            'confidence_source': 'none'
        }
        
    def update_mediapipe_gaze(self, pose_data: Optional[dict], looking: bool) -> dict:
        """Update gaze detection from MediaPipe + OpenCV (your current model)."""
        self.gaze_data['mediapipe_result'] = {
            'active': True,
            'looking': looking,
            'pose_data': pose_data,
            'confidence': 0.8 if pose_data and pose_data.get('ok') else 0.0
        }
        return self.gaze_data['mediapipe_result']
    
    def update_roboflow_gaze(self, eye_detection_result: Optional[dict]) -> dict:
        """Update gaze detection from Roboflow eye detection (placeholder for future)."""
        # Placeholder for Roboflow eye detection integration
        self.gaze_data['roboflow_result'] = {
            'active': self.roboflow_active,
            'eye_detected': False,
            'gaze_direction': None,
            'confidence': 0.0
        }
        
        if eye_detection_result and self.roboflow_active:
            self.gaze_data['roboflow_result'].update(eye_detection_result)
        
        return self.gaze_data['roboflow_result']
    
    def get_combined_gaze_result(self) -> dict:
        """Combine results from both gaze detection methods."""
        mediapipe = self.gaze_data['mediapipe_result']
        roboflow = self.gaze_data['roboflow_result']
        
        # Priority: Roboflow (when available) > MediaPipe
        if roboflow and roboflow['active'] and roboflow['confidence'] > 0.6:
            combined_result = {
                'looking': roboflow.get('eye_detected', False),
                'method': 'roboflow',
                'confidence': roboflow['confidence'],
                'details': roboflow
            }
            self.gaze_data['confidence_source'] = 'roboflow'
        elif mediapipe and mediapipe['active'] and mediapipe['confidence'] > 0.5:
            combined_result = {
                'looking': mediapipe.get('looking', False),
                'method': 'mediapipe',
                'confidence': mediapipe['confidence'],
                'details': mediapipe
            }
            self.gaze_data['confidence_source'] = 'mediapipe'
        else:
            combined_result = {
                'looking': False,
                'method': 'none',
                'confidence': 0.0,
                'details': None
            }
            self.gaze_data['confidence_source'] = 'none'
        
        self.gaze_data['combined_result'] = combined_result
        return combined_result

class BaseBehaviorLayer:
    """Layer 5: Base robot behaviors when no higher priority layers are active."""
    
    def __init__(self):
        self.base_behaviors = ['PATROL', 'IDLE', 'CHARGE_SEEKING', 'STANDBY']
        self.current_base_behavior = 'IDLE'
        self.behavior_start_time = None
        
    def update_base_behavior(self, timestamp: float, battery_level: float, 
                           no_interactions: bool = True) -> str:
        """Update base behavior based on current conditions."""
        
        if battery_level < 30.0:
            new_behavior = 'CHARGE_SEEKING'
        elif no_interactions and battery_level > 50.0:
            new_behavior = 'PATROL'
        elif no_interactions:
            new_behavior = 'STANDBY'
        else:
            new_behavior = 'IDLE'
        
        if new_behavior != self.current_base_behavior:
            self.current_base_behavior = new_behavior
            self.behavior_start_time = timestamp
            print(f">>> BASE BEHAVIOR: Switching to {new_behavior}")
        
        return self.current_base_behavior

class BehaviorManager:
    """Original behavior manager - now integrated with enhanced subsumption layers."""
    
    def __init__(self, sustained_seconds: float = SUSTAINED_GAZE_SEC, sensor_gate: Optional[SensorGate] = None):
        self.sustained_seconds = sustained_seconds
        self.sensor_gate = sensor_gate or SensorGate()
        self.gaze_start_t = None
        self.approach_triggered = False

    def update(self, now: float, looking: bool) -> dict:
        """Update behavior state based on current conditions."""
        # Highest priority: sensor interrupts
        if self.sensor_gate.should_interrupt():
            self.gaze_start_t = None
            return {'state': 'INTERRUPTED', 'elapsed': 0.0}

        if looking:
            if self.gaze_start_t is None:
                self.gaze_start_t = now
            elapsed = now - self.gaze_start_t
            if (elapsed >= self.sustained_seconds) and not self.approach_triggered:
                self.trigger_behavior()
                self.approach_triggered = True
            return {'state': 'LOOKING', 'elapsed': elapsed}
        else:
            self.gaze_start_t = None
            # Reset trigger for repeated detection
            self.approach_triggered = False
            return {'state': 'NOT_LOOKING', 'elapsed': 0.0}

    def trigger_behavior(self):
        """Trigger behavior action (extend for robot commands)."""
        print(">>> BEHAVIOR TRIGGERED: Sustained gaze detected for {:.1f}s (TEMI 5-LAYER SUBSUMPTION)".format(self.sustained_seconds))

# -----------------------
# Main Subsumption Pipeline
# -----------------------

def main():
    """Main Temi 5-layer subsumption pipeline execution."""
    print("Attempting to open camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): 
        print("ERROR: Cannot open camera (index 0)")
        print("Trying camera index 1...")
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("ERROR: Cannot open any camera")
            exit(1)
    
    print("Camera opened successfully!")

    # Initialize original detection layers (for Layer 4 - Gaze Detection)
    person_detector = PersonDetector(YOLO_MODEL)
    face_estimator = FacePoseEstimator(MAX_NUM_FACES)
    gaze_classifier = GazeClassifier()
    proximity_gate = ProximityGate()
    sensor_gate = SensorGate()
    behavior_manager = BehaviorManager(sensor_gate=sensor_gate)

    # Initialize Temi 5-Layer Subsumption Architecture
    layer1_battery = LowBatteryLayer()
    layer2_sensors = TemiSensorLayer()
    layer3_approaching = ApproachingLayer()
    layer4_gaze = GazeDetectionLayer()
    layer5_base = BaseBehaviorLayer()

    print("TEMI 5-LAYER SUBSUMPTION ARCHITECTURE RUNNING")
    print("Layer 1: Low Battery (HIGHEST PRIORITY)")
    print("Layer 2: Temi Sensor Readings (collision, proximity, mapping, virtual walls)")
    print("Layer 3: Approaching Someone (interaction management)")
    print("Layer 4: Gaze Detection (MediaPipe+OpenCV + Roboflow)")
    print("Layer 5: Base Behavior (patrol, idle, standby)")
    print("Press 'q' to quit.")
    
    last_t = time.time()
    frame_count = 0
    
    # Simulation variables for Temi sensors (replace with actual Temi API calls)
    simulated_battery = 85.0
    simulated_battery_drain = 0.01  # Battery drains 0.01% per frame
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        now = time.time()
        frame_count += 1
        
        # Simulate battery drain (replace with actual Temi battery reading)
        simulated_battery -= simulated_battery_drain
        if simulated_battery < 0:
            simulated_battery = 0
        
        # -----------------------
        # LAYER 1: LOW BATTERY (HIGHEST PRIORITY)
        # -----------------------
        battery_status = layer1_battery.update_battery_status(simulated_battery, charging=False)
        battery_behavior = layer1_battery.get_battery_behavior()
        
        if layer1_battery.should_interrupt():
            # Battery emergency - override all other behaviors
            cv2.putText(frame, "EMERGENCY: LOW BATTERY - SEEKING CHARGE", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Battery: {simulated_battery:.1f}%", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow("Temi 5-Layer Subsumption", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        
        # -----------------------
        # LAYER 2: TEMI SENSORS
        # -----------------------
        # Update sensor readings (simulate for now - replace with actual Temi sensor API)
        sensor_readings = {
            'collision_detected': False,
            'proximity_warning': frame_count % 200 < 10,  # Simulate occasional proximity warning
            'virtual_wall_detected': False,
            'mapping_active': True,
            'obstacle_distance': 2.0 + (frame_count % 100) / 50.0,  # Simulate varying distance
            'temi_moving': False,
            'manual_control': False
        }
        layer2_sensors.update_sensors(**sensor_readings)
        sensor_status = layer2_sensors.get_sensor_status()
        
        if layer2_sensors.should_interrupt():
            # Sensor emergency - override lower priority behaviors
            cv2.putText(frame, f"SENSOR ALERT: {sensor_status}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            cv2.putText(frame, f"Obstacle: {sensor_readings['obstacle_distance']:.1f}m", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        
        # -----------------------
        # LAYER 4: GAZE DETECTION (Your MediaPipe + OpenCV model)
        # -----------------------
        predictions = []
        
        # Original person detection
        persons = person_detector.detect(frame)
        for (x1, y1, x2, y2, conf) in persons:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(frame, "Person", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Original face pose estimation
        pose = face_estimator.estimate(frame) if persons else None
        
        looking = False
        estimated_distance = None
        sustained_gaze = False
        
        if pose and pose.get('ok', False):
            pitch, yaw, roll = pose['pitch'], pose['yaw'], pose['roll']
            pts2d = pose['pts2d']
            face_kps = pose['face_kps']
            estimated_distance = pose.get('estimated_distance')

            # Draw face landmarks
            for pt in face_kps:
                x, y = int(pt['x']), int(pt['y'])
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

            # Calculate face bounding box
            xs = [pt['x'] for pt in face_kps]
            ys = [pt['y'] for pt in face_kps]
            x_min, x_max = int(min(xs)), int(max(xs))
            y_min, y_max = int(min(ys)), int(max(ys))
            box_for_prox = (x_min, y_min, x_max, y_max)

            # Draw head pose axes
            rvec, tvec = pose['rvec'], pose['tvec']
            cam_matrix, dist_coeffs = pose['cam_matrix'], pose['dist_coeffs']
            axis = np.float32([[80,0,0],[0,80,0],[0,0,80]])
            imgpts, _ = cv2.projectPoints(axis, rvec, tvec, cam_matrix, dist_coeffs)
            nose = tuple(pts2d[0].astype(int))
            for i, pt in enumerate(imgpts):
                pt = tuple(pt.ravel().astype(int))
                color = [(0,0,255),(0,255,0),(255,0,0)][i]
                cv2.line(frame, nose, pt, color, 2)

            # Display pose and distance
            distance_text = f"Dist: {estimated_distance:.1f}m" if estimated_distance and estimated_distance < 10 else "Dist: ??"
            cv2.putText(frame, f"Yaw {yaw:+.1f}, Pitch {pitch:+.1f}", (x_min, y_min-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)
            cv2.putText(frame, distance_text, (x_min, y_min-35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)

            # Gaze classification
            coarse_looking = gaze_classifier.is_looking(pitch, yaw)
            face_crop = frame[max(0,y_min):min(frame.shape[0],y_max), 
                             max(0,x_min):min(frame.shape[1],x_max)]
            looking = refine_gaze(coarse_looking, pitch, yaw, roll, face_crop, pts2d)

            # Update Layer 4: Gaze Detection
            layer4_gaze.update_mediapipe_gaze(pose, looking)
            layer4_gaze.update_roboflow_gaze(None)  # Placeholder for Roboflow
            gaze_result = layer4_gaze.get_combined_gaze_result()
            
            # Original behavior manager for sustained gaze detection
            bstate = behavior_manager.update(now, looking)
            sustained_gaze = bstate['state'] == 'LOOKING' and bstate['elapsed'] >= SUSTAINED_GAZE_SEC

            # Display gaze detection info
            status_color = (0,255,0) if looking else (0,0,255)
            cv2.putText(frame, f"Gaze: {gaze_result['method']} Conf:{gaze_result['confidence']:.2f}",
                        (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)

            # Collect prediction data (maintaining original format)
            face_dict = {
                'face': {
                    'x': float((x_min + x_max) / 2),
                    'y': float((y_min + y_max) / 2),
                    'width': float(x_max - x_min),
                    'height': float(y_max - y_min),
                    'confidence': float(persons[0][4]) if persons else 0.0,
                    'class': 'face',
                    'class_confidence': None,
                    'class_id': 0,
                    'tracker_id': None,
                    'detection_id': None,
                    'parent_id': None,
                    'landmarks': face_kps
                },
                'yaw': float(yaw),
                'pitch': float(pitch),
                'estimated_distance': float(estimated_distance) if estimated_distance else None,
                'gaze_method': gaze_result['method']
            }
            predictions.append(face_dict)

        # -----------------------
        # LAYER 3: APPROACHING SOMEONE
        # -----------------------
        interaction_state = layer3_approaching.update_interaction(
            person_detected=bool(persons and pose and pose.get('ok')), 
            estimated_distance=estimated_distance,
            sustained_gaze=sustained_gaze, 
            timestamp=now
        )
        approach_behavior = layer3_approaching.get_approach_behavior()
        
        if layer3_approaching.should_interrupt():
            cv2.putText(frame, f"INTERACTION: {approach_behavior}", (10, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
            if interaction_state['target_distance']:
                cv2.putText(frame, f"Target: {interaction_state['target_distance']:.1f}m", (10, 210),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1)

        # -----------------------
        # LAYER 5: BASE BEHAVIOR
        # -----------------------
        no_higher_priority = (not layer1_battery.should_interrupt() and 
                             not layer2_sensors.should_interrupt() and 
                             not layer3_approaching.should_interrupt())
        
        if no_higher_priority:
            base_behavior = layer5_base.update_base_behavior(now, simulated_battery, 
                                                           no_interactions=not bool(predictions))
            cv2.putText(frame, f"Base: {base_behavior}", (10, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)

        # -----------------------
        # DISPLAY LAYER STATUS
        # -----------------------
        # Layer status display
        layer_y_start = frame.shape[0] - 150
        cv2.putText(frame, "TEMI 5-LAYER STATUS:", (10, layer_y_start),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        cv2.putText(frame, f"1.Battery: {simulated_battery:.1f}% ({battery_behavior})", 
                    (10, layer_y_start + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 
                    (0,0,255) if layer1_battery.should_interrupt() else (255,255,255), 1)
        
        cv2.putText(frame, f"2.Sensors: {sensor_status}", 
                    (10, layer_y_start + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (0,165,255) if layer2_sensors.should_interrupt() else (255,255,255), 1)
        
        cv2.putText(frame, f"3.Approach: {approach_behavior}", 
                    (10, layer_y_start + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255,100,0) if layer3_approaching.should_interrupt() else (255,255,255), 1)
        
        gaze_method = gaze_result['method'] if 'gaze_result' in locals() else 'none'
        cv2.putText(frame, f"4.Gaze: {gaze_method} ({'ON' if looking else 'OFF'})", 
                    (10, layer_y_start + 85), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (0,255,0) if looking else (255,255,255), 1)
        
        cv2.putText(frame, f"5.Base: {layer5_base.current_base_behavior if no_higher_priority else 'SUPPRESSED'}", 
                    (10, layer_y_start + 105), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (200,200,200) if no_higher_priority else (100,100,100), 1)

        # Original output format for compatibility
        elapsed = now - last_t
        results = [{
            'predictions': predictions,
            'time': elapsed,
            'time_face_det': None,
            'time_gaze_det': None,
            'temi_layer_status': {
                'battery': battery_status,
                'sensors': sensor_readings,
                'interaction': interaction_state,
                'gaze': gaze_result if 'gaze_result' in locals() else None,
                'base_behavior': layer5_base.current_base_behavior
            }
        }]
        
        # Print results (maintaining original format)
        if predictions:
            print("\n# TEMI 5-LAYER SUBSUMPTION RESULTS:\n# Gaze detection result:\n", results)

        # Display FPS
        dt = now - last_t
        fps = 1.0/max(dt, 1e-6)
        last_t = now
        H = frame.shape[0]
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, H-30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

        cv2.imshow("Temi 5-Layer Subsumption", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()













# Next save










"""

source ~/venvs/TemiGaze/bin/activate



python temi_5layer_subsumption.py







gaze_offline.py - Subsumption Architecture Version
Converted from monolithic structure to subsumption-based gaze detection system.

Pipeline (every frame):
  YOLOv8 (person gate) -> MediaPipe FaceMesh -> solvePnP -> (yaw, pitch, roll)
  -> GazeClassifier (+ optional refine_gaze hook) -> BehaviorManager

Design:
  - Subsumption layers with priority handling
  - Modular components for easy extension
  - Configurable thresholds and parameters
  - Behavior triggers for sustained gaze detection
"""

import cv2
import numpy as np
import math
import time
from ultralytics import YOLO
import mediapipe as mp
from dataclasses import dataclass
from typing import List, Optional, Tuple

# -----------------------
# Configuration Constants
# -----------------------
# YOLO_MODEL = 'yolov8n-face.pt'
YOLO_MODEL = 'yolov8s.pt'
FACE_YOLO_MODEL = 'yolov8n-face.pt'  # Face-specific detection for distant subjects
MIN_PERSON_CONF = 0.3  # Lowered for better distant detection
MIN_FACE_CONF = 0.1   # Very low confidence for distant face detection
MAX_NUM_FACES = 1
MIN_DETECTION_CONFIDENCE = 0.3  # Lowered for distant faces

# Gaze direction settings
FLIP_GAZE_DIRECTION = False  # Set to True if the arrow still points the wrong way

# Head-pose thresholds for gaze detection
YAW_THRESH_DEG = 20.0  # -20 to +20 degrees
PITCH_THRESH_DEG = 15.0  # Will be overridden by range check
PITCH_MIN_DEG = -180.0  # Minimum pitch for looking (extended to handle -175°)
PITCH_MAX_DEG = 180.0   # Maximum pitch for looking (full range)

# Distance-adaptive proximity detection
FACE_AREA_FRAC_CLOSE = 0.04     # Close range (< 1.5m): 4% of frame
FACE_AREA_FRAC_MEDIUM = 0.015   # Medium range (1.5-3m): 1.5% of frame  
FACE_AREA_FRAC_FAR = 0.008      # Far range (3-5m): 0.8% of frame
MIN_FACE_PIXELS = 30            # Minimum face size in pixels (width or height)

# Distance estimation parameters
CLOSE_DISTANCE_THRESHOLD = 1.5   # meters
MEDIUM_DISTANCE_THRESHOLD = 3.0  # meters
REFERENCE_FACE_SIZE_MM = 150     # Average face width in mm
CAMERA_FOCAL_LENGTH_MM = 4.0     # Typical webcam focal length

# Sustained gaze time to trigger behavior
SUSTAINED_GAZE_SEC = 3.0

# 3D model points for solvePnP (same as original)
MODEL_POINTS = np.array([
    (   0.0,    0.0,    0.0),    # Nose tip        (landmark 1)
    (   0.0, -63.6, -12.5),      # Chin            (landmark 199)
    (-43.3,  32.7, -26.0),       # Left eye left   (landmark 33)
    ( 43.3,  32.7, -26.0),       # Right eye right (landmark 263)
    (-28.9, -28.9, -24.1),       # Mouth left      (landmark 61)
    ( 28.9, -28.9, -24.1)        # Mouth right     (landmark 291)
], dtype=np.float64)
LANDMARK_IDS = [1, 199, 33, 263, 61, 291]

# -----------------------
# Utility Functions
# -----------------------

def rotation_to_euler(rvec: np.ndarray) -> Tuple[float, float, float]:
    """Convert OpenCV rotation vector to Euler angles in degrees."""
    R, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2,1], R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.degrees(x), np.degrees(y), np.degrees(z)

def face_box_from_pts(pts2d: np.ndarray) -> Tuple[int,int,int,int]:
    """Calculate bounding box (x1,y1,x2,y2) from a set of 2D points."""
    xs = pts2d[:,0]
    ys = pts2d[:,1]
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

def face_area_fraction(box, frame_shape) -> float:
    """Calculate face box area as fraction of total frame area."""
    x1, y1, x2, y2 = box
    area = max(0, x2-x1) * max(0, y2-y1)
    H, W = frame_shape[:2]
    return area / float(W*H + 1e-6)

def estimate_distance_from_face(face_width_pixels: float, frame_width: int) -> float:
    """
    Estimate distance to face using pinhole camera model.
    Returns distance in meters.
    """
    if face_width_pixels <= 0:
        return float('inf')
    
    # Convert camera specs to appropriate units
    # Typical webcam: focal length ~4mm, sensor width ~4.8mm
    sensor_width_mm = 4.8
    focal_length_pixels = (CAMERA_FOCAL_LENGTH_MM * frame_width) / sensor_width_mm
    
    # Distance = (real_object_size * focal_length_pixels) / object_size_pixels
    distance_mm = (REFERENCE_FACE_SIZE_MM * focal_length_pixels) / face_width_pixels
    return distance_mm / 1000.0  # Convert to meters

def get_adaptive_proximity_threshold(distance_meters: float) -> float:
    """
    Return appropriate proximity threshold based on estimated distance.
    """
    if distance_meters < CLOSE_DISTANCE_THRESHOLD:
        return FACE_AREA_FRAC_CLOSE
    elif distance_meters < MEDIUM_DISTANCE_THRESHOLD:
        return FACE_AREA_FRAC_MEDIUM
    else:
        return FACE_AREA_FRAC_FAR

def preprocess_for_distant_detection(frame, estimated_distance: Optional[float] = None, 
                                     face_region: Optional[Tuple[int,int,int,int]] = None, 
                                     enhancement_level: int = 1) -> np.ndarray:
    """
    Preprocess frame to improve distant face detection.
    Returns enhanced frame for MediaPipe processing.
    
    enhancement_level: 1=light, 2=medium, 3=aggressive
    """
    enhanced = frame.copy()
    
    # Apply different enhancement levels based on distance and detection attempts
    if enhancement_level == 1 and (estimated_distance is None or estimated_distance < MEDIUM_DISTANCE_THRESHOLD):
        return enhanced  # Minimal processing for close range
    
    # Enhancement parameters based on level
    if enhancement_level == 1:  # Light enhancement
        alpha, beta = 1.2, 8
        blur_reduce = False
    elif enhancement_level == 2:  # Medium enhancement  
        alpha, beta = 1.4, 15
        blur_reduce = True
    else:  # Aggressive enhancement for very distant detection
        alpha, beta = 1.6, 25
        blur_reduce = True
    
    # 1. Increase contrast and brightness
    enhanced = cv2.convertScaleAbs(enhanced, alpha=alpha, beta=beta)
    
    # 2. Reduce noise for distant detection
    if blur_reduce:
        enhanced = cv2.bilateralFilter(enhanced, 5, 80, 80)
    
    # 3. Enhanced sharpening filter
    if enhancement_level >= 2:
        kernel = np.array([[-1,-1,-1,-1,-1],
                          [-1, 2, 2, 2,-1],
                          [-1, 2, 8, 2,-1], 
                          [-1, 2, 2, 2,-1],
                          [-1,-1,-1,-1,-1]]) / 8.0
    else:
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
    enhanced = cv2.filter2D(enhanced, -1, kernel)
    
    # 4. Region-specific enhancement if face region is known
    if face_region is not None and enhancement_level >= 2:
        x1, y1, x2, y2 = face_region
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(enhanced.shape[1], x2), min(enhanced.shape[0], y2)
        
        if x2 > x1 and y2 > y1:
            face_crop = enhanced[y1:y2, x1:x2]
            # Apply histogram equalization to face region
            face_crop_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            face_crop_eq = cv2.equalizeHist(face_crop_gray)
            face_crop_enhanced = cv2.cvtColor(face_crop_eq, cv2.COLOR_GRAY2BGR)
            # Blend enhanced face back into frame
            enhanced[y1:y2, x1:x2] = cv2.addWeighted(face_crop, 0.6, face_crop_enhanced, 0.4, 0)
    
    # 5. Gamma correction for very distant detection
    if enhancement_level >= 3:
        gamma = 0.8
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        enhanced = cv2.LUT(enhanced, table)
    
    return enhanced

def get_adaptive_detection_confidence(distance_meters: float) -> float:
    """
    Return appropriate MediaPipe detection confidence based on distance.
    """
    if distance_meters < CLOSE_DISTANCE_THRESHOLD:
        return 0.5  # Standard confidence for close range
    elif distance_meters < MEDIUM_DISTANCE_THRESHOLD:
        return 0.3  # Lower confidence for medium range
    else:
        return 0.2  # Very low confidence for far range

def estimate_landmarks_from_face_box(face_box: Tuple[int,int,int,int], frame_shape) -> Tuple[np.ndarray, List[dict]]:
    """
    Estimate facial landmarks based on face bounding box when MediaPipe fails.
    Returns (pts2d, face_kps) in same format as MediaPipe.
    """
    x1, y1, x2, y2 = face_box
    face_width = x2 - x1
    face_height = y2 - y1
    
    # Estimate landmark positions based on typical face proportions
    # These are approximate but allow pose estimation to work
    landmarks_relative = {
        'nose_tip': (0.5, 0.6),      # Center, slightly down (landmark 1)
        'chin': (0.5, 0.95),         # Center bottom (landmark 199) 
        'left_eye_left': (0.25, 0.4), # Left side, upper (landmark 33)
        'right_eye_right': (0.75, 0.4), # Right side, upper (landmark 263)
        'mouth_left': (0.35, 0.8),   # Left of center, lower (landmark 61)
        'mouth_right': (0.65, 0.8)   # Right of center, lower (landmark 291)
    }
    
    pts2d = []
    face_kps = []
    
    for landmark_name, (rel_x, rel_y) in landmarks_relative.items():
        x = int(x1 + rel_x * face_width)
        y = int(y1 + rel_y * face_height)
        
        # Clamp to frame boundaries
        x = max(0, min(frame_shape[1] - 1, x))
        y = max(0, min(frame_shape[0] - 1, y))
        
        pts2d.append([x, y])
        face_kps.append({'x': float(x), 'y': float(y)})
    
    return np.array(pts2d, dtype=np.float64), face_kps

# -----------------------
# Subsumption Architecture Classes
# -----------------------

class FaceDetector:
    """YOLOv8-based face detection for distant subjects when MediaPipe fails."""
    
    def __init__(self, model_path: str = FACE_YOLO_MODEL, min_conf: float = MIN_FACE_CONF):
        try:
            self.model = YOLO(model_path)
            self.available = True
            print(f"Face detector loaded: {model_path}")
        except Exception as e:
            print(f"Warning: Face detector not available ({e}). Using person detector only.")
            self.model = None
            self.available = False
        self.base_min_conf = min_conf
    
    def detect_faces(self, frame, adaptive_conf: bool = True) -> List[Tuple[int,int,int,int,float]]:
        """Return face boxes [(x1,y1,x2,y2,conf), ...] with very low confidence thresholds."""
        if not self.available:
            return []
            
        res = self.model(frame, verbose=False)[0]
        out = []
        frame_area = frame.shape[0] * frame.shape[1]
        
        for det in res.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = det
            # Face detection typically has class 0 for faces
            face_area = (x2 - x1) * (y2 - y1)
            face_area_fraction = face_area / frame_area
            
            # Very aggressive confidence thresholds for distant faces
            if adaptive_conf:
                if face_area_fraction > 0.01:  # Medium face
                    min_conf = self.base_min_conf
                elif face_area_fraction > 0.003:  # Small face  
                    min_conf = max(0.05, self.base_min_conf - 0.15)
                else:  # Very small face (distant)
                    min_conf = max(0.02, self.base_min_conf - 0.20)
            else:
                min_conf = self.base_min_conf
            
            if conf >= min_conf:
                out.append((int(x1), int(y1), int(x2), int(y2), float(conf)))
        return out

class PersonDetector:
    """YOLOv8-based person detection layer with distance-adaptive confidence."""
    
    def __init__(self, model_path: str = YOLO_MODEL, min_conf: float = MIN_PERSON_CONF):
        self.model = YOLO(model_path)
        self.base_min_conf = min_conf

    def detect(self, frame, adaptive_conf: bool = True) -> List[Tuple[int,int,int,int,float]]:
        """Return person boxes [(x1,y1,x2,y2,conf), ...] with optional adaptive confidence."""
        res = self.model(frame, verbose=False)[0]
        out = []
        frame_area = frame.shape[0] * frame.shape[1]
        
        for det in res.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = det
            if int(cls) == 0:  # class 0 = person
                # Calculate person box size for distance estimation
                person_area = (x2 - x1) * (y2 - y1)
                person_area_fraction = person_area / frame_area
                
                # Adaptive confidence: lower threshold for smaller (distant) persons
                if adaptive_conf:
                    if person_area_fraction > 0.15:  # Large person (close)
                        min_conf = self.base_min_conf
                    elif person_area_fraction > 0.05:  # Medium person
                        min_conf = max(0.2, self.base_min_conf - 0.2)
                    else:  # Small person (distant)
                        min_conf = max(0.15, self.base_min_conf - 0.35)
                else:
                    min_conf = self.base_min_conf
                
                if conf >= min_conf:
                    out.append((int(x1), int(y1), int(x2), int(y2), float(conf)))
        return out

class FacePoseEstimator:
    """MediaPipe FaceMesh + solvePnP head pose estimation layer with adaptive confidence and fallback detection."""
    
    def __init__(self, max_faces: int = MAX_NUM_FACES):
        self.max_faces = max_faces
        self.mp_face_mesh = mp.solutions.face_mesh
        # Initialize with default confidence, will be updated adaptively
        self.mesh = None
        self.current_confidence = MIN_DETECTION_CONFIDENCE
        self._init_mesh(MIN_DETECTION_CONFIDENCE)
        # Add face detector as fallback
        self.face_detector = FaceDetector()
        self.last_successful_distance = None

    def _init_mesh(self, detection_confidence: float):
        """Initialize MediaPipe mesh with given confidence."""
        self.current_confidence = detection_confidence
        self.mesh = self.mp_face_mesh.FaceMesh(
            refine_landmarks=True,
            static_image_mode=False,
            max_num_faces=self.max_faces,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=max(0.1, detection_confidence - 0.2)  # Tracking can be lower
        )

    def estimate(self, frame, estimated_distance: Optional[float] = None, 
                person_boxes: Optional[List] = None) -> Optional[dict]:
        """Return dict with pitch, yaw, roll, pts2d, face_kps, distance and ok status.
        
        Uses cascade approach: MediaPipe -> Face YOLO -> Estimated landmarks
        """
        H, W = frame.shape[:2]
        
        # Try multiple detection approaches in order of preference
        result = None
        detection_method = "failed"
        
        # Approach 1: Standard MediaPipe detection
        result = self._try_mediapipe_detection(frame, estimated_distance)
        if result and result.get('ok'):
            detection_method = "mediapipe_standard"
            return {**result, 'detection_method': detection_method}
        
        # Approach 2: Enhanced MediaPipe with preprocessing
        for enhancement_level in [1, 2, 3]:
            face_region = None
            if person_boxes:
                # Use person box as hint for face region
                face_region = person_boxes[0][:4] if person_boxes else None
            
            result = self._try_mediapipe_detection(
                frame, estimated_distance, enhancement_level, face_region
            )
            if result and result.get('ok'):
                detection_method = f"mediapipe_enhanced_L{enhancement_level}"
                return {**result, 'detection_method': detection_method}
        
        # Approach 3: Face YOLO detection with landmark estimation
        if self.face_detector.available:
            face_boxes = self.face_detector.detect_faces(frame, adaptive_conf=True)
            if face_boxes:
                # Use the most confident face detection
                best_face = max(face_boxes, key=lambda x: x[4])  # Sort by confidence
                face_box = best_face[:4]
                face_conf = best_face[4]
                
                # Estimate landmarks from face box
                pts2d, face_kps = estimate_landmarks_from_face_box(face_box, (H, W))
                
                # Calculate face width for distance estimation
                face_width_pixels = max([pt[0] for pt in pts2d]) - min([pt[0] for pt in pts2d])
                
                if estimated_distance is None:
                    estimated_distance = estimate_distance_from_face(face_width_pixels, W)
                
                # Attempt pose estimation with estimated landmarks
                result = self._estimate_pose_from_landmarks(pts2d, face_kps, estimated_distance, W, H)
                if result and result.get('ok'):
                    result['face_detection_confidence'] = face_conf
                    result['detection_method'] = "face_yolo_estimated"
                    return result
        
        # Approach 4: Fallback to person box landmark estimation
        if person_boxes:
            person_box = person_boxes[0][:4]  # Use first person detection
            # Estimate face region within person box (upper 40% typically)
            px1, py1, px2, py2 = person_box
            face_height = int(0.4 * (py2 - py1))
            estimated_face_box = (px1, py1, px2, py1 + face_height)
            
            pts2d, face_kps = estimate_landmarks_from_face_box(estimated_face_box, (H, W))
            face_width_pixels = max([pt[0] for pt in pts2d]) - min([pt[0] for pt in pts2d])
            
            if estimated_distance is None:
                estimated_distance = estimate_distance_from_face(face_width_pixels, W)
            
            result = self._estimate_pose_from_landmarks(pts2d, face_kps, estimated_distance, W, H)
            if result and result.get('ok'):
                result['detection_method'] = "person_box_estimated"
                return result
        
        return None  # All approaches failed
    
    def _try_mediapipe_detection(self, frame, estimated_distance: Optional[float] = None,
                                enhancement_level: int = 0, face_region: Optional[Tuple] = None) -> Optional[dict]:
        """Try MediaPipe detection with optional preprocessing."""
        H, W = frame.shape[:2]
        
        # Adapt confidence based on distance
        if estimated_distance is not None:
            new_confidence = get_adaptive_detection_confidence(estimated_distance)
            if abs(new_confidence - self.current_confidence) > 0.05:
                self._init_mesh(new_confidence)
        
        # Apply preprocessing if requested
        if enhancement_level > 0:
            processed_frame = preprocess_for_distant_detection(
                frame, estimated_distance, face_region, enhancement_level
            )
        else:
            processed_frame = frame
        
        rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        results = self.mesh.process(rgb)
        
        if not results.multi_face_landmarks:
            return None

        lm = results.multi_face_landmarks[0].landmark
        pts2d = []
        face_kps = []
        
        for idx in LANDMARK_IDS:
            x = int(lm[idx].x * W)
            y = int(lm[idx].y * H)
            pts2d.append([x, y])
            face_kps.append({'x': float(x), 'y': float(y)})
            
        pts2d = np.array(pts2d, dtype=np.float64)
        face_width_pixels = max([pt[0] for pt in pts2d]) - min([pt[0] for pt in pts2d])
        
        if estimated_distance is None:
            estimated_distance = estimate_distance_from_face(face_width_pixels, W)
        
        return self._estimate_pose_from_landmarks(pts2d, face_kps, estimated_distance, W, H)
    
    def _estimate_pose_from_landmarks(self, pts2d: np.ndarray, face_kps: List[dict],
                                    estimated_distance: float, W: int, H: int) -> Optional[dict]:
        """Estimate head pose from landmarks using solvePnP."""
        # Camera intrinsics (pinhole approximation)
        focal = W
        cam_matrix = np.array([[focal, 0, W/2],
                               [0, focal, H/2],
                               [0,     0,   1]], dtype=np.float64)
        dist_coeffs = np.zeros((4,1), dtype=np.float64)

        ok, rvec, tvec = cv2.solvePnP(
            MODEL_POINTS, pts2d, cam_matrix, dist_coeffs, 
            flags=cv2.SOLVEPNP_ITERATIVE)
            
        if not ok:
            return {'ok': False}

        pitch, yaw, roll = rotation_to_euler(rvec)
        face_width_pixels = max([pt[0] for pt in pts2d]) - min([pt[0] for pt in pts2d])
        
        return {
            'ok': True,
            'pitch': float(pitch),
            'yaw': float(yaw),
            'roll': float(roll),
            'pts2d': pts2d,
            'face_kps': face_kps,
            'rvec': rvec,
            'tvec': tvec,
            'cam_matrix': cam_matrix,
            'dist_coeffs': dist_coeffs,
            'estimated_distance': float(estimated_distance),
            'face_width_pixels': float(face_width_pixels),
            'detection_confidence': float(self.current_confidence)
        }

@dataclass
class GazeClassifier:
    """Gaze classification layer based on head pose thresholds."""
    yaw_thresh: float = YAW_THRESH_DEG
    pitch_min: float = PITCH_MIN_DEG
    pitch_max: float = PITCH_MAX_DEG

    def is_looking(self, pitch: float, yaw: float) -> bool:
        """Determine if person is looking based on head pose ranges."""
        yaw_ok = abs(yaw) <= self.yaw_thresh  # -20 to +20 degrees
        
        # Handle pitch with full range since -175° should be valid for looking at camera
        # Since you're getting -175° when looking at camera, we'll accept the full range
        pitch_ok = True  # Accept all pitch values for now
        
        return yaw_ok and pitch_ok

def refine_gaze(is_looking: bool, pitch: float, yaw: float, roll: float, 
                face_crop, landmarks_2d: np.ndarray) -> bool:
    """
    Accuracy refinement hook (placeholder for future enhancements).
    Later: integrate Roboflow eye-gaze, iris tracking, temporal smoothing, etc.
    For now: return the coarse decision unchanged.
    """
    # TODO: integrate eye model here and override is_looking if needed
    return is_looking

class ProximityGate:
    """Proximity detection layer - adaptive thresholds for monitoring (not gating detection)."""
    
    def __init__(self, min_frac: float = FACE_AREA_FRAC_CLOSE):
        self.default_min_frac = float(min_frac)

    def is_close(self, box, frame_shape, estimated_distance: Optional[float] = None) -> Tuple[bool, float, float]:
        """
        Check if face meets proximity thresholds for monitoring purposes.
        NOTE: This no longer gates gaze detection - used for display/debugging only.
        Returns (is_close, area_fraction, threshold_used)
        """
        area_frac = face_area_fraction(box, frame_shape)
        
        # Use adaptive threshold if distance is estimated
        if estimated_distance is not None:
            threshold = get_adaptive_proximity_threshold(estimated_distance)
        else:
            threshold = self.default_min_frac
        
        # Also check minimum pixel size to avoid tiny detections
        x1, y1, x2, y2 = box
        face_width = max(0, x2-x1)
        face_height = max(0, y2-y1)
        min_size_ok = face_width >= MIN_FACE_PIXELS and face_height >= MIN_FACE_PIXELS
        
        is_close = area_frac >= threshold and min_size_ok
        return is_close, area_frac, threshold

class SensorGate:
    """Sensor interrupt layer (highest priority in subsumption)."""
    
    def __init__(self):
        self.state = {
            'emergency_stop': False,
            'obstacle_close': False,
            'override_hold': False
        }
    
    def update_from_sensors(self, sensor_state: dict):
        """Update sensor state from external inputs."""
        self.state.update(sensor_state or {})
    
    def should_interrupt(self) -> bool:
        """Check if sensors require behavior interruption."""
        s = self.state
        return s.get('emergency_stop') or s.get('obstacle_close') or s.get('override_hold')

# -----------------------
# TEMI ROBOT 5-LAYER SUBSUMPTION ARCHITECTURE
# -----------------------

class LowBatteryLayer:
    """Layer 1: Low battery detection and emergency power management (HIGHEST PRIORITY)."""
    
    def __init__(self, critical_battery_level: float = 15.0, low_battery_level: float = 25.0):
        self.critical_battery_level = critical_battery_level
        self.low_battery_level = low_battery_level
        self.battery_status = {'level': 100.0, 'charging': False, 'critical': False}
        
    def update_battery_status(self, battery_level: float, charging: bool = False) -> dict:
        """Update battery status from Temi system."""
        self.battery_status = {
            'level': battery_level,
            'charging': charging,
            'critical': battery_level <= self.critical_battery_level,
            'low': battery_level <= self.low_battery_level
        }
        return self.battery_status
    
    def should_interrupt(self) -> bool:
        """Check if low battery requires immediate action."""
        return self.battery_status['critical'] and not self.battery_status['charging']
    
    def get_battery_behavior(self) -> str:
        """Return required battery behavior."""
        if self.battery_status['critical']:
            return 'EMERGENCY_CHARGE'
        elif self.battery_status['low']:
            return 'SEEK_CHARGING'
        else:
            return 'NORMAL'

class TemiSensorLayer:
    """Layer 2: Temi robot sensor readings and safety systems."""
    
    def __init__(self):
        self.sensor_data = {
            'collision_detected': False,
            'proximity_warning': False,
            'virtual_wall_detected': False,
            'mapping_active': False,
            'obstacle_distance': float('inf'),
            'temi_moving': False,
            'manual_control': False
        }
        
    def update_sensors(self, **sensor_readings) -> dict:
        """Update sensor readings from Temi robot systems."""
        self.sensor_data.update(sensor_readings)
        return self.sensor_data
    
    def should_interrupt(self) -> bool:
        """Check if sensors require immediate behavior interruption."""
        return (self.sensor_data['collision_detected'] or 
                self.sensor_data['virtual_wall_detected'] or
                self.sensor_data['manual_control'] or
                self.sensor_data['obstacle_distance'] < 0.3)  # 30cm safety distance
    
    def get_sensor_status(self) -> str:
        """Return current sensor status for behavior selection."""
        if self.sensor_data['collision_detected']:
            return 'COLLISION'
        elif self.sensor_data['virtual_wall_detected']:
            return 'VIRTUAL_WALL'
        elif self.sensor_data['manual_control']:
            return 'MANUAL_OVERRIDE'
        elif self.sensor_data['proximity_warning']:
            return 'PROXIMITY_WARNING'
        else:
            return 'CLEAR'

class ApproachingLayer:
    """Layer 3: Human approach and interaction management."""
    
    def __init__(self, approach_distance: float = 1.5, interaction_timeout: float = 30.0):
        self.approach_distance = approach_distance
        self.interaction_timeout = interaction_timeout
        self.interaction_state = {
            'person_identified': False,
            'approach_initiated': False,
            'interaction_active': False,
            'interaction_start_time': None,
            'target_distance': None,
            'help_requested': False
        }
        
    def update_interaction(self, person_detected: bool, estimated_distance: Optional[float], 
                          sustained_gaze: bool, timestamp: float) -> dict:
        """Update interaction state based on person detection and gaze."""
        
        # Check if person needs help (sustained gaze indicates need for assistance)
        if person_detected and sustained_gaze and estimated_distance:
            if not self.interaction_state['person_identified']:
                self.interaction_state['person_identified'] = True
                self.interaction_state['target_distance'] = estimated_distance
                print(f">>> PERSON IDENTIFIED for assistance at {estimated_distance:.1f}m")
            
            # Initiate approach if person is far enough to warrant approaching
            if (estimated_distance > self.approach_distance and 
                not self.interaction_state['approach_initiated']):
                self.interaction_state['approach_initiated'] = True
                self.interaction_state['interaction_start_time'] = timestamp
                print(f">>> APPROACHING: Moving closer to person at {estimated_distance:.1f}m")
            
            # Start interaction if close enough
            elif (estimated_distance <= self.approach_distance and 
                  self.interaction_state['approach_initiated']):
                self.interaction_state['interaction_active'] = True
                print(f">>> INTERACTION STARTED: Person reached at {estimated_distance:.1f}m")
        
        # Reset if person is no longer detected or timeout
        elif not person_detected or not sustained_gaze:
            if self.interaction_state['interaction_active']:
                print(">>> INTERACTION ENDED: Person no longer detected or looking away")
            self.interaction_state = {
                'person_identified': False,
                'approach_initiated': False,
                'interaction_active': False,
                'interaction_start_time': None,
                'target_distance': None,
                'help_requested': False
            }
        
        # Check for timeout
        if (self.interaction_state['interaction_start_time'] and 
            timestamp - self.interaction_state['interaction_start_time'] > self.interaction_timeout):
            print(">>> INTERACTION TIMEOUT: Returning to patrol")
            self.interaction_state['person_identified'] = False
            self.interaction_state['approach_initiated'] = False
            self.interaction_state['interaction_active'] = False
        
        return self.interaction_state
    
    def should_interrupt(self) -> bool:
        """Check if approaching layer should control behavior."""
        return (self.interaction_state['approach_initiated'] or 
                self.interaction_state['interaction_active'])
    
    def get_approach_behavior(self) -> str:
        """Return current approach behavior requirement."""
        if self.interaction_state['interaction_active']:
            return 'INTERACTING'
        elif self.interaction_state['approach_initiated']:
            return 'APPROACHING'
        elif self.interaction_state['person_identified']:
            return 'PERSON_IDENTIFIED'
        else:
            return 'PATROL'

class GazeDetectionLayer:
    """Layer 4: Combined gaze detection using MediaPipe+OpenCV and Roboflow."""
    
    def __init__(self):
        self.mediapipe_active = True
        self.roboflow_active = False  # Placeholder for future Roboflow integration
        self.gaze_data = {
            'mediapipe_result': None,
            'roboflow_result': None,
            'combined_result': None,
            'confidence_source': 'none'
        }
        
    def update_mediapipe_gaze(self, pose_data: Optional[dict], looking: bool) -> dict:
        """Update gaze detection from MediaPipe + OpenCV (your current model)."""
        self.gaze_data['mediapipe_result'] = {
            'active': True,
            'looking': looking,
            'pose_data': pose_data,
            'confidence': 0.8 if pose_data and pose_data.get('ok') else 0.0
        }
        return self.gaze_data['mediapipe_result']
    
    def update_roboflow_gaze(self, eye_detection_result: Optional[dict]) -> dict:
        """Update gaze detection from Roboflow eye detection (placeholder for future)."""
        # Placeholder for Roboflow eye detection integration
        self.gaze_data['roboflow_result'] = {
            'active': self.roboflow_active,
            'eye_detected': False,
            'gaze_direction': None,
            'confidence': 0.0
        }
        
        if eye_detection_result and self.roboflow_active:
            self.gaze_data['roboflow_result'].update(eye_detection_result)
        
        return self.gaze_data['roboflow_result']
    
    def get_combined_gaze_result(self) -> dict:
        """Combine results from both gaze detection methods."""
        mediapipe = self.gaze_data['mediapipe_result']
        roboflow = self.gaze_data['roboflow_result']
        
        # Priority: Roboflow (when available) > MediaPipe
        if roboflow and roboflow['active'] and roboflow['confidence'] > 0.6:
            combined_result = {
                'looking': roboflow.get('eye_detected', False),
                'method': 'roboflow',
                'confidence': roboflow['confidence'],
                'details': roboflow
            }
            self.gaze_data['confidence_source'] = 'roboflow'
        elif mediapipe and mediapipe['active'] and mediapipe['confidence'] > 0.5:
            combined_result = {
                'looking': mediapipe.get('looking', False),
                'method': 'mediapipe',
                'confidence': mediapipe['confidence'],
                'details': mediapipe
            }
            self.gaze_data['confidence_source'] = 'mediapipe'
        else:
            combined_result = {
                'looking': False,
                'method': 'none',
                'confidence': 0.0,
                'details': None
            }
            self.gaze_data['confidence_source'] = 'none'
        
        self.gaze_data['combined_result'] = combined_result
        return combined_result

class BaseBehaviorLayer:
    """Layer 5: Base robot behaviors when no higher priority layers are active."""
    
    def __init__(self):
        self.base_behaviors = ['PATROL', 'IDLE', 'CHARGE_SEEKING', 'STANDBY']
        self.current_base_behavior = 'IDLE'
        self.behavior_start_time = None
        
    def update_base_behavior(self, timestamp: float, battery_level: float, 
                           no_interactions: bool = True) -> str:
        """Update base behavior based on current conditions."""
        
        if battery_level < 30.0:
            new_behavior = 'CHARGE_SEEKING'
        elif no_interactions and battery_level > 50.0:
            new_behavior = 'PATROL'
        elif no_interactions:
            new_behavior = 'STANDBY'
        else:
            new_behavior = 'IDLE'
        
        if new_behavior != self.current_base_behavior:
            self.current_base_behavior = new_behavior
            self.behavior_start_time = timestamp
            print(f">>> BASE BEHAVIOR: Switching to {new_behavior}")
        
        return self.current_base_behavior

class BehaviorManager:
    """Original behavior manager - now integrated with enhanced subsumption layers."""
    
    def __init__(self, sustained_seconds: float = SUSTAINED_GAZE_SEC, sensor_gate: Optional[SensorGate] = None):
        self.sustained_seconds = sustained_seconds
        self.sensor_gate = sensor_gate or SensorGate()
        self.gaze_start_t = None
        self.approach_triggered = False

    def update(self, now: float, looking: bool) -> dict:
        """Update behavior state based on current conditions."""
        # Highest priority: sensor interrupts
        if self.sensor_gate.should_interrupt():
            self.gaze_start_t = None
            return {'state': 'INTERRUPTED', 'elapsed': 0.0}

        if looking:
            if self.gaze_start_t is None:
                self.gaze_start_t = now
            elapsed = now - self.gaze_start_t
            if (elapsed >= self.sustained_seconds) and not self.approach_triggered:
                self.trigger_behavior()
                self.approach_triggered = True
            return {'state': 'LOOKING', 'elapsed': elapsed}
        else:
            self.gaze_start_t = None
            # Reset trigger for repeated detection
            self.approach_triggered = False
            return {'state': 'NOT_LOOKING', 'elapsed': 0.0}

    def trigger_behavior(self):
        """Trigger behavior action (extend for robot commands)."""
        print(">>> BEHAVIOR TRIGGERED: Sustained gaze detected for {:.1f}s (TEMI 5-LAYER SUBSUMPTION)".format(self.sustained_seconds))

# -----------------------
# Main Subsumption Pipeline
# -----------------------

def main():
    """Main Temi 5-layer subsumption pipeline execution."""
    print("Attempting to open camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): 
        print("ERROR: Cannot open camera (index 0)")
        print("Trying camera index 1...")
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("ERROR: Cannot open any camera")
            exit(1)
    
    print("Camera opened successfully!")

    # Initialize original detection layers (for Layer 4 - Gaze Detection)
    person_detector = PersonDetector(YOLO_MODEL)
    face_estimator = FacePoseEstimator(MAX_NUM_FACES)
    gaze_classifier = GazeClassifier()
    proximity_gate = ProximityGate()
    sensor_gate = SensorGate()
    behavior_manager = BehaviorManager(sensor_gate=sensor_gate)

    # Initialize Temi 5-Layer Subsumption Architecture
    layer1_battery = LowBatteryLayer()
    layer2_sensors = TemiSensorLayer()
    layer3_approaching = ApproachingLayer()
    layer4_gaze = GazeDetectionLayer()
    layer5_base = BaseBehaviorLayer()

    print("TEMI 5-LAYER SUBSUMPTION ARCHITECTURE RUNNING")
    print("Layer 1: Low Battery (HIGHEST PRIORITY)")
    print("Layer 2: Temi Sensor Readings (collision, proximity, mapping, virtual walls)")
    print("Layer 3: Approaching Someone (interaction management)")
    print("Layer 4: Gaze Detection (MediaPipe+OpenCV + Roboflow)")
    print("Layer 5: Base Behavior (patrol, idle, standby)")
    print("Press 'q' to quit.")
    
    last_t = time.time()
    frame_count = 0
    
    # Simulation variables for Temi sensors (replace with actual Temi API calls)
    simulated_battery = 85.0
    simulated_battery_drain = 0.01  # Battery drains 0.01% per frame
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        now = time.time()
        frame_count += 1
        
        # Simulate battery drain (replace with actual Temi battery reading)
        simulated_battery -= simulated_battery_drain
        if simulated_battery < 0:
            simulated_battery = 0
        
        # -----------------------
        # LAYER 1: LOW BATTERY (HIGHEST PRIORITY)
        # -----------------------
        battery_status = layer1_battery.update_battery_status(simulated_battery, charging=False)
        battery_behavior = layer1_battery.get_battery_behavior()
        
        if layer1_battery.should_interrupt():
            # Battery emergency - override all other behaviors
            cv2.putText(frame, "EMERGENCY: LOW BATTERY - SEEKING CHARGE", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Battery: {simulated_battery:.1f}%", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow("Temi 5-Layer Subsumption", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        
        # -----------------------
        # LAYER 2: TEMI SENSORS
        # -----------------------
        # Update sensor readings (simulate for now - replace with actual Temi sensor API)
        sensor_readings = {
            'collision_detected': False,
            'proximity_warning': frame_count % 200 < 10,  # Simulate occasional proximity warning
            'virtual_wall_detected': False,
            'mapping_active': True,
            'obstacle_distance': 2.0 + (frame_count % 100) / 50.0,  # Simulate varying distance
            'temi_moving': False,
            'manual_control': False
        }
        layer2_sensors.update_sensors(**sensor_readings)
        sensor_status = layer2_sensors.get_sensor_status()
        
        if layer2_sensors.should_interrupt():
            # Sensor emergency - override lower priority behaviors
            cv2.putText(frame, f"SENSOR ALERT: {sensor_status}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            cv2.putText(frame, f"Obstacle: {sensor_readings['obstacle_distance']:.1f}m", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        
        # -----------------------
        # LAYER 4: GAZE DETECTION (Your MediaPipe + OpenCV model)
        # -----------------------
        predictions = []
        
        # Original person detection
        persons = person_detector.detect(frame)
        for (x1, y1, x2, y2, conf) in persons:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(frame, "Person", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Enhanced cascade face pose estimation
        pose = face_estimator.estimate(frame, person_boxes=persons) if persons else None
        
        looking = False
        estimated_distance = None
        sustained_gaze = False
        
        if pose and pose.get('ok', False):
            pitch, yaw, roll = pose['pitch'], pose['yaw'], pose['roll']
            pts2d = pose['pts2d']
            face_kps = pose['face_kps']
            estimated_distance = pose.get('estimated_distance')

            # Draw face landmarks
            for pt in face_kps:
                x, y = int(pt['x']), int(pt['y'])
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

            # Calculate face bounding box
            xs = [pt['x'] for pt in face_kps]
            ys = [pt['y'] for pt in face_kps]
            x_min, x_max = int(min(xs)), int(max(xs))
            y_min, y_max = int(min(ys)), int(max(ys))
            box_for_prox = (x_min, y_min, x_max, y_max)

            # Draw head pose axes
            rvec, tvec = pose['rvec'], pose['tvec']
            cam_matrix, dist_coeffs = pose['cam_matrix'], pose['dist_coeffs']
            axis = np.float32([[80,0,0],[0,80,0],[0,0,80]])
            imgpts, _ = cv2.projectPoints(axis, rvec, tvec, cam_matrix, dist_coeffs)
            nose = tuple(pts2d[0].astype(int))
            for i, pt in enumerate(imgpts):
                pt = tuple(pt.ravel().astype(int))
                color = [(0,0,255),(0,255,0),(255,0,0)][i]
                cv2.line(frame, nose, pt, color, 2)

            # Display pose, distance, and detection method
            distance_text = f"Dist: {estimated_distance:.1f}m" if estimated_distance and estimated_distance < 10 else "Dist: ??"
            detection_method = pose.get('detection_method', 'unknown')[:15]  # Truncate for display
            cv2.putText(frame, f"Yaw {yaw:+.1f}, Pitch {pitch:+.1f}", (x_min, y_min-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)
            cv2.putText(frame, distance_text, (x_min, y_min-35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)
            cv2.putText(frame, f"Method: {detection_method}", (x_min, y_min-60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

            # Gaze classification
            coarse_looking = gaze_classifier.is_looking(pitch, yaw)
            face_crop = frame[max(0,y_min):min(frame.shape[0],y_max), 
                             max(0,x_min):min(frame.shape[1],x_max)]
            looking = refine_gaze(coarse_looking, pitch, yaw, roll, face_crop, pts2d)

            # Update Layer 4: Gaze Detection
            layer4_gaze.update_mediapipe_gaze(pose, looking)
            layer4_gaze.update_roboflow_gaze(None)  # Placeholder for Roboflow
            gaze_result = layer4_gaze.get_combined_gaze_result()
            
            # Original behavior manager for sustained gaze detection
            bstate = behavior_manager.update(now, looking)
            sustained_gaze = bstate['state'] == 'LOOKING' and bstate['elapsed'] >= SUSTAINED_GAZE_SEC

            # Display gaze detection info
            status_color = (0,255,0) if looking else (0,0,255)
            cv2.putText(frame, f"Gaze: {gaze_result['method']} Conf:{gaze_result['confidence']:.2f}",
                        (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)

            # Collect prediction data (maintaining original format)
            face_dict = {
                'face': {
                    'x': float((x_min + x_max) / 2),
                    'y': float((y_min + y_max) / 2),
                    'width': float(x_max - x_min),
                    'height': float(y_max - y_min),
                    'confidence': float(persons[0][4]) if persons else 0.0,
                    'class': 'face',
                    'class_confidence': None,
                    'class_id': 0,
                    'tracker_id': None,
                    'detection_id': None,
                    'parent_id': None,
                    'landmarks': face_kps
                },
                'yaw': float(yaw),
                'pitch': float(pitch),
                'estimated_distance': float(estimated_distance) if estimated_distance else None,
                'gaze_method': gaze_result['method'],
                'detection_method': pose.get('detection_method', 'unknown')
            }
            predictions.append(face_dict)

        # -----------------------
        # LAYER 3: APPROACHING SOMEONE
        # -----------------------
        interaction_state = layer3_approaching.update_interaction(
            person_detected=bool(persons and pose and pose.get('ok')), 
            estimated_distance=estimated_distance,
            sustained_gaze=sustained_gaze, 
            timestamp=now
        )
        approach_behavior = layer3_approaching.get_approach_behavior()
        
        if layer3_approaching.should_interrupt():
            cv2.putText(frame, f"INTERACTION: {approach_behavior}", (10, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
            if interaction_state['target_distance']:
                cv2.putText(frame, f"Target: {interaction_state['target_distance']:.1f}m", (10, 210),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1)

        # -----------------------
        # LAYER 5: BASE BEHAVIOR
        # -----------------------
        no_higher_priority = (not layer1_battery.should_interrupt() and 
                             not layer2_sensors.should_interrupt() and 
                             not layer3_approaching.should_interrupt())
        
        if no_higher_priority:
            base_behavior = layer5_base.update_base_behavior(now, simulated_battery, 
                                                           no_interactions=not bool(predictions))
            cv2.putText(frame, f"Base: {base_behavior}", (10, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)

        # -----------------------
        # DISPLAY LAYER STATUS
        # -----------------------
        # Layer status display
        layer_y_start = frame.shape[0] - 150
        cv2.putText(frame, "TEMI 5-LAYER STATUS:", (10, layer_y_start),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        cv2.putText(frame, f"1.Battery: {simulated_battery:.1f}% ({battery_behavior})", 
                    (10, layer_y_start + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 
                    (0,0,255) if layer1_battery.should_interrupt() else (255,255,255), 1)
        
        cv2.putText(frame, f"2.Sensors: {sensor_status}", 
                    (10, layer_y_start + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (0,165,255) if layer2_sensors.should_interrupt() else (255,255,255), 1)
        
        cv2.putText(frame, f"3.Approach: {approach_behavior}", 
                    (10, layer_y_start + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255,100,0) if layer3_approaching.should_interrupt() else (255,255,255), 1)
        
        gaze_method = gaze_result['method'] if 'gaze_result' in locals() else 'none'
        cv2.putText(frame, f"4.Gaze: {gaze_method} ({'ON' if looking else 'OFF'})", 
                    (10, layer_y_start + 85), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (0,255,0) if looking else (255,255,255), 1)
        
        cv2.putText(frame, f"5.Base: {layer5_base.current_base_behavior if no_higher_priority else 'SUPPRESSED'}", 
                    (10, layer_y_start + 105), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (200,200,200) if no_higher_priority else (100,100,100), 1)

        # Original output format for compatibility
        elapsed = now - last_t
        results = [{
            'predictions': predictions,
            'time': elapsed,
            'time_face_det': None,
            'time_gaze_det': None,
            'temi_layer_status': {
                'battery': battery_status,
                'sensors': sensor_readings,
                'interaction': interaction_state,
                'gaze': gaze_result if 'gaze_result' in locals() else None,
                'base_behavior': layer5_base.current_base_behavior
            }
        }]
        
        # Print results (maintaining original format)
        if predictions:
            print("\n# TEMI 5-LAYER SUBSUMPTION RESULTS:\n# Gaze detection result:\n", results)

        # Display FPS
        dt = now - last_t
        fps = 1.0/max(dt, 1e-6)
        last_t = now
        H = frame.shape[0]
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, H-30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

        cv2.imshow("Temi 5-Layer Subsumption", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


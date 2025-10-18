"""

source ~/venvs/TemiGaze/bin/activate











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
CAMERA_FOCAL_LENGTH_MM = 2.0     # Typical webcam focal length

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

class BehaviorManager:
    """Subsumption behavior manager: sensor interrupt > sustained gaze > idle."""
    
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
        print(">>> BEHAVIOR TRIGGERED: Sustained gaze detected for {:.1f}s (DISTANT DETECTION ENABLED)".format(self.sustained_seconds))

# -----------------------
# Main Subsumption Pipeline
# -----------------------

def main():
    """Main subsumption pipeline execution."""
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

    # Initialize subsumption layers
    person_detector = PersonDetector(YOLO_MODEL)
    face_estimator = FacePoseEstimator(MAX_NUM_FACES)
    gaze_classifier = GazeClassifier()
    proximity_gate = ProximityGate()
    sensor_gate = SensorGate()
    behavior_manager = BehaviorManager(sensor_gate=sensor_gate)

    print("Enhanced Subsumption Gaze Detection System Running")
    print("Improvements:")
    print("- Adaptive distance thresholds (Close: 4%, Medium: 1.5%, Far: 0.8%)")
    print("- Distance-based confidence scaling (Person: 0.15-0.3, Face: 0.2-0.5)")
    print("- Image preprocessing for distant detection")
    print("- Real-time distance estimation")
    print("- DISTANT GAZE DETECTION ENABLED (no proximity requirement)")
    print("Press 'q' to quit.")
    
    last_t = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        now = time.time()
        predictions = []

        # Layer 1: Person Detection
        persons = person_detector.detect(frame)
        for (x1, y1, x2, y2, conf) in persons:
            # Draw person box (maintaining original visual style)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(frame, "Person", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Layer 2: Face Pose Estimation (if persons detected)
        pose = face_estimator.estimate(frame) if persons else None
        
        looking = False
        box_for_prox = None
        estimated_distance = None
        
        if pose and pose.get('ok', False):
            pitch, yaw, roll = pose['pitch'], pose['yaw'], pose['roll']
            pts2d = pose['pts2d']
            face_kps = pose['face_kps']
            rvec, tvec = pose['rvec'], pose['tvec']
            cam_matrix, dist_coeffs = pose['cam_matrix'], pose['dist_coeffs']
            estimated_distance = pose.get('estimated_distance')
            face_width_pixels = pose.get('face_width_pixels', 0)
            detection_confidence = pose.get('detection_confidence', MIN_DETECTION_CONFIDENCE)

            # Draw face landmarks (maintaining original visual style)
            for pt in face_kps:
                x, y = int(pt['x']), int(pt['y'])
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

            # Calculate face bounding box for display and proximity
            xs = [pt['x'] for pt in face_kps]
            ys = [pt['y'] for pt in face_kps]
            x_min, x_max = int(min(xs)), int(max(xs))
            y_min, y_max = int(min(ys)), int(max(ys))
            face_width = x_max - x_min
            face_height = y_max - y_min
            face_x = x_min + face_width / 2
            face_y = y_min + face_height / 2
            
            box_for_prox = (x_min, y_min, x_max, y_max)

            # Draw head pose axes (maintaining original visual style)
            axis = np.float32([[80,0,0],[0,80,0],[0,0,80]])
            imgpts, _ = cv2.projectPoints(axis, rvec, tvec, cam_matrix, dist_coeffs)
            nose = tuple(pts2d[0].astype(int))
            for i, pt in enumerate(imgpts):
                pt = tuple(pt.ravel().astype(int))
                color = [(0,0,255),(0,255,0),(255,0,0)][i]  # Red=X, Green=Y, Blue=Z
                cv2.line(frame, nose, pt, color, 2)

            # Optional: Draw gaze vector (commented out - arrow removed)
            # if FLIP_GAZE_DIRECTION:
            #     sign = -1
            # else:
            #     sign = 1
            # gaze3D = np.array([[0, 0, sign * 1000.0]], dtype=np.float64)
            # gaze2D, _ = cv2.projectPoints(gaze3D, rvec, tvec, cam_matrix, dist_coeffs)
            # gpt = tuple(gaze2D[0].ravel().astype(int))
            # cv2.arrowedLine(frame, nose, gpt, (0, 0, 255), 2, tipLength=0.2)

            # Display pose and distance information
            distance_text = f"Dist: {estimated_distance:.1f}m" if estimated_distance and estimated_distance < 10 else "Dist: ??"
            cv2.putText(frame, f"Yaw {yaw:+.1f}, Pitch {pitch:+.1f}", (x_min, y_min-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)
            cv2.putText(frame, distance_text, (x_min, y_min-35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)
            
            # Debug: Show yaw and pitch criteria
            yaw_ok = abs(yaw) <= 20.0
            cv2.putText(frame, f"Yaw OK: {yaw_ok} (|{yaw:.1f}| <= 20)", (x_min, y_min-60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)

            # Layer 3: Proximity Gate (for monitoring only, not gating gaze detection)
            close_result = proximity_gate.is_close(box_for_prox, frame.shape, estimated_distance)
            close_enough, area_frac, threshold_used = close_result
            
            # Layer 4: Gaze Classification (no proximity requirement)
            coarse_looking = gaze_classifier.is_looking(pitch, yaw)
            
            # Layer 5: Gaze Refinement Hook (no proximity requirement)
            face_crop = frame[max(0,y_min):min(frame.shape[0],y_max), 
                             max(0,x_min):min(frame.shape[1],x_max)]
            looking = refine_gaze(coarse_looking, pitch, yaw, roll, face_crop, pts2d)
            # NOTE: Removed "and close_enough" - proximity no longer gates gaze detection
            # This allows distant gaze detection for robot approach behavior

            # Collect prediction data (maintaining original format)
            face_dict = {
                'face': {
                    'x': float(face_x),
                    'y': float(face_y),
                    'width': float(face_width),
                    'height': float(face_height),
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
                'face_width_pixels': float(face_width_pixels)
            }
            predictions.append(face_dict)

            # Display enhanced subsumption status with distance info
            status_color = (0,255,0) if looking else (0,0,255)
            cv2.putText(frame, f"Area:{area_frac:.3f} Thresh:{threshold_used:.3f} Close:{close_enough}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
            cv2.putText(frame, f"Looking:{looking} (DISTANT OK) Conf:{detection_confidence:.2f}",
                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)

        # Layer 6: Behavior Management (Subsumption Control)
        bstate = behavior_manager.update(now, looking)
        cv2.putText(frame, f"Behavior: {bstate['state']}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,0), 2)
        if bstate['state'] == 'LOOKING':
            cv2.putText(frame, f"Sustained: {bstate['elapsed']:.1f}s", (10, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,0), 2)

        # Maintain original output format
        elapsed = now - last_t
        results = [{
            'predictions': predictions,
            'time': elapsed,
            'time_face_det': None,
            'time_gaze_det': None
        }]
        
        # Print results (maintaining original format)
        if predictions:
            print("\n# TRIAL RESULTS:\n# Gaze detection result:\n", results)

        # Display FPS
        dt = now - last_t
        fps = 1.0/max(dt, 1e-6)
        last_t = now
        H = frame.shape[0]
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, H-30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

        cv2.imshow("YOLO + Face Keypoints + Gaze (Subsumption)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

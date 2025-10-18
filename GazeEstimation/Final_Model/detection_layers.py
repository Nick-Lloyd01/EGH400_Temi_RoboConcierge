"""
Detection Layers Module for Temi 5-Layer Subsumption Architecture

This module contains the core detection classes:
- FaceDetector: YOLO-based face detection (primary)
- PersonDetector: YOLO-based person detection (backup)
- FacePoseEstimator: Face-first pose estimation pipeline
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import List, Optional, Tuple
from ultralytics import YOLO

from config import *
from utils import *

# =============================================================================
# FACE DETECTION CLASS (PRIMARY DETECTION METHOD)
# =============================================================================

class FaceDetector:
    """
    YOLOv8n-face.pt based face detection - primary detection method.
    
    This class handles face detection using a specialized YOLO model trained
    specifically for face detection. It provides ultra-low confidence thresholds
    for distant face detection and adaptive confidence based on face size.
    """
    
    def __init__(self, model_path: str = FACE_YOLO_MODEL, min_conf: float = MIN_FACE_CONF):
        """Initialize the face detector with YOLO model."""
        try:
            self.model = YOLO(model_path)
            self.available = True
            print(f"✓ Face detector loaded successfully: {model_path}")
        except Exception as e:
            print(f"⚠ Warning: Face detector not available ({e}). Using fallback methods.")
            self.model = None
            self.available = False
            
        self.base_min_conf = min_conf
        self.max_faces = MAX_NUM_FACES
    
    def detect_faces(self, frame: np.ndarray, adaptive_conf: bool = True) -> List[Tuple[int,int,int,int,float]]:
        """Detect faces in frame using YOLO face model with adaptive confidence."""
        if not self.available:
            return []
            
        # Run YOLO face detection
        results = self.model(frame, verbose=False)[0]
        detections = []
        frame_area = frame.shape[0] * frame.shape[1]
        
        for detection in results.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = detection
            
            # Calculate face area for adaptive confidence
            face_area = (x2 - x1) * (y2 - y1)
            face_area_fraction = face_area / frame_area
            
            # Determine confidence threshold based on face size
            if adaptive_conf:
                min_conf = self._get_adaptive_confidence(face_area_fraction)
            else:
                min_conf = self.base_min_conf
            
            # Check minimum pixel size to avoid tiny false positives
            face_width = x2 - x1
            face_height = y2 - y1
            min_size_ok = (face_width >= MIN_FACE_PIXELS and 
                          face_height >= MIN_FACE_PIXELS)
            
            # Accept detection if it meets confidence and size requirements
            if conf >= min_conf and min_size_ok:
                detections.append((int(x1), int(y1), int(x2), int(y2), float(conf)))
        
        # Sort by confidence (highest first) and limit to max_faces
        detections.sort(key=lambda x: x[4], reverse=True)
        return detections[:self.max_faces]
    
    def _get_adaptive_confidence(self, face_area_fraction: float) -> float:
        """Calculate adaptive confidence threshold based on face size."""
        if face_area_fraction > 0.015:          # Medium-large face
            return self.base_min_conf
        elif face_area_fraction > 0.008:        # Small face
            return max(0.03, self.base_min_conf - 0.1)
        elif face_area_fraction > 0.003:        # Very small face
            return max(0.02, self.base_min_conf - 0.15)
        else:                                   # Extremely small face (distant)
            return max(0.01, self.base_min_conf - 0.2)

# =============================================================================
# PERSON DETECTION CLASS (BACKUP DETECTION METHOD)
# =============================================================================

class PersonDetector:
    """YOLOv8-based person detection layer with distance-adaptive confidence."""
    
    def __init__(self, model_path: str = YOLO_MODEL, min_conf: float = MIN_PERSON_CONF):
        """Initialize person detector with YOLO model."""
        self.model = YOLO(model_path)
        self.base_min_conf = min_conf
        print(f"✓ Person detector loaded: {model_path}")

    def detect(self, frame: np.ndarray, adaptive_conf: bool = True) -> List[Tuple[int,int,int,int,float]]:
        """Detect persons in frame with optional adaptive confidence."""
        results = self.model(frame, verbose=False)[0]
        detections = []
        frame_area = frame.shape[0] * frame.shape[1]
        
        for detection in results.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = detection
            
            # Check if detection is a person (class 0 in COCO dataset)
            if int(cls) == 0:
                # Calculate person size for adaptive confidence
                person_area = (x2 - x1) * (y2 - y1)
                person_area_fraction = person_area / frame_area
                
                # Adaptive confidence: lower threshold for smaller (distant) persons
                if adaptive_conf:
                    min_conf = self._get_adaptive_confidence(person_area_fraction)
                else:
                    min_conf = self.base_min_conf
                
                if conf >= min_conf:
                    detections.append((int(x1), int(y1), int(x2), int(y2), float(conf)))
                    
        return detections
    
    def _get_adaptive_confidence(self, person_area_fraction: float) -> float:
        """Calculate adaptive confidence threshold based on person size."""
        if person_area_fraction > 0.15:         # Large person (close)
            return self.base_min_conf
        elif person_area_fraction > 0.05:       # Medium person
            return max(0.2, self.base_min_conf - 0.2)
        else:                                   # Small person (distant)
            return max(0.15, self.base_min_conf - 0.35)

# =============================================================================
# FACE POSE ESTIMATION CLASS (MAIN PROCESSING PIPELINE)
# =============================================================================

class FacePoseEstimator:
    """Face-first pose estimation pipeline: YOLO face detection to MediaPipe landmarks to pose estimation."""
    
    def __init__(self, max_faces: int = MAX_NUM_FACES):
        """Initialize face pose estimator with MediaPipe and YOLO detectors."""
        self.max_faces = max_faces
        self.mp_face_mesh = mp.solutions.face_mesh
        
        # Initialize MediaPipe with default confidence
        self.mesh = None
        self.current_confidence = MIN_DETECTION_CONFIDENCE
        self._init_mesh(MIN_DETECTION_CONFIDENCE)
        
        # Initialize detection components
        self.face_detector = FaceDetector()
        self.person_detector = PersonDetector()
        
        print("✓ Face pose estimator initialized")
    
    def _init_mesh(self, detection_confidence: float):
        """Initialize MediaPipe FaceMesh with specified confidence."""
        self.current_confidence = detection_confidence
        self.mesh = self.mp_face_mesh.FaceMesh(
            refine_landmarks=True,
            static_image_mode=False,
            max_num_faces=self.max_faces,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=max(0.1, detection_confidence - 0.2)
        )
    
    def estimate(self, frame: np.ndarray, 
                estimated_distance: Optional[float] = None) -> Optional[dict]:
        """Main estimation method using face-first cascade approach."""
        H, W = frame.shape[:2]
        
        # Step 1: Primary face detection using YOLO
        face_boxes = self.face_detector.detect_faces(frame, adaptive_conf=True)
        
        if face_boxes:
            return self._process_detected_face(frame, face_boxes[0], estimated_distance)
        
        # Step 2: Fallback to person detection
        if ENABLE_DEBUG_PRINTS:
            print("No faces detected, trying person detection fallback")
            
        person_boxes = self.person_detector.detect(frame, adaptive_conf=True)
        if person_boxes:
            return self._process_person_fallback(frame, person_boxes[0], estimated_distance)
        
        if ENABLE_DEBUG_PRINTS:
            print("All detection methods failed")
        return None
    
    def _process_detected_face(self, frame: np.ndarray, 
                              face_detection: Tuple[int,int,int,int,float],
                              estimated_distance: Optional[float]) -> Optional[dict]:
        """Process a detected face through the MediaPipe to geometric fallback pipeline."""
        face_box = face_detection[:4]
        face_conf = face_detection[4]
        
        if ENABLE_DEBUG_PRINTS:
            print(f"Processing face: confidence={face_conf:.3f}, box={face_box}")
        
        # Try MediaPipe landmarks with progressive enhancement
        for enhancement_level in [0, 1, 2, 3]:  # 0 = no enhancement, 3 = maximum
            result = self._try_mediapipe_on_face_region(
                frame, face_box, estimated_distance, enhancement_level
            )
            
            if result and result.get('ok'):
                # Validate frontal face (reject back-of-head detections)
                if REQUIRE_FRONTAL_FACE:
                    yaw = abs(result.get('yaw', 0))
                    if yaw > MAX_YAW_FOR_FRONTAL:
                        if ENABLE_DEBUG_PRINTS:
                            print(f"Rejected back-of-head detection: yaw={yaw:.1f}° > {MAX_YAW_FOR_FRONTAL}°")
                        continue  # Try next enhancement level or fail
                
                # Success with MediaPipe and frontal face validated
                result['face_detection_confidence'] = face_conf
                result['face_box'] = face_box
                method_suffix = f"_enh{enhancement_level}" if enhancement_level > 0 else ""
                result['detection_method'] = f"face_yolo_mediapipe{method_suffix}"
                return result
        
        # MediaPipe failed or all results rejected, use geometric landmark estimation
        if ENABLE_DEBUG_PRINTS:
            print("MediaPipe failed on face region, using geometric estimation")
            
        result = self._estimate_pose_with_geometric_landmarks(
            frame, face_box, face_conf, estimated_distance, "face_yolo_geometric"
        )
        
        # Validate geometric result as well
        if result and REQUIRE_FRONTAL_FACE:
            yaw = abs(result.get('yaw', 0))
            if yaw > MAX_YAW_FOR_FRONTAL:
                if ENABLE_DEBUG_PRINTS:
                    print(f"Rejected geometric back-of-head: yaw={yaw:.1f}° > {MAX_YAW_FOR_FRONTAL}°")
                return None
        
        return result
    
    def _process_person_fallback(self, frame: np.ndarray,
                               person_detection: Tuple[int,int,int,int,float],
                               estimated_distance: Optional[float]) -> Optional[dict]:
        """Process person detection as fallback when no faces are found."""
        person_box = person_detection[:4]
        px1, py1, px2, py2 = person_box
        
        # Estimate face region within person box (upper 40%)
        face_height = int(0.4 * (py2 - py1))
        estimated_face_box = (px1, py1, px2, py1 + face_height)
        
        # Try MediaPipe on estimated face region
        result = self._try_mediapipe_on_face_region(
            frame, estimated_face_box, estimated_distance, enhancement_level=2
        )
        
        if result and result.get('ok'):
            result['detection_method'] = "person_box_mediapipe"
            return result
        
        # Final fallback: geometric estimation from person box
        return self._estimate_pose_with_geometric_landmarks(
            frame, estimated_face_box, person_detection[4], 
            estimated_distance, "person_box_geometric"
        )
    
    def _try_mediapipe_on_face_region(self, frame: np.ndarray, 
                                     face_box: Tuple[int,int,int,int],
                                     estimated_distance: Optional[float] = None,
                                     enhancement_level: int = 0) -> Optional[dict]:
        """Attempt MediaPipe landmark detection on a specific face region."""
        H, W = frame.shape[:2]
        x1, y1, x2, y2 = face_box
        
        # Expand face region for better landmark detection
        margin_x = int(FACE_REGION_MARGIN * (x2 - x1))
        margin_y = int(FACE_REGION_MARGIN * (y2 - y1))
        
        x1_exp = max(0, x1 - margin_x)
        y1_exp = max(0, y1 - margin_y)
        x2_exp = min(W, x2 + margin_x)
        y2_exp = min(H, y2 + margin_y)
        
        # Extract face region
        face_region = frame[y1_exp:y2_exp, x1_exp:x2_exp]
        
        if face_region.size == 0:
            return None
        
        # Adapt MediaPipe confidence for face region processing
        if estimated_distance is not None:
            new_confidence = get_adaptive_detection_confidence(estimated_distance)
        else:
            new_confidence = 0.1  # Lower confidence since we already have face detection
        
        if abs(new_confidence - self.current_confidence) > 0.05:
            self._init_mesh(new_confidence)
        
        # Apply preprocessing if requested
        if enhancement_level > 0:
            face_region = preprocess_for_distant_detection(
                face_region, estimated_distance, None, enhancement_level
            )
        
        # Run MediaPipe on face region
        rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
        results = self.mesh.process(rgb)
        
        if not results.multi_face_landmarks:
            return None

        # Extract landmarks and convert back to full frame coordinates
        lm = results.multi_face_landmarks[0].landmark
        pts2d = []
        face_kps = []
        
        region_h, region_w = face_region.shape[:2]
        
        for idx in LANDMARK_IDS:
            # Convert from face region coordinates to full frame coordinates
            x_region = lm[idx].x * region_w
            y_region = lm[idx].y * region_h
            
            x_full = int(x1_exp + x_region)
            y_full = int(y1_exp + y_region)
            
            # Clamp to frame boundaries
            x_full = max(0, min(W-1, x_full))
            y_full = max(0, min(H-1, y_full))
            
            pts2d.append([x_full, y_full])
            face_kps.append({'x': float(x_full), 'y': float(y_full)})
            
        pts2d = np.array(pts2d, dtype=np.float64)
        
        # Estimate distance if not provided
        face_width_pixels = max([pt[0] for pt in pts2d]) - min([pt[0] for pt in pts2d])
        if estimated_distance is None:
            estimated_distance = estimate_distance_from_face(face_width_pixels, W)
        
        return self._estimate_pose_from_landmarks(pts2d, face_kps, estimated_distance, W, H)
    
    def _estimate_pose_with_geometric_landmarks(self, frame: np.ndarray,
                                              face_box: Tuple[int,int,int,int],
                                              detection_conf: float,
                                              estimated_distance: Optional[float],
                                              method_name: str) -> Optional[dict]:
        """Estimate pose using geometrically estimated landmarks."""
        H, W = frame.shape[:2]
        
        # Generate geometric landmarks
        pts2d, face_kps = estimate_landmarks_from_face_box(face_box, (H, W))
        face_width_pixels = max([pt[0] for pt in pts2d]) - min([pt[0] for pt in pts2d])
        
        if estimated_distance is None:
            estimated_distance = estimate_distance_from_face(face_width_pixels, W)
        
        result = self._estimate_pose_from_landmarks(pts2d, face_kps, estimated_distance, W, H)
        if result and result.get('ok'):
            result['face_detection_confidence'] = detection_conf
            result['face_box'] = face_box
            result['detection_method'] = method_name
            return result
        
        return None
    
    def _estimate_pose_from_landmarks(self, pts2d: np.ndarray, face_kps: List[dict],
                                    estimated_distance: float, W: int, H: int) -> Optional[dict]:
        """Estimate head pose from facial landmarks using solvePnP."""
        # Set up camera model (pinhole camera approximation)
        focal = W  # Simple focal length approximation
        cam_matrix = np.array([[focal, 0, W/2],
                               [0, focal, H/2],
                               [0,     0,   1]], dtype=np.float64)
        dist_coeffs = np.zeros((4,1), dtype=np.float64)

        # Solve for pose using Perspective-n-Point algorithm
        ok, rvec, tvec = cv2.solvePnP(
            np.array(MODEL_POINTS, dtype=np.float64), pts2d, 
            cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
            
        if not ok:
            return {'ok': False}

        # Convert rotation vector to Euler angles
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
"""
Utility Functions Module for Temi 5-Layer Subsumption Architecture

This module contains utility functions for:
- Geometric calculations (rotation, distance estimation)
- Image preprocessing and enhancement
- Coordinate transformations
- Face landmark estimation
"""

import cv2
import numpy as np
import math
from typing import Tuple, Optional, List
from config import *

# =============================================================================
# GEOMETRIC UTILITY FUNCTIONS
# =============================================================================

def rotation_to_euler(rvec: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert OpenCV rotation vector to Euler angles in degrees.
    
    Args:
        rvec: OpenCV rotation vector from solvePnP
        
    Returns:
        Tuple of (pitch, yaw, roll) in degrees
        
    Note:
        - Pitch: Up/down head movement (nodding)
        - Yaw: Left/right head movement (shaking head "no")  
        - Roll: Tilting head to shoulder
    """
    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    
    # Extract Euler angles from rotation matrix
    sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-6
    
    if not singular:
        # Non-singular case - standard extraction
        x = math.atan2(R[2,1], R[2,2])      # Pitch (rotation around X-axis)
        y = math.atan2(-R[2,0], sy)         # Yaw (rotation around Y-axis)
        z = math.atan2(R[1,0], R[0,0])      # Roll (rotation around Z-axis)
    else:
        # Singular case - gimbal lock situation
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
        
    # Convert from radians to degrees for easier interpretation
    return np.degrees(x), np.degrees(y), np.degrees(z)

def face_box_from_pts(pts2d: np.ndarray) -> Tuple[int,int,int,int]:
    """
    Calculate bounding box coordinates from a set of 2D points.
    
    Args:
        pts2d: Array of 2D points (N x 2)
        
    Returns:
        Tuple of (x1, y1, x2, y2) representing bounding box corners
    """
    xs = pts2d[:,0]
    ys = pts2d[:,1]
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

def face_area_fraction(box: Tuple[int,int,int,int], frame_shape: Tuple[int,int]) -> float:
    """
    Calculate face bounding box area as fraction of total frame area.
    
    Args:
        box: Face bounding box (x1, y1, x2, y2)
        frame_shape: Frame dimensions (height, width)
        
    Returns:
        Face area as fraction of total frame area (0.0 to 1.0)
    """
    x1, y1, x2, y2 = box
    face_area = max(0, x2-x1) * max(0, y2-y1)
    H, W = frame_shape[:2]
    total_area = W * H
    return face_area / float(total_area + 1e-6)  # Add small epsilon to avoid division by zero

# =============================================================================
# DISTANCE ESTIMATION FUNCTIONS
# =============================================================================

def estimate_distance_from_face(face_width_pixels: float, frame_width: int) -> float:
    """
    Estimate distance to face using pinhole camera model.
    
    Args:
        face_width_pixels: Width of detected face in pixels
        frame_width: Total frame width in pixels
        
    Returns:
        Estimated distance to face in meters
        
    Note:
        Uses typical webcam specifications:
        - Focal length: ~4mm
        - Sensor width: ~4.8mm  
        - Average face width: ~150mm
    """
    if face_width_pixels <= 0:
        return float('inf')
    
    # Convert camera specifications to appropriate units
    sensor_width_mm = 4.8
    focal_length_pixels = (CAMERA_FOCAL_LENGTH_MM * frame_width) / sensor_width_mm
    
    # Pinhole camera model: distance = (real_size * focal_length) / image_size
    distance_mm = (REFERENCE_FACE_SIZE_MM * focal_length_pixels) / face_width_pixels
    return distance_mm / 1000.0  # Convert millimeters to meters

def get_adaptive_proximity_threshold(distance_meters: float) -> float:
    """
    Return appropriate proximity threshold based on estimated distance.
    
    Args:
        distance_meters: Estimated distance to subject in meters
        
    Returns:
        Appropriate face area fraction threshold for proximity detection
    """
    if distance_meters < CLOSE_DISTANCE_THRESHOLD:
        return FACE_AREA_FRAC_CLOSE
    elif distance_meters < MEDIUM_DISTANCE_THRESHOLD:
        return FACE_AREA_FRAC_MEDIUM
    else:
        return FACE_AREA_FRAC_FAR

def get_adaptive_detection_confidence(distance_meters: float) -> float:
    """
    Return appropriate MediaPipe detection confidence based on distance.
    
    Args:
        distance_meters: Estimated distance to subject in meters
        
    Returns:
        Appropriate confidence threshold for MediaPipe face detection
    """
    if distance_meters < CLOSE_DISTANCE_THRESHOLD:
        return 0.5  # Standard confidence for close range
    elif distance_meters < MEDIUM_DISTANCE_THRESHOLD:
        return 0.3  # Lower confidence for medium range
    else:
        return 0.2  # Very low confidence for far range

# =============================================================================
# IMAGE PREPROCESSING FUNCTIONS
# =============================================================================

def preprocess_for_distant_detection(frame: np.ndarray, 
                                     estimated_distance: Optional[float] = None,
                                     face_region: Optional[Tuple[int,int,int,int]] = None, 
                                     enhancement_level: int = 1) -> np.ndarray:
    """
    Preprocess frame to improve distant face detection with progressive enhancement.
    
    Args:
        frame: Input frame to enhance
        estimated_distance: Estimated distance to subject (if known)
        face_region: Known face region for targeted enhancement
        enhancement_level: Enhancement intensity (1=light, 2=medium, 3=aggressive)
        
    Returns:
        Enhanced frame optimized for face detection
        
    Enhancement Techniques:
        Level 1: Light contrast/brightness adjustment
        Level 2: + Noise reduction, advanced sharpening
        Level 3: + Histogram equalization, gamma correction
    """
    enhanced = frame.copy()
    
    # Skip enhancement for close subjects unless specifically requested
    if enhancement_level == 1 and (estimated_distance is None or 
                                   estimated_distance < MEDIUM_DISTANCE_THRESHOLD):
        return enhanced
    
    # Get enhancement parameters based on level
    if enhancement_level <= 1:
        params = ENHANCEMENT_PARAMS['light']
    elif enhancement_level == 2:
        params = ENHANCEMENT_PARAMS['medium']
    else:
        params = ENHANCEMENT_PARAMS['aggressive']
    
    # Step 1: Contrast and brightness adjustment
    enhanced = cv2.convertScaleAbs(enhanced, alpha=params['alpha'], beta=params['beta'])
    
    # Step 2: Noise reduction for better distant detection
    if params['blur_reduce']:
        enhanced = cv2.bilateralFilter(enhanced, 5, 80, 80)
    
    # Step 3: Sharpening filter (adaptive based on enhancement level)
    if enhancement_level >= 2:
        # Advanced 5x5 sharpening kernel for higher levels
        kernel = np.array([[-1,-1,-1,-1,-1],
                          [-1, 2, 2, 2,-1],
                          [-1, 2, 8, 2,-1], 
                          [-1, 2, 2, 2,-1],
                          [-1,-1,-1,-1,-1]]) / 8.0
    else:
        # Basic 3x3 sharpening kernel
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
    enhanced = cv2.filter2D(enhanced, -1, kernel)
    
    # Step 4: Region-specific enhancement for known face areas
    if face_region is not None and enhancement_level >= 2:
        enhanced = _enhance_face_region(enhanced, face_region)
    
    # Step 5: Gamma correction for extreme distant detection
    if enhancement_level >= 3:
        enhanced = _apply_gamma_correction(enhanced, params['gamma'])
    
    return enhanced

def _enhance_face_region(frame: np.ndarray, face_region: Tuple[int,int,int,int]) -> np.ndarray:
    """
    Apply targeted enhancement to a specific face region.
    
    Args:
        frame: Input frame
        face_region: Face bounding box (x1, y1, x2, y2)
        
    Returns:
        Frame with enhanced face region
    """
    x1, y1, x2, y2 = face_region
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    
    if x2 <= x1 or y2 <= y1:
        return frame
    
    # Extract and enhance face region
    face_crop = frame[y1:y2, x1:x2]
    
    # Apply histogram equalization to face region
    face_crop_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    face_crop_eq = cv2.equalizeHist(face_crop_gray)
    face_crop_enhanced = cv2.cvtColor(face_crop_eq, cv2.COLOR_GRAY2BGR)
    
    # Blend enhanced face back into frame (60% original, 40% enhanced)
    frame[y1:y2, x1:x2] = cv2.addWeighted(face_crop, 0.6, face_crop_enhanced, 0.4, 0)
    
    return frame

def _apply_gamma_correction(frame: np.ndarray, gamma: float) -> np.ndarray:
    """
    Apply gamma correction to improve visibility.
    
    Args:
        frame: Input frame
        gamma: Gamma value (< 1.0 brightens, > 1.0 darkens)
        
    Returns:
        Gamma-corrected frame
    """
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 
                     for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(frame, table)

# =============================================================================
# LANDMARK ESTIMATION FUNCTIONS
# =============================================================================

def estimate_landmarks_from_face_box(face_box: Tuple[int,int,int,int], 
                                   frame_shape: Tuple[int,int]) -> Tuple[np.ndarray, List[dict]]:
    """
    Estimate facial landmarks based on face bounding box geometry.
    
    This function provides approximate landmark positions when MediaPipe fails,
    enabling pose estimation to continue with reduced accuracy.
    
    Args:
        face_box: Face bounding box (x1, y1, x2, y2)
        frame_shape: Frame dimensions (height, width)
        
    Returns:
        Tuple of (pts2d, face_kps) in same format as MediaPipe:
        - pts2d: Numpy array of landmark coordinates 
        - face_kps: List of landmark dictionaries with 'x', 'y' keys
        
    Note:
        Landmark positions are based on typical human facial proportions:
        - Nose tip: Center, slightly below middle
        - Chin: Center bottom
        - Eye corners: Left/right sides, upper third
        - Mouth corners: Left/right of center, lower third
    """
    x1, y1, x2, y2 = face_box
    face_width = x2 - x1
    face_height = y2 - y1
    
    # Define landmark positions as relative coordinates (0.0 to 1.0)
    # Based on typical human facial proportions
    landmarks_relative = {
        'nose_tip': (0.5, 0.6),         # Center, slightly down (landmark 1)
        'chin': (0.5, 0.95),            # Center bottom (landmark 199) 
        'left_eye_left': (0.25, 0.4),   # Left side, upper (landmark 33)
        'right_eye_right': (0.75, 0.4), # Right side, upper (landmark 263)
        'mouth_left': (0.35, 0.8),      # Left of center, lower (landmark 61)
        'mouth_right': (0.65, 0.8)      # Right of center, lower (landmark 291)
    }
    
    pts2d = []
    face_kps = []
    
    # Convert relative coordinates to absolute pixel coordinates
    for landmark_name, (rel_x, rel_y) in landmarks_relative.items():
        x = int(x1 + rel_x * face_width)
        y = int(y1 + rel_y * face_height)
        
        # Clamp coordinates to frame boundaries
        x = max(0, min(frame_shape[1] - 1, x))
        y = max(0, min(frame_shape[0] - 1, y))
        
        pts2d.append([x, y])
        face_kps.append({'x': float(x), 'y': float(y)})
    
    return np.array(pts2d, dtype=np.float64), face_kps

# =============================================================================
# VISUALIZATION UTILITY FUNCTIONS
# =============================================================================

def draw_pose_axes(frame: np.ndarray, rvec: np.ndarray, tvec: np.ndarray,
                  cam_matrix: np.ndarray, dist_coeffs: np.ndarray,
                  nose_point: Tuple[int, int], axis_length: float = 80) -> np.ndarray:
    """
    Draw 3D pose axes on the frame to visualize head orientation.
    
    Args:
        frame: Input frame to draw on
        rvec, tvec: Rotation and translation vectors from solvePnP
        cam_matrix, dist_coeffs: Camera calibration parameters
        nose_point: 2D nose tip location
        axis_length: Length of axes in 3D space
        
    Returns:
        Frame with pose axes drawn
    """
    # Define 3D axis points
    axis_3d = np.float32([[axis_length,0,0],      # X-axis (red)
                         [0,axis_length,0],       # Y-axis (green)
                         [0,0,axis_length]])      # Z-axis (blue)
    
    # Project 3D axes to 2D image coordinates
    imgpts, _ = cv2.projectPoints(axis_3d, rvec, tvec, cam_matrix, dist_coeffs)
    
    # Draw axes with different colors
    colors = [COLORS['pose_axes']['x'], 
              COLORS['pose_axes']['y'], 
              COLORS['pose_axes']['z']]
    
    for i, (color, pt) in enumerate(zip(colors, imgpts)):
        pt = tuple(pt.ravel().astype(int))
        cv2.line(frame, nose_point, pt, color, 2)
    
    return frame

def add_info_text(frame: np.ndarray, text_lines: List[str], 
                 start_y: int = 30, line_spacing: int = 25) -> np.ndarray:
    """
    Add multiple lines of information text to frame.
    
    Args:
        frame: Input frame
        text_lines: List of text strings to display
        start_y: Y coordinate for first line
        line_spacing: Vertical spacing between lines
        
    Returns:
        Frame with text added
    """
    for i, text in enumerate(text_lines):
        y_pos = start_y + (i * line_spacing)
        cv2.putText(frame, text, (10, y_pos), TEXT_FONT, TEXT_SCALE, 
                   COLORS['status_text'], TEXT_THICKNESS)
    
    return frame
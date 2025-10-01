"""
Behavior Manager Module for Temi 5-Layer Subsumption Architecture

This module contains behavior management components:
- GazeClassifier: Determines if person is looking at camera
- ProximityGate: Monitors face proximity for interaction readiness
- BehaviorManager: Tracks sustained gaze and triggers behaviors
- SensorGate: Handles sensor-based behavior interrupts

Author: Nicholas Lloyd
Date: October 2025
"""

import time
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass

from config import *
from utils import face_area_fraction, get_adaptive_proximity_threshold

# =============================================================================
# GAZE CLASSIFICATION
# =============================================================================

@dataclass
class GazeClassifier:
    """
    Gaze classification based on head pose thresholds.
    
    This class determines whether a person is looking at the camera based on
    head pose angles (yaw and pitch) derived from facial landmark analysis.
    
    Attributes:
        yaw_thresh: Maximum absolute yaw angle for "looking" classification
        pitch_min: Minimum pitch angle for "looking" classification  
        pitch_max: Maximum pitch angle for "looking" classification
    """
    yaw_thresh: float = YAW_THRESH_DEG
    pitch_min: float = PITCH_MIN_DEG
    pitch_max: float = PITCH_MAX_DEG

    def is_looking(self, pitch: float, yaw: float) -> bool:
        """
        Determine if person is looking at camera based on head pose.
        
        Args:
            pitch: Head pitch angle in degrees (up/down rotation)
            yaw: Head yaw angle in degrees (left/right rotation)
            
        Returns:
            True if head pose indicates person is looking at camera
            
        Logic:
            - Yaw must be within ±yaw_thresh degrees (typically ±20°)
            - Pitch must be within specified range (currently full range accepted)
            - Accounts for camera positioning and typical interaction angles
        """
        yaw_ok = abs(yaw) <= self.yaw_thresh  # -20 to +20 degrees
        
        # Accept full pitch range since -175° has been observed for camera-looking
        # This handles various camera mounting positions and user heights
        pitch_ok = True  # Accept all pitch values for robust detection
        
        return yaw_ok and pitch_ok

def refine_gaze(is_looking: bool, pitch: float, yaw: float, roll: float, 
                face_crop, landmarks_2d) -> bool:
    """
    Accuracy refinement hook for gaze detection enhancement.
    
    This function serves as a placeholder for future gaze detection improvements
    such as eye tracking, iris detection, or temporal smoothing.
    
    Args:
        is_looking: Coarse gaze classification from head pose
        pitch, yaw, roll: Head pose angles in degrees
        face_crop: Cropped face region from frame
        landmarks_2d: 2D facial landmark coordinates
        
    Returns:
        Refined gaze classification (currently unchanged from input)
        
    Future Enhancements:
        - Roboflow eye-gaze model integration
        - MediaPipe iris tracking
        - Temporal smoothing of gaze decisions
        - Eye state analysis (open/closed, blink detection)
        - Gaze direction vector calculation
    """
    # TODO: Integrate advanced eye tracking models here
    # For now, return the head pose-based decision unchanged
    return is_looking

# =============================================================================
# PROXIMITY MONITORING
# =============================================================================

class ProximityGate:
    """
    Proximity detection layer for monitoring face size and distance.
    
    This class monitors whether detected faces meet size thresholds for reliable
    interaction. Uses adaptive thresholds based on estimated distance.
    
    Note: This no longer gates gaze detection but provides monitoring info.
    """
    
    def __init__(self, min_frac: float = FACE_AREA_FRAC_CLOSE):
        """
        Initialize proximity monitoring.
        
        Args:
            min_frac: Default minimum face area fraction for proximity
        """
        self.default_min_frac = float(min_frac)
        
        print(f"✓ Proximity Gate initialized (min face area: {min_frac:.3f})")

    def is_close(self, box: Tuple[int,int,int,int], frame_shape: Tuple[int,int], 
                 estimated_distance: Optional[float] = None) -> Tuple[bool, float, float]:
        """
        Check if face meets proximity thresholds for reliable interaction.
        
        Args:
            box: Face bounding box (x1, y1, x2, y2)
            frame_shape: Frame dimensions (height, width)
            estimated_distance: Optional distance estimate for adaptive thresholds
            
        Returns:
            Tuple of (is_close, area_fraction, threshold_used)
            
        Note:
            This provides monitoring information but does not gate detection.
            Useful for display, debugging, and interaction quality assessment.
        """
        area_frac = face_area_fraction(box, frame_shape)
        
        # Use adaptive threshold if distance is estimated
        if estimated_distance is not None:
            threshold = get_adaptive_proximity_threshold(estimated_distance)
        else:
            threshold = self.default_min_frac
        
        # Check minimum pixel size to avoid tiny detections
        x1, y1, x2, y2 = box
        face_width = max(0, x2-x1)
        face_height = max(0, y2-y1)
        min_size_ok = face_width >= MIN_FACE_PIXELS and face_height >= MIN_FACE_PIXELS
        
        is_close = area_frac >= threshold and min_size_ok
        return is_close, area_frac, threshold

# =============================================================================
# SENSOR INTERRUPT HANDLING
# =============================================================================

class SensorGate:
    """
    Sensor interrupt layer for handling hardware-based behavior interruptions.
    
    This class manages sensor-based interrupts that can override gaze detection
    and other behaviors for safety or operational reasons.
    """
    
    def __init__(self):
        """Initialize sensor gate with default state."""
        self.state = {
            'emergency_stop': False,
            'obstacle_close': False,
            'override_hold': False,
            'manual_control': False,
            'system_fault': False,
            'last_update': time.time()
        }
        
        print("✓ Sensor Gate initialized")
    
    def update_from_sensors(self, sensor_state: Optional[Dict[str, Any]]):
        """
        Update sensor state from external sensor inputs.
        
        Args:
            sensor_state: Dictionary of sensor readings and states
        """
        if sensor_state:
            self.state.update(sensor_state)
            self.state['last_update'] = time.time()
    
    def should_interrupt(self) -> bool:
        """
        Check if sensors require behavior interruption.
        
        Returns:
            True if any sensor condition requires immediate attention
        """
        return (self.state.get('emergency_stop', False) or 
                self.state.get('obstacle_close', False) or 
                self.state.get('override_hold', False) or
                self.state.get('manual_control', False) or
                self.state.get('system_fault', False))
    
    def get_interrupt_reason(self) -> str:
        """
        Get reason for current interrupt state.
        
        Returns:
            String describing why interruption is active
        """
        if self.state.get('emergency_stop'):
            return 'EMERGENCY_STOP'
        elif self.state.get('obstacle_close'):
            return 'OBSTACLE_CLOSE'
        elif self.state.get('manual_control'):
            return 'MANUAL_CONTROL'
        elif self.state.get('override_hold'):
            return 'OVERRIDE_HOLD'
        elif self.state.get('system_fault'):
            return 'SYSTEM_FAULT'
        else:
            return 'NORMAL'

# =============================================================================
# BEHAVIOR MANAGER (SUSTAINED GAZE TRACKING)
# =============================================================================

class BehaviorManager:
    """
    Behavior manager for sustained gaze detection and behavior triggering.
    
    This class tracks how long a person has been looking at the camera and
    triggers behaviors when sustained gaze thresholds are met. Integrates
    with sensor interrupts for safety override.
    """
    
    def __init__(self, sustained_seconds: float = SUSTAINED_GAZE_SEC, 
                 sensor_gate: Optional[SensorGate] = None):
        """
        Initialize behavior manager.
        
        Args:
            sustained_seconds: Time threshold for sustained gaze detection
            sensor_gate: Optional sensor gate for interrupt handling
        """
        self.sustained_seconds = sustained_seconds
        self.sensor_gate = sensor_gate or SensorGate()
        self.gaze_start_t = None
        self.approach_triggered = False
        self.gaze_history = []
        
        print(f"✓ Behavior Manager initialized (sustained gaze: {sustained_seconds}s)")

    def update(self, now: float, looking: bool) -> Dict[str, Any]:
        """
        Update behavior state based on current gaze detection.
        
        Args:
            now: Current timestamp
            looking: Whether person is currently looking at camera
            
        Returns:
            Dictionary containing behavior state information:
            {
                'state': str,           # 'LOOKING', 'NOT_LOOKING', 'INTERRUPTED'
                'elapsed': float,       # Time spent in current state
                'sustained': bool,      # Whether sustained threshold is met
                'triggered': bool       # Whether behavior has been triggered
            }
        """
        # Highest priority: check for sensor interrupts
        if self.sensor_gate.should_interrupt():
            self.gaze_start_t = None
            return {
                'state': 'INTERRUPTED', 
                'elapsed': 0.0,
                'sustained': False,
                'triggered': False,
                'interrupt_reason': self.sensor_gate.get_interrupt_reason()
            }

        if looking:
            # Person is looking at camera
            if self.gaze_start_t is None:
                self.gaze_start_t = now
                self.approach_triggered = False  # Reset trigger for new gaze session
            
            elapsed = now - self.gaze_start_t
            sustained = elapsed >= self.sustained_seconds
            
            # Trigger behavior when sustained threshold is first reached
            if sustained and not self.approach_triggered:
                self.trigger_behavior(elapsed)
                self.approach_triggered = True
            
            return {
                'state': 'LOOKING', 
                'elapsed': elapsed,
                'sustained': sustained,
                'triggered': self.approach_triggered
            }
        else:
            # Person is not looking at camera
            if self.gaze_start_t is not None:
                # Log completed gaze session
                final_duration = now - self.gaze_start_t
                self.gaze_history.append({
                    'start_time': self.gaze_start_t,
                    'end_time': now,
                    'duration': final_duration,
                    'triggered': self.approach_triggered
                })
                
                # Keep only recent history
                self.gaze_history = self.gaze_history[-20:]
            
            self.gaze_start_t = None
            self.approach_triggered = False  # Reset for next detection
            
            return {
                'state': 'NOT_LOOKING', 
                'elapsed': 0.0,
                'sustained': False,
                'triggered': False
            }

    def trigger_behavior(self, elapsed_time: float):
        """
        Trigger behavior action when sustained gaze is detected.
        
        Args:
            elapsed_time: Duration of sustained gaze
            
        Note:
            This method can be extended to send commands to robot systems,
            log interaction events, or coordinate with other subsumption layers.
        """
        print(f"🎯 BEHAVIOR TRIGGERED: Sustained gaze detected for {elapsed_time:.1f}s")
        
        # Future: Add robot behavior commands here
        # - Send navigation commands to approach person
        # - Activate interaction protocols  
        # - Log interaction events
        # - Coordinate with Temi API
    
    def get_gaze_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about recent gaze interactions.
        
        Returns:
            Dictionary containing gaze interaction statistics
        """
        if not self.gaze_history:
            return {
                'total_sessions': 0,
                'average_duration': 0.0,
                'triggered_sessions': 0,
                'trigger_rate': 0.0
            }
        
        total_sessions = len(self.gaze_history)
        total_duration = sum(session['duration'] for session in self.gaze_history)
        triggered_sessions = sum(1 for session in self.gaze_history if session['triggered'])
        
        return {
            'total_sessions': total_sessions,
            'average_duration': total_duration / total_sessions,
            'triggered_sessions': triggered_sessions,
            'trigger_rate': triggered_sessions / total_sessions if total_sessions > 0 else 0.0,
            'recent_sessions': self.gaze_history[-5:]  # Last 5 sessions
        }
    
    def reset_state(self):
        """Reset behavior manager state (useful for testing or reinitialization)."""
        self.gaze_start_t = None
        self.approach_triggered = False
        print("🔄 Behavior manager state reset")

# =============================================================================
# INTEGRATED BEHAVIOR COORDINATOR
# =============================================================================

class BehaviorCoordinator:
    """
    Coordinates all behavior management components for integrated operation.
    
    This class brings together gaze classification, proximity monitoring,
    sensor interrupts, and behavior triggering into a unified interface.
    """
    
    def __init__(self):
        """Initialize integrated behavior coordinator."""
        self.gaze_classifier = GazeClassifier()
        self.proximity_gate = ProximityGate()
        self.sensor_gate = SensorGate()
        self.behavior_manager = BehaviorManager(sensor_gate=self.sensor_gate)
        
        print("✓ Behavior Coordinator initialized")
    
    def process_frame_data(self, pose_data: Optional[Dict], timestamp: float, 
                          sensor_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Process complete frame data through behavior pipeline.
        
        Args:
            pose_data: Head pose estimation results from detection layers
            timestamp: Current timestamp
            sensor_data: Optional sensor readings for interrupt handling
            
        Returns:
            Comprehensive behavior analysis results
        """
        # Update sensor gate if sensor data provided
        if sensor_data:
            self.sensor_gate.update_from_sensors(sensor_data)
        
        # Initialize result structure
        result = {
            'timestamp': timestamp,
            'pose_valid': False,
            'looking': False,
            'proximity_ok': False,
            'behavior_state': 'NOT_LOOKING',
            'sustained_gaze': False,
            'distance_estimate': None,
            'detection_method': 'none'
        }
        
        # Process pose data if available
        if pose_data and pose_data.get('ok'):
            result['pose_valid'] = True
            result['detection_method'] = pose_data.get('detection_method', 'unknown')
            result['distance_estimate'] = pose_data.get('estimated_distance')
            
            # Gaze classification
            pitch = pose_data['pitch']
            yaw = pose_data['yaw']
            roll = pose_data['roll']
            
            # Get coarse gaze classification
            coarse_looking = self.gaze_classifier.is_looking(pitch, yaw)
            
            # Apply refinement (currently pass-through, future enhancement hook)
            face_crop = None  # Would extract from frame data
            landmarks_2d = pose_data.get('pts2d')
            looking = refine_gaze(coarse_looking, pitch, yaw, roll, face_crop, landmarks_2d)
            
            result['looking'] = looking
            
            # Proximity assessment
            if 'face_box' in pose_data:
                face_box = pose_data['face_box']
                # Frame shape would be passed in - using default for now
                frame_shape = (480, 640)  # Default camera resolution
                is_close, area_frac, threshold = self.proximity_gate.is_close(
                    face_box, frame_shape, result['distance_estimate']
                )
                result['proximity_ok'] = is_close
                result['face_area_fraction'] = area_frac
                result['proximity_threshold'] = threshold
        
        # Update behavior manager
        behavior_state = self.behavior_manager.update(timestamp, result['looking'])
        result['behavior_state'] = behavior_state['state']
        result['sustained_gaze'] = behavior_state.get('sustained', False)
        result['gaze_elapsed'] = behavior_state.get('elapsed', 0.0)
        result['behavior_triggered'] = behavior_state.get('triggered', False)
        
        # Add interrupt information
        if behavior_state['state'] == 'INTERRUPTED':
            result['interrupt_reason'] = behavior_state.get('interrupt_reason', 'UNKNOWN')
        
        return result
    
    def get_status_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive status summary of all behavior components.
        
        Returns:
            Dictionary containing status of all behavior management components
        """
        return {
            'gaze_classifier': {
                'yaw_threshold': self.gaze_classifier.yaw_thresh,
                'pitch_range': (self.gaze_classifier.pitch_min, self.gaze_classifier.pitch_max)
            },
            'proximity_gate': {
                'default_threshold': self.proximity_gate.default_min_frac
            },
            'sensor_gate': {
                'interrupt_active': self.sensor_gate.should_interrupt(),
                'interrupt_reason': self.sensor_gate.get_interrupt_reason(),
                'state': self.sensor_gate.state
            },
            'behavior_manager': {
                'sustained_threshold': self.behavior_manager.sustained_seconds,
                'current_gaze_active': self.behavior_manager.gaze_start_t is not None,
                'statistics': self.behavior_manager.get_gaze_statistics()
            }
        }
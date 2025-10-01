"""
Subsumption Layers Module for Temi 5-Layer Architecture

This module implements the five-layer subsumption architecture for the Temi robot:
Layer 1: Battery Management (Highest Priority)
Layer 2: Sensor Safety Systems  
Layer 3: Human Interaction Management
Layer 4: Gaze Detection Processing
Layer 5: Base Robot Behaviors (Lowest Priority)

Each layer can suppress lower-priority layers when active.

Author: Nicholas Lloyd
Date: October 2025
"""

import time
from typing import Dict, Optional, Any
from dataclasses import dataclass

from config import *

# =============================================================================
# LAYER 1: BATTERY MANAGEMENT (HIGHEST PRIORITY)
# =============================================================================

class LowBatteryLayer:
    """
    Layer 1: Low battery detection and emergency power management.
    
    This is the highest priority layer that monitors battery levels and can
    override all other behaviors when battery is critically low.
    
    Behaviors:
    - EMERGENCY_CHARGE: Critical battery, immediate charging required
    - SEEK_CHARGING: Low battery, should find charging station
    - NORMAL: Battery levels acceptable
    """
    
    def __init__(self, critical_level: float = CRITICAL_BATTERY_LEVEL, 
                 low_level: float = LOW_BATTERY_LEVEL):
        """
        Initialize battery management layer.
        
        Args:
            critical_level: Battery percentage for emergency mode
            low_level: Battery percentage for low battery warning
        """
        self.critical_battery_level = critical_level
        self.low_battery_level = low_level
        self.battery_status = {
            'level': 100.0,
            'charging': False,
            'critical': False,
            'low': False,
            'last_update': time.time()
        }
        
        print(f"✓ Battery Layer initialized (Critical: {critical_level}%, Low: {low_level}%)")
        
    def update_battery_status(self, battery_level: float, charging: bool = False) -> Dict[str, Any]:
        """
        Update battery status from Temi robot sensors.
        
        Args:
            battery_level: Current battery percentage (0-100)
            charging: Whether robot is currently charging
            
        Returns:
            Updated battery status dictionary
        """
        self.battery_status = {
            'level': battery_level,
            'charging': charging,
            'critical': battery_level <= self.critical_battery_level,
            'low': battery_level <= self.low_battery_level,
            'last_update': time.time()
        }
        
        # Log battery state changes
        if self.battery_status['critical'] and not charging:
            print(f"🔋 CRITICAL: Battery at {battery_level:.1f}% - Emergency charging required!")
        elif self.battery_status['low'] and not charging:
            print(f"🔋 LOW: Battery at {battery_level:.1f}% - Seeking charging station")
            
        return self.battery_status
    
    def should_interrupt(self) -> bool:
        """
        Check if battery level requires immediate behavior interruption.
        
        Returns:
            True if critical battery and not charging
        """
        return self.battery_status['critical'] and not self.battery_status['charging']
    
    def get_battery_behavior(self) -> str:
        """
        Get required battery management behavior.
        
        Returns:
            String indicating required behavior:
            - 'EMERGENCY_CHARGE': Immediate charging required
            - 'SEEK_CHARGING': Should find charging station
            - 'NORMAL': No battery action needed
        """
        if self.battery_status['critical']:
            return 'EMERGENCY_CHARGE'
        elif self.battery_status['low']:
            return 'SEEK_CHARGING'
        else:
            return 'NORMAL'

# =============================================================================
# LAYER 2: SENSOR SAFETY SYSTEMS
# =============================================================================

class TemiSensorLayer:
    """
    Layer 2: Temi robot sensor readings and safety systems.
    
    This layer monitors all robot sensors for safety-critical conditions:
    - Collision detection
    - Proximity warnings
    - Virtual wall detection
    - Manual control override
    - Obstacle avoidance
    """
    
    def __init__(self):
        """Initialize sensor monitoring layer."""
        self.sensor_data = {
            'collision_detected': False,
            'proximity_warning': False,
            'virtual_wall_detected': False,
            'mapping_active': False,
            'obstacle_distance': float('inf'),
            'temi_moving': False,
            'manual_control': False,
            'sensor_health': 'OK',
            'last_update': time.time()
        }
        
        print("✓ Sensor Layer initialized")
        
    def update_sensors(self, **sensor_readings) -> Dict[str, Any]:
        """
        Update sensor readings from Temi robot systems.
        
        Args:
            **sensor_readings: Keyword arguments containing sensor data
            
        Returns:
            Updated sensor data dictionary
            
        Example:
            update_sensors(
                collision_detected=False,
                proximity_warning=True,
                obstacle_distance=0.8
            )
        """
        self.sensor_data.update(sensor_readings)
        self.sensor_data['last_update'] = time.time()
        
        # Log critical sensor events
        if sensor_readings.get('collision_detected'):
            print("⚠️ COLLISION DETECTED!")
        elif sensor_readings.get('virtual_wall_detected'):
            print("🚧 Virtual wall detected")
        elif sensor_readings.get('obstacle_distance', float('inf')) < OBSTACLE_SAFETY_DISTANCE:
            print(f"⚠️ Obstacle too close: {sensor_readings['obstacle_distance']:.2f}m")
            
        return self.sensor_data
    
    def should_interrupt(self) -> bool:
        """
        Check if sensors require immediate behavior interruption.
        
        Returns:
            True if any safety-critical sensor condition is detected
        """
        return (self.sensor_data['collision_detected'] or 
                self.sensor_data['virtual_wall_detected'] or
                self.sensor_data['manual_control'] or
                self.sensor_data['obstacle_distance'] < OBSTACLE_SAFETY_DISTANCE)
    
    def get_sensor_status(self) -> str:
        """
        Get current sensor status for behavior selection.
        
        Returns:
            String describing current sensor condition:
            - 'COLLISION': Collision detected
            - 'VIRTUAL_WALL': Virtual boundary detected
            - 'MANUAL_OVERRIDE': Human taking manual control
            - 'PROXIMITY_WARNING': Object nearby but not critical
            - 'CLEAR': All sensors normal
        """
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

# =============================================================================
# LAYER 3: HUMAN INTERACTION MANAGEMENT
# =============================================================================

class ApproachingLayer:
    """
    Layer 3: Human approach and interaction management.
    
    This layer handles the robot's interaction with humans who need assistance:
    - Detecting people who need help (sustained gaze)
    - Approaching people at appropriate distances
    - Managing interaction timeouts
    - Coordinating help-seeking behaviors
    """
    
    def __init__(self, approach_distance: float = APPROACH_DISTANCE, 
                 interaction_timeout: float = INTERACTION_TIMEOUT):
        """
        Initialize human interaction layer.
        
        Args:
            approach_distance: Distance threshold for initiating approach
            interaction_timeout: Maximum interaction duration in seconds
        """
        self.approach_distance = approach_distance
        self.interaction_timeout = interaction_timeout
        self.interaction_state = {
            'person_identified': False,
            'approach_initiated': False,
            'interaction_active': False,
            'interaction_start_time': None,
            'target_distance': None,
            'help_requested': False,
            'last_update': time.time()
        }
        
        print(f"✓ Approaching Layer initialized (Approach: {approach_distance}m, Timeout: {interaction_timeout}s)")
        
    def update_interaction(self, person_detected: bool, estimated_distance: Optional[float], 
                          sustained_gaze: bool, timestamp: float) -> Dict[str, Any]:
        """
        Update interaction state based on person detection and gaze analysis.
        
        Args:
            person_detected: Whether a person is currently detected
            estimated_distance: Distance to detected person (if available)
            sustained_gaze: Whether person has been looking at camera for sufficient time
            timestamp: Current timestamp
            
        Returns:
            Updated interaction state dictionary
        """
        
        # Check if person needs help (sustained gaze indicates need for assistance)
        if person_detected and sustained_gaze and estimated_distance:
            if not self.interaction_state['person_identified']:
                self.interaction_state['person_identified'] = True
                self.interaction_state['target_distance'] = estimated_distance
                print(f"👤 Person identified for assistance at {estimated_distance:.1f}m")
            
            # Initiate approach if person is far enough to warrant approaching
            if (estimated_distance > self.approach_distance and 
                not self.interaction_state['approach_initiated']):
                self.interaction_state['approach_initiated'] = True
                self.interaction_state['interaction_start_time'] = timestamp
                print(f"🤖 APPROACHING: Moving closer to person at {estimated_distance:.1f}m")
            
            # Start interaction if close enough
            elif (estimated_distance <= self.approach_distance and 
                  self.interaction_state['approach_initiated']):
                self.interaction_state['interaction_active'] = True
                print(f"💬 INTERACTION STARTED: Person reached at {estimated_distance:.1f}m")
        
        # Reset if person is no longer detected or no longer looking
        elif not person_detected or not sustained_gaze:
            if self.interaction_state['interaction_active']:
                print("👋 INTERACTION ENDED: Person no longer detected or looking away")
            self._reset_interaction_state()
        
        # Check for interaction timeout
        if (self.interaction_state['interaction_start_time'] and 
            timestamp - self.interaction_state['interaction_start_time'] > self.interaction_timeout):
            print("⏰ INTERACTION TIMEOUT: Returning to patrol")
            self._reset_interaction_state()
        
        self.interaction_state['last_update'] = timestamp
        return self.interaction_state
    
    def _reset_interaction_state(self):
        """Reset interaction state to default values."""
        self.interaction_state.update({
            'person_identified': False,
            'approach_initiated': False,
            'interaction_active': False,
            'interaction_start_time': None,
            'target_distance': None,
            'help_requested': False
        })
    
    def should_interrupt(self) -> bool:
        """
        Check if approaching layer should control robot behavior.
        
        Returns:
            True if actively approaching or interacting with person
        """
        return (self.interaction_state['approach_initiated'] or 
                self.interaction_state['interaction_active'])
    
    def get_approach_behavior(self) -> str:
        """
        Get current approach behavior requirement.
        
        Returns:
            String indicating required behavior:
            - 'INTERACTING': Currently interacting with person
            - 'APPROACHING': Moving toward person who needs help
            - 'PERSON_IDENTIFIED': Person detected but no approach needed
            - 'PATROL': Normal patrol behavior
        """
        if self.interaction_state['interaction_active']:
            return 'Approach'
        elif self.interaction_state['approach_initiated']:
            return 'Approach'
        elif self.interaction_state['person_identified']:
            return 'PERSON_IDENTIFIED'
        else:
            return 'PATROL'

# =============================================================================
# LAYER 4: GAZE DETECTION PROCESSING
# =============================================================================

class GazeDetectionLayer:
    """
    Layer 4: Combined gaze detection using multiple detection methods.
    
    This layer coordinates different gaze detection approaches:
    - MediaPipe + OpenCV head pose estimation (primary)
    - Roboflow eye detection (future enhancement)
    - Confidence weighting and method selection
    """
    
    def __init__(self):
        """Initialize gaze detection coordination layer."""
        self.mediapipe_active = True
        self.roboflow_active = False  # Placeholder for future integration
        self.gaze_data = {
            'mediapipe_result': None,
            'roboflow_result': None,
            'combined_result': None,
            'confidence_source': 'none',
            'last_update': time.time()
        }
        
        print("✓ Gaze Detection Layer initialized")
        
    def update_mediapipe_gaze(self, pose_data: Optional[Dict], looking: bool) -> Dict[str, Any]:
        """
        Update gaze detection from MediaPipe + OpenCV pose estimation.
        
        Args:
            pose_data: Head pose estimation data from detection layers
            looking: Boolean result from gaze classification
            
        Returns:
            MediaPipe gaze result dictionary
        """
        self.gaze_data['mediapipe_result'] = {
            'active': True,
            'looking': looking,
            'pose_data': pose_data,
            'confidence': 0.8 if pose_data and pose_data.get('ok') else 0.0,
            'method': pose_data.get('detection_method', 'unknown') if pose_data else 'failed'
        }
        return self.gaze_data['mediapipe_result']
    
    def update_roboflow_gaze(self, eye_detection_result: Optional[Dict]) -> Dict[str, Any]:
        """
        Update gaze detection from Roboflow eye detection (placeholder for future).
        
        Args:
            eye_detection_result: Result from Roboflow eye detection model
            
        Returns:
            Roboflow gaze result dictionary
        """
        # Placeholder for future Roboflow eye detection integration
        self.gaze_data['roboflow_result'] = {
            'active': self.roboflow_active,
            'eye_detected': False,
            'gaze_direction': None,
            'confidence': 0.0
        }
        
        if eye_detection_result and self.roboflow_active:
            self.gaze_data['roboflow_result'].update(eye_detection_result)
        
        return self.gaze_data['roboflow_result']
    
    def get_combined_gaze_result(self) -> Dict[str, Any]:
        """
        Combine results from all available gaze detection methods.
        
        Returns:
            Combined gaze detection result with method prioritization:
            - Roboflow (highest priority when available)
            - MediaPipe (primary method currently)
            - None (no reliable detection)
        """
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
        self.gaze_data['last_update'] = time.time()
        return combined_result

# =============================================================================
# LAYER 5: BASE ROBOT BEHAVIORS (LOWEST PRIORITY)
# =============================================================================

class BaseBehaviorLayer:
    """
    Layer 5: Base robot behaviors when no higher priority layers are active.
    
    This layer implements the default behaviors when the robot is not
    responding to battery, sensor, or interaction needs:
    - PATROL: Active movement and area monitoring
    - IDLE: Stationary but alert
    - CHARGE_SEEKING: Looking for charging station
    - STANDBY: Low-power waiting mode
    """
    
    def __init__(self):
        """Initialize base behavior layer."""
        self.base_behaviors = ['PATROL', 'IDLE', 'CHARGE_SEEKING', 'STANDBY']
        self.current_base_behavior = 'IDLE'
        self.behavior_start_time = None
        self.behavior_history = []
        
        print("✓ Base Behavior Layer initialized")
        
    def update_base_behavior(self, timestamp: float, battery_level: float, 
                           no_interactions: bool = True) -> str:
        """
        Update base behavior based on current robot conditions.
        
        Args:
            timestamp: Current timestamp
            battery_level: Current battery percentage
            no_interactions: Whether higher-priority layers are inactive
            
        Returns:
            Current base behavior string
            
        Behavior Logic:
        - Battery < 30%: CHARGE_SEEKING
        - Battery > 50% + no interactions: PATROL
        - No interactions but lower battery: STANDBY
        - Interactions active: IDLE
        """
        
        # Determine appropriate behavior based on conditions
        if battery_level < 30.0:
            new_behavior = 'CHARGE_SEEKING'
        elif no_interactions and battery_level > PATROL_BATTERY_THRESHOLD:
            new_behavior = 'PATROL'
        elif no_interactions:
            new_behavior = 'STANDBY'
        else:
            new_behavior = 'IDLE'
        
        # Update behavior if it has changed
        if new_behavior != self.current_base_behavior:
            # Log behavior change
            if self.current_base_behavior != 'IDLE':  # Don't log initial idle state
                duration = timestamp - self.behavior_start_time if self.behavior_start_time else 0
                self.behavior_history.append({
                    'behavior': self.current_base_behavior,
                    'duration': duration,
                    'end_time': timestamp
                })
            
            print(f"🤖 BASE BEHAVIOR: {self.current_base_behavior} → {new_behavior}")
            self.current_base_behavior = new_behavior
            self.behavior_start_time = timestamp
        
        return self.current_base_behavior
    
    def get_behavior_duration(self, timestamp: float) -> float:
        """
        Get duration of current behavior in seconds.
        
        Args:
            timestamp: Current timestamp
            
        Returns:
            Duration of current behavior in seconds
        """
        if self.behavior_start_time:
            return timestamp - self.behavior_start_time
        return 0.0
    
    def get_behavior_history(self) -> list:
        """
        Get history of recent behaviors.
        
        Returns:
            List of recent behavior records
        """
        return self.behavior_history[-10:]  # Return last 10 behaviors

# =============================================================================
# SUBSUMPTION ARCHITECTURE COORDINATOR
# =============================================================================

class SubsumptionCoordinator:
    """
    Coordinates all five subsumption layers and determines which layer controls the robot.
    
    Layer Priority (highest to lowest):
    1. Battery Management
    2. Sensor Safety
    3. Human Interaction
    4. Gaze Detection
    5. Base Behaviors
    """
    
    def __init__(self):
        """Initialize the subsumption architecture coordinator."""
        self.layers = {
            1: LowBatteryLayer(),
            2: TemiSensorLayer(),
            3: ApproachingLayer(),
            4: GazeDetectionLayer(),
            5: BaseBehaviorLayer()
        }
        
        print("✓ Subsumption Architecture Coordinator initialized")
        print("  Layer priorities: Battery > Sensors > Interaction > Gaze > Base")
    
    def update_all_layers(self, sensor_data: Dict, gaze_data: Dict, 
                         timestamp: float) -> Dict[str, Any]:
        """
        Update all layers and determine active behavior.
        
        Args:
            sensor_data: Current sensor readings
            gaze_data: Current gaze detection results
            timestamp: Current timestamp
            
        Returns:
            Dictionary containing active layer info and behavior commands
        """
        # Update each layer with current data
        battery_status = self.layers[1].update_battery_status(
            sensor_data.get('battery_level', 100.0),
            sensor_data.get('charging', False)
        )
        
        sensor_status = self.layers[2].update_sensors(**sensor_data)
        
        interaction_status = self.layers[3].update_interaction(
            gaze_data.get('person_detected', False),
            gaze_data.get('estimated_distance'),
            gaze_data.get('sustained_gaze', False),
            timestamp
        )
        
        gaze_result = self.layers[4].get_combined_gaze_result()
        
        # Determine which layer should control behavior (highest priority active layer)
        active_layer = self._determine_active_layer()
        
        # Update base behavior only if no higher priority layers are active
        no_higher_priority = active_layer == 5
        base_behavior = self.layers[5].update_base_behavior(
            timestamp, battery_status['level'], no_higher_priority
        )
        
        return {
            'active_layer': active_layer,
            'layer_status': {
                1: {'name': 'Battery', 'status': self.layers[1].get_battery_behavior()},
                2: {'name': 'Sensors', 'status': self.layers[2].get_sensor_status()},
                3: {'name': 'Interaction', 'status': self.layers[3].get_approach_behavior()},
                4: {'name': 'Gaze', 'status': gaze_result.get('method', 'none')},
                5: {'name': 'Base', 'status': base_behavior}
            },
            'behavior_command': self._get_behavior_command(active_layer),
            'priority_suppressed': [i for i in range(active_layer + 1, 6)],
            'timestamp': timestamp
        }
    
    def _determine_active_layer(self) -> int:
        """
        Determine which layer should control robot behavior.
        
        Returns:
            Layer number (1-5) of highest priority active layer
        """
        # Check layers in priority order (1 = highest priority)
        for layer_num in [1, 2, 3, 4, 5]:
            layer = self.layers[layer_num]
            
            if hasattr(layer, 'should_interrupt') and layer.should_interrupt():
                return layer_num
        
        # If no layer is interrupting, base behavior is active
        return 5
    
    def _get_behavior_command(self, active_layer: int) -> str:
        """
        Get behavior command from the active layer.
        
        Args:
            active_layer: Number of currently active layer
            
        Returns:
            Behavior command string
        """
        layer = self.layers[active_layer]
        
        if active_layer == 1:
            return layer.get_battery_behavior()
        elif active_layer == 2:
            return layer.get_sensor_status()
        elif active_layer == 3:
            return layer.get_approach_behavior()
        elif active_layer == 4:
            return "GAZE_PROCESSING"
        else:  # Layer 5
            return layer.current_base_behavior
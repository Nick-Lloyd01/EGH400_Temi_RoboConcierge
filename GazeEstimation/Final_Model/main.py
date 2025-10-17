"""


source ~/venvs/TemiGaze/bin/activate



Main Control Center for Temi 5-Layer Subsumption Architecture

This is the main orchestration module that coordinates all components:
- Camera initialization and frame capture
- Face detection and pose estimation pipeline
- Subsumption architecture coordination
- Behavior management and gaze tracking
- Visualization and user interface
- Performance monitoring and logging

Author: Nicholas Lloyd
Date: October 2025

Usage:
    python main.py

Controls:
    'q' - Quit application
    'r' - Reset behavior manager state
    's' - Print status summary
    'p' - Toggle performance logging
"""

import cv2
import time
import numpy as np
from typing import Dict, List, Optional, Any

# Import all our modular components
from config import *
from utils import *
from detection_layers import FacePoseEstimator
from subsumption_layers import SubsumptionCoordinator  
from behavior_manager import BehaviorCoordinator

# =============================================================================
# MAIN APPLICATION CLASS
# =============================================================================

class TemiControlCenter:
    """
    Main control center for the Temi 5-layer subsumption architecture.
    
    This class orchestrates all components and provides the main execution loop
    for the face-first gaze detection and behavior management system.
    """
    
    def __init__(self):
        """Initialize the complete Temi control system."""
        print("🤖 Initializing Temi 5-Layer Subsumption Architecture...")
        print("=" * 60)
        
        # Initialize camera or video input
        self.cap = self._initialize_camera()
        self.video_writer = None  # For saving processed video output
        
        # Initialize core detection components
        self.face_estimator = FacePoseEstimator(MAX_NUM_FACES)
        
        # Initialize subsumption architecture
        self.subsumption = SubsumptionCoordinator()
        
        # Initialize behavior management
        self.behavior_coord = BehaviorCoordinator()
        
        # Performance tracking
        self.frame_count = 0
        self.last_time = time.time()
        self.performance_data = []
        
        # Application state
        self.running = True
        self.show_debug = ENABLE_DEBUG_PRINTS
        self.log_performance = ENABLE_PERFORMANCE_LOGGING
        self.is_video_mode = VIDEO_MODE  # Track if we're processing video file
        
        print("=" * 60)
        print("✅ Temi Control Center initialized successfully!")
        print("\nMode: " + ("📹 VIDEO PROCESSING" if VIDEO_MODE else "📷 LIVE CAMERA"))
        if VIDEO_MODE:
            print(f"  Input:  {INPUT_VIDEO_PATH}")
            print(f"  Output: {OUTPUT_VIDEO_PATH}")
            print(f"  Display: {'ON (slower)' if SHOW_DISPLAY_WINDOW else 'OFF (faster processing)'}")
        print("\nFace-First Detection Pipeline:")
        print("  YOLO Face Detection → MediaPipe Landmarks → Pose Estimation")
        print("\nSubsumption Layers (Priority Order):")
        print("  1. Battery Management (HIGHEST)")
        print("  2. Sensor Safety Systems")
        print("  3. Human Interaction Management")
        print("  4. Face-First Gaze Detection")
        print("  5. Base Robot Behaviors (LOWEST)")
        print("\nControls: 'q'=quit, 'r'=reset, 's'=status, 'p'=toggle performance")
        print("=" * 60)
    
    def _initialize_camera(self) -> cv2.VideoCapture:
        """
        Initialize camera or video file input based on VIDEO_MODE.
        
        Returns:
            OpenCV VideoCapture object
            
        Raises:
            RuntimeError: If no camera/video can be opened
        """
        if VIDEO_MODE:
            # Video file processing mode
            print(f"📹 Opening video file: {INPUT_VIDEO_PATH}")
            cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
            if not cap.isOpened():
                raise RuntimeError(f"❌ ERROR: Could not open video file: {INPUT_VIDEO_PATH}")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"✓ Video opened successfully")
            print(f"  Resolution: {width}x{height}")
            print(f"  FPS: {fps}")
            print(f"  Total frames: {total_frames}")
            
            return cap
        else:
            # Live camera mode
            print("📹 Initializing camera...")
            
            # Try primary camera
            cap = cv2.VideoCapture(CAMERA_INDEX)
            if cap.isOpened():
                print(f"✓ Camera opened successfully (index {CAMERA_INDEX})")
                self._configure_camera(cap)
                return cap
            
            # Try backup camera
            print(f"⚠ Camera {CAMERA_INDEX} failed, trying backup...")
            cap = cv2.VideoCapture(CAMERA_BACKUP_INDEX)
            if cap.isOpened():
                print(f"✓ Backup camera opened (index {CAMERA_BACKUP_INDEX})")
                self._configure_camera(cap)
                return cap
            
            # No camera available
            raise RuntimeError("❌ ERROR: No camera available")
    
    def _configure_camera(self, cap: cv2.VideoCapture):
        """Configure camera settings for optimal performance."""
        # Set resolution if specified
        if CAMERA_RESOLUTION != (0, 0):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])
        
        # Set frame rate
        cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        
        # Get actual settings
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"  Resolution: {width}x{height}")
        print(f"  Frame rate: {fps:.1f} FPS")
    
    def run(self):
        """Main execution loop for the Temi control system."""
        print("\n🚀 Starting Temi Control Center...")
        
        # Initialize video writer if in video mode
        if self.is_video_mode:
            self._initialize_video_writer()
        
        try:
            while self.running and self.cap.isOpened():
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    if self.is_video_mode:
                        print("✅ Video processing complete - End of file reached")
                    else:
                        print("❌ Failed to capture frame")
                    break
                
                # Process frame through complete pipeline
                results = self._process_frame(frame)
                
                # Render frame with visualizations
                display_frame = self._render_frame(frame, results)
                
                # Write frame to output video if in video mode
                if self.is_video_mode and self.video_writer is not None:
                    self.video_writer.write(display_frame)
                
                # Show frame (skip display in video mode if disabled for faster processing)
                if not self.is_video_mode or SHOW_DISPLAY_WINDOW:
                    cv2.imshow("Temi 5-Layer Face-First Detection", display_frame)
                
                # Handle user input (minimal wait if no display)
                if not self.is_video_mode or SHOW_DISPLAY_WINDOW:
                    key = cv2.waitKey(1) & 0xFF
                else:
                    # In video mode without display, check for keyboard less frequently
                    key = cv2.waitKey(1) & 0xFF if self.frame_count % 10 == 0 else 0xFF
                    
                if not self._handle_user_input(key, results):
                    break
                
                # Performance tracking
                self._update_performance_metrics()
                
        except KeyboardInterrupt:
            print("\n⏹ Interrupted by user")
        except Exception as e:
            print(f"\n❌ Error in main loop: {e}")
        finally:
            self._cleanup()
    
    def _process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Process single frame through complete detection and behavior pipeline.
        
        Args:
            frame: Input camera frame
            
        Returns:
            Dictionary containing all processing results
        """
        start_time = time.time()
        now = time.time()
        
        # =================================================================
        # DETECTION PIPELINE: Face-first approach
        # =================================================================
        
        # Face detection and pose estimation
        pose_data = self.face_estimator.estimate(frame)
        
        # =================================================================
        # BEHAVIOR PROCESSING: Gaze classification and behavior management
        # =================================================================
        
        # Process through behavior coordinator
        behavior_results = self.behavior_coord.process_frame_data(pose_data, now)
        
        # =================================================================
        # SUBSUMPTION ARCHITECTURE: Layer coordination
        # =================================================================
        
        # Prepare sensor data (simulated for demo - replace with real Temi sensors)
        sensor_data = self._get_sensor_data(now)
        
        # Prepare gaze data for subsumption layers
        gaze_data = {
            'person_detected': behavior_results['pose_valid'],
            'estimated_distance': behavior_results['distance_estimate'],
            'sustained_gaze': behavior_results['sustained_gaze'],
            'looking': behavior_results['looking']
        }
        
        # Update all subsumption layers
        subsumption_results = self.subsumption.update_all_layers(
            sensor_data, gaze_data, now
        )
        
        # =================================================================
        # COMPILE RESULTS
        # =================================================================
        
        processing_time = time.time() - start_time
        
        return {
            'frame': frame,
            'pose_data': pose_data,
            'behavior_results': behavior_results,
            'subsumption_results': subsumption_results,
            'sensor_data': sensor_data,
            'processing_time': processing_time,
            'timestamp': now
        }
    
    def _get_sensor_data(self, timestamp: float) -> Dict[str, Any]:
        """
        Get sensor data (simulated for demo - replace with real Temi API calls).
        
        Args:
            timestamp: Current timestamp
            
        Returns:
            Dictionary of sensor readings
        """
        # Simulate battery drain and sensor readings
        # In real implementation, replace with actual Temi sensor API calls
        
        # Simulated battery that drains over time
        elapsed_minutes = (timestamp - self.last_time) / 60.0
        battery_drain_rate = 0.1  # 0.1% per minute
        simulated_battery = max(0, 85.0 - (elapsed_minutes * battery_drain_rate))
        
        # Simulated sensor readings with occasional events
        return {
            'battery_level': simulated_battery,
            'charging': False,
            'collision_detected': False,
            'proximity_warning': self.frame_count % 300 < 10,  # Occasional proximity warning
            'virtual_wall_detected': False,
            'mapping_active': True,
            'obstacle_distance': 2.0 + (self.frame_count % 100) / 50.0,  # Varying distance
            'temi_moving': False,
            'manual_control': False,
            'sensor_health': 'OK'
        }
    
    def _render_frame(self, frame: np.ndarray, results: Dict[str, Any]) -> np.ndarray:
        """
        Render frame with all visualizations and status information.
        
        Args:
            frame: Input frame
            results: Processing results from _process_frame
            
        Returns:
            Frame with visualizations added
        """
        display_frame = frame.copy()
        H, W = display_frame.shape[:2]
        
        pose_data = results['pose_data']
        behavior_results = results['behavior_results']
        subsumption_results = results['subsumption_results']
        
        # =================================================================
        # DRAW FACE DETECTION AND POSE VISUALIZATION
        # =================================================================
        
        if pose_data and pose_data.get('ok'):
            # Extract pose information first (initialize all variables)
            face_conf = pose_data.get('face_detection_confidence', 0.0)
            pitch = pose_data.get('pitch', 0)
            yaw = pose_data.get('yaw', 0)
            roll = pose_data.get('roll', 0)
            distance = pose_data.get('estimated_distance', 0)
            
            # Draw face detection box (without confidence text)
            if 'face_box' in pose_data:
                x1, y1, x2, y2 = pose_data['face_box']
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), COLORS['face_box'], 2)
                # Confidence will be shown in top right corner instead
            
            # Draw facial landmarks
            face_kps = pose_data['face_kps']
            for pt in face_kps:
                x, y = int(pt['x']), int(pt['y'])
                cv2.circle(display_frame, (x, y), 3, COLORS['landmarks'], -1)
            
            # Draw pose axes
            if 'rvec' in pose_data:
                pts2d = pose_data['pts2d']
                nose_point = tuple(pts2d[0].astype(int))
                display_frame = draw_pose_axes(
                    display_frame, pose_data['rvec'], pose_data['tvec'],
                    pose_data['cam_matrix'], pose_data['dist_coeffs'], nose_point
                )
            
            # Display pose information in TOP RIGHT CORNER
            if pitch != 0 or yaw != 0:  # Check if we have actual pose data
                # Calculate top-right position
                info_lines = [
                    f"Face Conf: {face_conf:.2f}",
                    f"Yaw: {yaw:+.1f}°",
                    f"Pitch: {pitch:+.1f}°",
                    f"Roll: {roll:+.1f}°",
                    f"Distance: {distance:.1f}m" if distance and distance < 10 else "Distance: ??"
                ]
                
                # Position in top-right corner
                y_start = 30
                text_scale = 0.7 * TEXT_MULTIPLIER  # Apply multiplier
                text_thickness = max(1, int(2 * TEXT_MULTIPLIER))  # Scale thickness too
                line_spacing = int(25 * TEXT_MULTIPLIER)  # Scale spacing
                
                for i, line in enumerate(info_lines):
                    # Get text size to align right
                    (text_width, text_height), baseline = cv2.getTextSize(
                        line, TEXT_FONT, text_scale, text_thickness
                    )
                    x_pos = W - text_width - 10  # 10 pixels from right edge
                    y_pos = y_start + i * line_spacing
                    
                    # Draw text with thicker stroke for visibility
                    cv2.putText(display_frame, line, (x_pos, y_pos),
                               TEXT_FONT, text_scale, COLORS['status_text'], text_thickness)
        
        # =================================================================
        # DRAW BEHAVIOR STATUS
        # =================================================================
        
        # Gaze status
        gaze_status = "LOOKING" if behavior_results['looking'] else "NOT LOOKING"
        gaze_color = COLORS['success'] if behavior_results['looking'] else COLORS['error']
        
        gaze_scale = TEXT_SCALE * TEXT_MULTIPLIER
        gaze_thickness = max(1, int(TEXT_THICKNESS * TEXT_MULTIPLIER))
        
        cv2.putText(display_frame, f"Gaze: {gaze_status}", (10, 30),
                   TEXT_FONT, gaze_scale, gaze_color, gaze_thickness)
        
        # Sustained gaze indicator - show Approach when sustained
        if behavior_results['sustained_gaze']:
            cv2.putText(display_frame, f"Approach ({behavior_results['gaze_elapsed']:.1f}s)",
                       (10, int(60 * TEXT_MULTIPLIER)), TEXT_FONT, gaze_scale, COLORS['warning'], gaze_thickness)
        
        # =================================================================
        # DRAW SUBSUMPTION LAYER STATUS
        # =================================================================
        
        # Active layer indicator
        active_layer = subsumption_results['active_layer']
        layer_status = subsumption_results['layer_status']
        
        # Layer status display
        layer_y_start = H - int(180 * TEXT_MULTIPLIER)  # Scale position
        layer_title_scale = 0.7 * TEXT_MULTIPLIER       # Apply multiplier to title
        layer_text_scale = 0.6 * TEXT_MULTIPLIER        # Apply multiplier to layer text
        layer_title_thickness = max(1, int(2 * TEXT_MULTIPLIER))
        layer_text_thickness = max(1, int(2 * TEXT_MULTIPLIER))
        layer_spacing = int(25 * TEXT_MULTIPLIER)       # Scale spacing
        
        cv2.putText(display_frame, "SUBSUMPTION LAYERS:", (10, layer_y_start),
                   TEXT_FONT, layer_title_scale, COLORS['status_text'], layer_title_thickness)
        
        for i, (layer_num, info) in enumerate(layer_status.items()):
            y_pos = layer_y_start + int(30 * TEXT_MULTIPLIER) + (i * layer_spacing)
            layer_name = info['name']
            layer_state = info['status']
            
            # Color coding: active layer = orange, suppressed = gray, normal = white
            if layer_num == active_layer:
                color = COLORS['warning']  # Active layer in orange
                text = f"{layer_num}.{layer_name}: {layer_state} (ACTIVE)"
            elif layer_num in subsumption_results.get('priority_suppressed', []):
                color = (100, 100, 100)  # Suppressed layers in gray
                text = f"{layer_num}.{layer_name}: SUPPRESSED"
            else:
                color = COLORS['status_text']
                text = f"{layer_num}.{layer_name}: {layer_state}"
            
            cv2.putText(display_frame, text, (10, y_pos),
                       TEXT_FONT, layer_text_scale, color, layer_text_thickness)
        
        # =================================================================
        # PERFORMANCE TRACKING (internal only)
        # =================================================================
        
        # Calculate FPS for internal tracking (not displayed on frame)
        fps = self._calculate_fps()
        
        return display_frame
    
    def _handle_user_input(self, key: int, results: Dict[str, Any]) -> bool:
        """
        Handle user keyboard input.
        
        Args:
            key: Pressed key code
            results: Current processing results
            
        Returns:
            True to continue running, False to quit
        """
        if key == ord('q'):
            print("\n👋 Quit requested by user")
            return False
        
        elif key == ord('r'):
            print("\n🔄 Resetting behavior manager state...")
            self.behavior_coord.behavior_manager.reset_state()
        
        elif key == ord('s'):
            print("\n📊 STATUS SUMMARY:")
            print("=" * 40)
            self._print_status_summary(results)
        
        elif key == ord('p'):
            self.log_performance = not self.log_performance
            status = "enabled" if self.log_performance else "disabled"
            print(f"\n📈 Performance logging {status}")
        
        return True
    
    def _print_status_summary(self, results: Dict[str, Any]):
        """Print comprehensive status summary."""
        behavior_status = self.behavior_coord.get_status_summary()
        
        print(f"Frame: {self.frame_count}")
        print(f"FPS: {self._calculate_fps():.1f}")
        print(f"Processing Time: {results.get('processing_time', 0)*1000:.1f}ms")
        print()
        
        if results['pose_data']:
            pose = results['pose_data']
            print(f"Face Detection: {pose.get('detection_method', 'none')}")
            print(f"Distance: {pose.get('estimated_distance', 0):.1f}m")
            print(f"Pose: Yaw={pose.get('yaw', 0):.1f}°, Pitch={pose.get('pitch', 0):.1f}°")
        else:
            print("Face Detection: None")
        
        print()
        behavior = results['behavior_results']
        print(f"Gaze Status: {behavior['behavior_state']}")
        print(f"Looking: {behavior['looking']}")
        print(f"Sustained: {behavior['sustained_gaze']}")
        print(f"Elapsed: {behavior.get('gaze_elapsed', 0):.1f}s")
        
        print()
        subsumption = results['subsumption_results']
        print(f"Active Layer: {subsumption['active_layer']}")
        print(f"Behavior Command: {subsumption['behavior_command']}")
        
        print("=" * 40)
    
    def _calculate_fps(self) -> float:
        """Calculate current frame rate."""
        now = time.time()
        dt = now - self.last_time
        fps = 1.0 / max(dt, 1e-6)
        self.last_time = now
        return fps
    
    def _update_performance_metrics(self):
        """Update performance tracking metrics."""
        self.frame_count += 1
        
        if self.log_performance and self.frame_count % 30 == 0:  # Log every 30 frames
            fps = self._calculate_fps()
            self.performance_data.append({
                'frame': self.frame_count,
                'fps': fps,
                'timestamp': time.time()
            })
            
            # Keep only recent data
            if len(self.performance_data) > 100:
                self.performance_data = self.performance_data[-100:]
    
    def _initialize_video_writer(self):
        """Initialize video writer for saving processed output video."""
        try:
            # Get video properties from input
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Use specified output FPS or match input FPS
            output_fps = OUTPUT_VIDEO_FPS if OUTPUT_VIDEO_FPS else fps
            
            # Create output directory if it doesn't exist
            import os
            output_dir = os.path.dirname(OUTPUT_VIDEO_PATH)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"📁 Created output directory: {output_dir}")
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*OUTPUT_VIDEO_CODEC)
            self.video_writer = cv2.VideoWriter(
                OUTPUT_VIDEO_PATH,
                fourcc,
                output_fps,
                (width, height)
            )
            
            if self.video_writer.isOpened():
                print(f"✓ Video writer initialized")
                print(f"  Output: {OUTPUT_VIDEO_PATH}")
                print(f"  Resolution: {width}x{height}")
                print(f"  FPS: {output_fps}")
            else:
                print(f"⚠ Warning: Could not initialize video writer")
                self.video_writer = None
                
        except Exception as e:
            print(f"⚠ Warning: Failed to initialize video writer: {e}")
            self.video_writer = None
    
    def _cleanup(self):
        """Clean up resources before exit."""
        print("\n🧹 Cleaning up...")
        
        # Release video writer if active
        if hasattr(self, 'video_writer') and self.video_writer is not None:
            self.video_writer.release()
            if self.is_video_mode:
                print(f"💾 Output video saved: {OUTPUT_VIDEO_PATH}")
        
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        # Print final statistics
        if self.performance_data:
            avg_fps = sum(d['fps'] for d in self.performance_data) / len(self.performance_data)
            print(f"📊 Average FPS: {avg_fps:.1f}")
        
        print(f"📊 Total frames processed: {self.frame_count}")
        print("✅ Cleanup complete")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """
    Main entry point for the Temi 5-Layer Subsumption Architecture.
    """
    try:
        # Create and run the control center
        control_center = TemiControlCenter()
        control_center.run()
        
    except Exception as e:
        print(f"\n Fatal error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n Temi Control Center terminated")

if __name__ == "__main__":
    main()
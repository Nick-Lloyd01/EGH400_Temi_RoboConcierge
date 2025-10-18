"""
Data Logging Module for Temi 5-Layer Subsumption Architecture

This module handles CSV logging of all detection metrics for quantitative analysis
and report generation.
"""

import csv
import os
import time
from typing import Dict, Any, Optional
from config import *


class DataLogger:
    """
    Handles comprehensive data logging for analysis and reporting.
    
    Logs multiple metrics:
    - Detection accuracy (face, landmarks, pose)
    - Distance estimation
    - Gaze classification
    - Processing performance (FPS, timing)
    - False positive/negative annotations
    """
    
    def __init__(self, csv_path: str):
        """
        Initialize data logger.
        
        Args:
            csv_path: Path to output CSV file
        """
        self.csv_path = csv_path
        self.csv_file = None
        self.csv_writer = None
        self.frame_data = []
        
        # Manual annotation tracking (confusion matrix)
        self.true_positive_count = 0   # Correctly detected face
        self.false_positive_count = 0  # Incorrectly detected (no face present)
        self.false_negative_count = 0  # Missed detection (face present but not detected)
        self.true_negative_count = 0   # Correctly detected no face
        
        # Performance tracking
        self.start_time = time.time()
        self.total_frames = 0
        
        # Initialize CSV file
        self._initialize_csv()
        
        print(f"✓ Data logger initialized: {csv_path}")
    
    def _initialize_csv(self):
        """Create CSV file with headers."""
        # Create directory if it doesn't exist
        csv_dir = os.path.dirname(self.csv_path)
        if csv_dir and not os.path.exists(csv_dir):
            os.makedirs(csv_dir)
            print(f"📁 Created directory: {csv_dir}")
        
        # Open CSV file
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        
        # Write header row
        headers = [
            # Frame info
            'Frame_Number',
            'Timestamp_Sec',
            
            # Detection metrics
            'Face_Detected',
            'Face_Confidence',
            'Landmarks_Detected',
            'Landmark_Count',
            'Pose_Estimated',
            
            # Pose data
            'Yaw_Degrees',
            'Pitch_Degrees',
            'Roll_Degrees',
            
            # Distance estimation
            'Distance_Meters',
            'Face_Area_Percent',
            'Proximity_Status',
            
            # Gaze classification
            'Looking_At_Camera',
            'Gaze_Sustained',
            'Gaze_Duration_Sec',
            
            # Detection method
            'Detection_Method',
            
            # Performance
            'Processing_FPS',
            'Frame_Time_Ms',
            
            # Manual annotations
            'Classification',  # 'True_Positive', 'False_Positive', 'False_Negative', 'True_Negative'
            'Notes'
        ]
        
        self.csv_writer.writerow(headers)
        self.csv_file.flush()
        print(f"✓ CSV headers written")
    
    def log_frame(self, frame_num: int, pose_data: Optional[Dict], 
                  behavior_results: Dict, fps: float, 
                  frame_time_ms: float, classification: str = 'True_Positive'):
        """
        Log data for a single frame.
        
        Args:
            frame_num: Current frame number
            pose_data: Pose estimation results
            behavior_results: Behavior manager results
            fps: Current processing FPS
            frame_time_ms: Time to process this frame in milliseconds
            classification: 'True_Positive', 'False_Positive', or 'False_Negative'
        """
        # Extract data with safe defaults
        timestamp = time.time() - self.start_time
        
        # Detection metrics
        face_detected = pose_data is not None and pose_data.get('ok', False)
        face_conf = pose_data.get('face_detection_confidence', 0.0) if pose_data else 0.0
        landmarks_detected = face_detected and len(pose_data.get('face_kps', [])) > 0
        landmark_count = len(pose_data.get('face_kps', [])) if pose_data else 0
        pose_estimated = face_detected and 'rvec' in pose_data if pose_data else False
        
        # Pose angles
        yaw = pose_data.get('yaw', 0.0) if pose_data else 0.0
        pitch = pose_data.get('pitch', 0.0) if pose_data else 0.0
        roll = pose_data.get('roll', 0.0) if pose_data else 0.0
        
        # Distance and proximity
        distance = pose_data.get('estimated_distance', 0.0) if pose_data else 0.0
        face_area_pct = pose_data.get('face_area_fraction', 0.0) * 100 if pose_data else 0.0
        proximity = behavior_results.get('proximity_status', 'NONE')
        
        # Gaze classification
        looking = behavior_results.get('looking', False)
        sustained = behavior_results.get('sustained_gaze', False)
        gaze_duration = behavior_results.get('gaze_elapsed', 0.0)
        
        # Detection method
        method = pose_data.get('detection_method', 'none') if pose_data else 'none'
        
        # Update classification counts
        if classification == 'False_Positive':
            self.false_positive_count += 1
        elif classification == 'False_Negative':
            self.false_negative_count += 1
        elif classification == 'True_Negative':
            self.true_negative_count += 1
        else:  # True_Positive
            self.true_positive_count += 1
        
        # Create row
        row = [
            frame_num,
            f"{timestamp:.3f}",
            
            'Yes' if face_detected else 'No',
            f"{face_conf:.3f}",
            'Yes' if landmarks_detected else 'No',
            landmark_count,
            'Yes' if pose_estimated else 'No',
            
            f"{yaw:.2f}",
            f"{pitch:.2f}",
            f"{roll:.2f}",
            
            f"{distance:.2f}",
            f"{face_area_pct:.2f}",
            proximity,
            
            'Yes' if looking else 'No',
            'Yes' if sustained else 'No',
            f"{gaze_duration:.2f}",
            
            method,
            
            f"{fps:.1f}",
            f"{frame_time_ms:.1f}",
            
            classification,
            ''  # Notes column for future use
        ]
        
        # Write to CSV
        self.csv_writer.writerow(row)
        self.csv_file.flush()  # Ensure data is written immediately
        
        self.total_frames += 1
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics for the logged session.
        
        Returns:
            Dictionary of summary statistics
        """
        total_time = time.time() - self.start_time
        avg_fps = self.total_frames / total_time if total_time > 0 else 0
        
        total_classifications = (self.true_positive_count + 
                                self.false_positive_count + 
                                self.false_negative_count +
                                self.true_negative_count)
        
        tp_pct = (self.true_positive_count / total_classifications * 100 
                  if total_classifications > 0 else 0)
        fp_pct = (self.false_positive_count / total_classifications * 100 
                  if total_classifications > 0 else 0)
        fn_pct = (self.false_negative_count / total_classifications * 100 
                  if total_classifications > 0 else 0)
        tn_pct = (self.true_negative_count / total_classifications * 100 
                  if total_classifications > 0 else 0)
        
        # Calculate accuracy and precision
        accuracy = ((self.true_positive_count + self.true_negative_count) / 
                   total_classifications * 100 if total_classifications > 0 else 0)
        precision = (self.true_positive_count / 
                    (self.true_positive_count + self.false_positive_count) * 100
                    if (self.true_positive_count + self.false_positive_count) > 0 else 0)
        recall = (self.true_positive_count / 
                 (self.true_positive_count + self.false_negative_count) * 100
                 if (self.true_positive_count + self.false_negative_count) > 0 else 0)
        
        return {
            'total_frames': self.total_frames,
            'total_time_sec': total_time,
            'average_fps': avg_fps,
            'true_positives': self.true_positive_count,
            'false_positives': self.false_positive_count,
            'false_negatives': self.false_negative_count,
            'true_negatives': self.true_negative_count,
            'tp_percentage': tp_pct,
            'fp_percentage': fp_pct,
            'fn_percentage': fn_pct,
            'tn_percentage': tn_pct,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall
        }
    
    def close(self):
        """Close CSV file and print summary."""
        if self.csv_file:
            # Write summary statistics as comments at the end
            self.csv_writer.writerow([])
            self.csv_writer.writerow(['=== SUMMARY STATISTICS ==='])
            
            stats = self.get_summary_stats()
            self.csv_writer.writerow(['Total Frames', stats['total_frames']])
            self.csv_writer.writerow(['Total Time (sec)', f"{stats['total_time_sec']:.2f}"])
            self.csv_writer.writerow(['Average FPS', f"{stats['average_fps']:.2f}"])
            self.csv_writer.writerow([])
            self.csv_writer.writerow(['=== CONFUSION MATRIX ==='])
            self.csv_writer.writerow(['True Positives (TP)', stats['true_positives'], f"{stats['tp_percentage']:.1f}%"])
            self.csv_writer.writerow(['False Positives (FP)', stats['false_positives'], f"{stats['fp_percentage']:.1f}%"])
            self.csv_writer.writerow(['False Negatives (FN)', stats['false_negatives'], f"{stats['fn_percentage']:.1f}%"])
            self.csv_writer.writerow(['True Negatives (TN)', stats['true_negatives'], f"{stats['tn_percentage']:.1f}%"])
            self.csv_writer.writerow([])
            self.csv_writer.writerow(['=== PERFORMANCE METRICS ==='])
            self.csv_writer.writerow(['Accuracy', f"{stats['accuracy']:.2f}%", '(TP+TN)/(TP+FP+FN+TN)'])
            self.csv_writer.writerow(['Precision', f"{stats['precision']:.2f}%", 'TP/(TP+FP)'])
            self.csv_writer.writerow(['Recall/Sensitivity', f"{stats['recall']:.2f}%", 'TP/(TP+FN)'])
            
            self.csv_file.close()
            
            print(f"\n📊 Data logging complete:")
            print(f"  Total frames: {stats['total_frames']}")
            print(f"  Average FPS: {stats['average_fps']:.2f}")
            print(f"\n  Confusion Matrix:")
            print(f"    True Positives:  {stats['true_positives']} ({stats['tp_percentage']:.1f}%)")
            print(f"    False Positives: {stats['false_positives']} ({stats['fp_percentage']:.1f}%)")
            print(f"    False Negatives: {stats['false_negatives']} ({stats['fn_percentage']:.1f}%)")
            print(f"    True Negatives:  {stats['true_negatives']} ({stats['tn_percentage']:.1f}%)")
            print(f"\n  Performance Metrics:")
            print(f"    Accuracy:  {stats['accuracy']:.2f}%")
            print(f"    Precision: {stats['precision']:.2f}%")
            print(f"    Recall:    {stats['recall']:.2f}%")
            print(f"\n  CSV saved: {self.csv_path}")

"""
CSV Results Analysis Tool for Temi Detection System

This script analyzes CSV log files and generates tables/visualizations
for your final report.

Usage:
    python analyze_results.py <csv_file_path>
    
Example:
    python analyze_results.py "Control Tests/Control Result CSV/output_1324_CSV_results.csv"
"""

import pandas as pd
import sys
import os
from pathlib import Path


def load_csv(csv_path):
    """Load CSV file and remove summary rows."""
    try:
        df = pd.read_csv(csv_path)
        # Remove summary statistics rows (they have non-numeric Frame_Number)
        df = df[pd.to_numeric(df['Frame_Number'], errors='coerce').notna()]
        df['Frame_Number'] = df['Frame_Number'].astype(int)
        return df
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return None


def print_confusion_matrix(df):
    """Print confusion matrix analysis."""
    print("\n" + "="*70)
    print("CONFUSION MATRIX ANALYSIS")
    print("="*70)
    
    # Count classifications
    counts = df['Classification'].value_counts()
    total = len(df)
    
    tp = counts.get('True_Positive', 0)
    fp = counts.get('False_Positive', 0)
    fn = counts.get('False_Negative', 0)
    tn = counts.get('True_Negative', 0)
    
    print(f"\n{'Metric':<25} {'Count':<10} {'Percentage'}")
    print("-" * 70)
    print(f"{'True Positives (TP)':<25} {tp:<10} {tp/total*100:>6.2f}%")
    print(f"{'False Positives (FP)':<25} {fp:<10} {fp/total*100:>6.2f}%")
    print(f"{'False Negatives (FN)':<25} {fn:<10} {fn/total*100:>6.2f}%")
    print(f"{'True Negatives (TN)':<25} {tn:<10} {tn/total*100:>6.2f}%")
    print("-" * 70)
    print(f"{'TOTAL':<25} {total:<10} 100.00%")
    
    # Calculate metrics
    accuracy = (tp + tn) / total * 100 if total > 0 else 0
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n{'Performance Metrics':<25} {'Value'}")
    print("-" * 70)
    print(f"{'Accuracy':<25} {accuracy:>6.2f}%")
    print(f"{'Precision':<25} {precision:>6.2f}%")
    print(f"{'Recall/Sensitivity':<25} {recall:>6.2f}%")
    print(f"{'Specificity':<25} {specificity:>6.2f}%")
    print(f"{'F1 Score':<25} {f1_score:>6.2f}%")


def analyze_by_distance(df):
    """Analyze detection accuracy by distance ranges."""
    print("\n" + "="*70)
    print("DETECTION ACCURACY BY DISTANCE")
    print("="*70)
    
    # Create distance bins
    bins = [0, 1, 2, 3, 4, 5, 100]  # 0-1m, 1-2m, 2-3m, 3-4m, 4-5m, 5m+
    labels = ['0-1m', '1-2m', '2-3m', '3-4m', '4-5m', '5m+']
    
    # Only analyze frames where face was detected or should have been detected
    df_detected = df[df['Face_Detected'] == 'Yes'].copy()
    
    if len(df_detected) == 0:
        print("⚠️  No face detections found in data")
        return
    
    df_detected['Distance_Range'] = pd.cut(df_detected['Distance_Meters'], 
                                           bins=bins, labels=labels, right=False)
    
    print(f"\n{'Distance':<12} {'Frames':<10} {'TP':<8} {'FP':<8} {'FN':<8} {'Accuracy':<10} {'Avg Conf'}")
    print("-" * 70)
    
    for distance_range in labels:
        subset = df_detected[df_detected['Distance_Range'] == distance_range]
        
        if len(subset) == 0:
            continue
        
        tp = len(subset[subset['Classification'] == 'True_Positive'])
        fp = len(subset[subset['Classification'] == 'False_Positive'])
        fn = len(subset[subset['Classification'] == 'False_Negative'])
        
        total = len(subset)
        accuracy = tp / total * 100 if total > 0 else 0
        avg_conf = subset['Face_Confidence'].astype(float).mean()
        
        print(f"{distance_range:<12} {total:<10} {tp:<8} {fp:<8} {fn:<8} {accuracy:>6.2f}%     {avg_conf:>6.3f}")


def analyze_performance(df):
    """Analyze processing performance metrics."""
    print("\n" + "="*70)
    print("PROCESSING PERFORMANCE")
    print("="*70)
    
    df['Processing_FPS'] = pd.to_numeric(df['Processing_FPS'], errors='coerce')
    df['Frame_Time_Ms'] = pd.to_numeric(df['Frame_Time_Ms'], errors='coerce')
    
    print(f"\n{'Metric':<30} {'Value'}")
    print("-" * 70)
    print(f"{'Total Frames Processed':<30} {len(df)}")
    print(f"{'Average FPS':<30} {df['Processing_FPS'].mean():>8.2f}")
    print(f"{'Max FPS':<30} {df['Processing_FPS'].max():>8.2f}")
    print(f"{'Min FPS':<30} {df['Processing_FPS'].min():>8.2f}")
    print(f"{'Std Dev FPS':<30} {df['Processing_FPS'].std():>8.2f}")
    print()
    print(f"{'Average Frame Time (ms)':<30} {df['Frame_Time_Ms'].mean():>8.2f}")
    print(f"{'Max Frame Time (ms)':<30} {df['Frame_Time_Ms'].max():>8.2f}")
    print(f"{'Min Frame Time (ms)':<30} {df['Frame_Time_Ms'].min():>8.2f}")


def analyze_gaze_classification(df):
    """Analyze gaze detection results."""
    print("\n" + "="*70)
    print("GAZE CLASSIFICATION ANALYSIS")
    print("="*70)
    
    # Only analyze frames where face was detected
    df_detected = df[df['Face_Detected'] == 'Yes'].copy()
    
    if len(df_detected) == 0:
        print("⚠️  No face detections found in data")
        return
    
    total = len(df_detected)
    looking = len(df_detected[df_detected['Looking_At_Camera'] == 'Yes'])
    sustained = len(df_detected[df_detected['Gaze_Sustained'] == 'Yes'])
    
    print(f"\n{'Metric':<30} {'Count':<10} {'Percentage'}")
    print("-" * 70)
    print(f"{'Total Frames with Face':<30} {total:<10} 100.00%")
    print(f"{'Looking at Camera':<30} {looking:<10} {looking/total*100:>6.2f}%")
    print(f"{'Sustained Gaze':<30} {sustained:<10} {sustained/total*100:>6.2f}%")
    
    # Average gaze duration
    avg_duration = df_detected['Gaze_Duration_Sec'].astype(float).mean()
    max_duration = df_detected['Gaze_Duration_Sec'].astype(float).max()
    
    print(f"\n{'Average Gaze Duration':<30} {avg_duration:>8.2f} seconds")
    print(f"{'Maximum Gaze Duration':<30} {max_duration:>8.2f} seconds")


def analyze_detection_methods(df):
    """Analyze which detection methods were used."""
    print("\n" + "="*70)
    print("DETECTION METHOD COMPARISON")
    print("="*70)
    
    df_detected = df[df['Face_Detected'] == 'Yes'].copy()
    
    if len(df_detected) == 0:
        print("⚠️  No face detections found in data")
        return
    
    methods = df_detected['Detection_Method'].value_counts()
    total = len(df_detected)
    
    print(f"\n{'Method':<25} {'Count':<10} {'Percentage'}")
    print("-" * 70)
    for method, count in methods.items():
        print(f"{method:<25} {count:<10} {count/total*100:>6.2f}%")


def main():
    """Main analysis function."""
    print("\n" + "="*70)
    print("TEMI DETECTION SYSTEM - RESULTS ANALYSIS")
    print("="*70)
    
    # Get CSV file path
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = input("\nEnter CSV file path: ").strip()
    
    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"❌ File not found: {csv_path}")
        return
    
    print(f"\n📊 Analyzing: {csv_path}")
    
    # Load data
    df = load_csv(csv_path)
    if df is None:
        return
    
    print(f"✓ Loaded {len(df)} frames")
    
    # Run all analyses
    print_confusion_matrix(df)
    analyze_by_distance(df)
    analyze_performance(df)
    analyze_gaze_classification(df)
    analyze_detection_methods(df)
    
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE")
    print("="*70)
    print("\nCopy the tables above into your Word document!")
    print("For graphs, run: python create_graphs.py <csv_file_path>\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

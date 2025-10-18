"""
CSV Results Analysis Tool for Temi Detection System

This script analyses CSV log files and generates separate CSV files for each analysis table.
Also prints tables to console for quick reference.

Usage:
    python analyse_results.py <csv_file_path>
    
Example:
    python analyse_results.py "Control Tests/Control Result CSV/1324/output_1324_CSV_results.csv"
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


def save_confusion_matrix(df, output_dir, base_name):
    """Save confusion matrix to CSV."""
    counts = df['Classification'].value_counts()
    total = len(df)
    
    tp = counts.get('True_Positive', 0)
    fp = counts.get('False_Positive', 0)
    fn = counts.get('False_Negative', 0)
    tn = counts.get('True_Negative', 0)
    
    # Calculate metrics
    accuracy = (tp + tn) / total * 100 if total > 0 else 0
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Create DataFrame
    confusion_data = {
        'Metric': ['True Positives (TP)', 'False Positives (FP)', 'False Negatives (FN)', 
                   'True Negatives (TN)', 'TOTAL', '', 
                   'Accuracy', 'Precision', 'Recall/Sensitivity', 'Specificity', 'F1 Score'],
        'Count': [tp, fp, fn, tn, total, '', '', '', '', '', ''],
        'Percentage': [f'{tp/total*100:.2f}%', f'{fp/total*100:.2f}%', f'{fn/total*100:.2f}%',
                      f'{tn/total*100:.2f}%', '100.00%', '',
                      f'{accuracy:.2f}%', f'{precision:.2f}%', f'{recall:.2f}%', 
                      f'{specificity:.2f}%', f'{f1_score:.2f}%']
    }
    
    confusion_df = pd.DataFrame(confusion_data)
    output_path = os.path.join(output_dir, f'{base_name}_confusion_matrix.csv')
    confusion_df.to_csv(output_path, index=False)
    print(f"  ✓ Saved: {os.path.basename(output_path)}")
    
    return confusion_df


def save_distance_analysis(df, output_dir, base_name):
    """Save distance vs accuracy analysis to CSV."""
    bins = [0, 1, 2, 3, 4, 5, 100]
    labels = ['0-1m', '1-2m', '2-3m', '3-4m', '4-5m', '5m+']
    
    df_detected = df[df['Face_Detected'] == 'Yes'].copy()
    
    if len(df_detected) == 0:
        print("  ⚠️  No face detections found")
        return None
    
    df_detected['Distance_Range'] = pd.cut(df_detected['Distance_Meters'], 
                                           bins=bins, labels=labels, right=False)
    
    distance_data = []
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
        
        distance_data.append({
            'Distance_Range': distance_range,
            'Total_Frames': total,
            'True_Positives': tp,
            'False_Positives': fp,
            'False_Negatives': fn,
            'Accuracy_%': f'{accuracy:.2f}',
            'Average_Confidence': f'{avg_conf:.3f}'
        })
    
    distance_df = pd.DataFrame(distance_data)
    output_path = os.path.join(output_dir, f'{base_name}_distance_analysis.csv')
    distance_df.to_csv(output_path, index=False)
    print(f"  ✓ Saved: {os.path.basename(output_path)}")
    
    return distance_df


def save_performance_metrics(df, output_dir, base_name):
    """Save processing performance to CSV."""
    df['Processing_FPS'] = pd.to_numeric(df['Processing_FPS'], errors='coerce')
    df['Frame_Time_Ms'] = pd.to_numeric(df['Frame_Time_Ms'], errors='coerce')
    
    performance_data = {
        'Metric': ['Total Frames Processed', 'Average FPS', 'Max FPS', 'Min FPS', 
                   'Std Dev FPS', '', 'Average Frame Time (ms)', 'Max Frame Time (ms)', 
                   'Min Frame Time (ms)'],
        'Value': [len(df), f'{df["Processing_FPS"].mean():.2f}', 
                 f'{df["Processing_FPS"].max():.2f}', f'{df["Processing_FPS"].min():.2f}',
                 f'{df["Processing_FPS"].std():.2f}', '',
                 f'{df["Frame_Time_Ms"].mean():.2f}', f'{df["Frame_Time_Ms"].max():.2f}',
                 f'{df["Frame_Time_Ms"].min():.2f}']
    }
    
    performance_df = pd.DataFrame(performance_data)
    output_path = os.path.join(output_dir, f'{base_name}_performance.csv')
    performance_df.to_csv(output_path, index=False)
    print(f"  ✓ Saved: {os.path.basename(output_path)}")
    
    return performance_df


def save_gaze_analysis(df, output_dir, base_name):
    """Save gaze classification analysis to CSV."""
    df_detected = df[df['Face_Detected'] == 'Yes'].copy()
    
    if len(df_detected) == 0:
        print("  ⚠️  No face detections found")
        return None
    
    total = len(df_detected)
    looking = len(df_detected[df_detected['Looking_At_Camera'] == 'Yes'])
    sustained = len(df_detected[df_detected['Gaze_Sustained'] == 'Yes'])
    avg_duration = df_detected['Gaze_Duration_Sec'].astype(float).mean()
    max_duration = df_detected['Gaze_Duration_Sec'].astype(float).max()
    
    gaze_data = {
        'Metric': ['Total Frames with Face', 'Looking at Camera', 'Sustained Gaze', 
                   '', 'Average Gaze Duration (sec)', 'Maximum Gaze Duration (sec)'],
        'Count': [total, looking, sustained, '', '', ''],
        'Percentage': [f'100.00%', f'{looking/total*100:.2f}%', f'{sustained/total*100:.2f}%',
                      '', f'{avg_duration:.2f}', f'{max_duration:.2f}']
    }
    
    gaze_df = pd.DataFrame(gaze_data)
    output_path = os.path.join(output_dir, f'{base_name}_gaze_analysis.csv')
    gaze_df.to_csv(output_path, index=False)
    print(f"  ✓ Saved: {os.path.basename(output_path)}")
    
    return gaze_df


def save_detection_methods(df, output_dir, base_name):
    """Save detection method comparison to CSV."""
    df_detected = df[df['Face_Detected'] == 'Yes'].copy()
    
    if len(df_detected) == 0:
        print("  ⚠️  No face detections found")
        return None
    
    methods = df_detected['Detection_Method'].value_counts()
    total = len(df_detected)
    
    method_data = []
    for method, count in methods.items():
        method_data.append({
            'Detection_Method': method,
            'Count': count,
            'Percentage': f'{count/total*100:.2f}%'
        })
    
    method_df = pd.DataFrame(method_data)
    output_path = os.path.join(output_dir, f'{base_name}_detection_methods.csv')
    method_df.to_csv(output_path, index=False)
    print(f"  ✓ Saved: {os.path.basename(output_path)}")
    
    return method_df


def save_confidence_vs_distance(df, output_dir, base_name):
    """Save confidence vs distance data for scatter plot."""
    df_detected = df[df['Face_Detected'] == 'Yes'].copy()
    
    if len(df_detected) == 0:
        print("  ⚠️  No face detections found")
        return None
    
    # Extract relevant columns
    conf_dist_df = df_detected[['Frame_Number', 'Distance_Meters', 'Face_Confidence', 'Classification']].copy()
    conf_dist_df['Face_Confidence'] = pd.to_numeric(conf_dist_df['Face_Confidence'], errors='coerce')
    conf_dist_df['Distance_Meters'] = pd.to_numeric(conf_dist_df['Distance_Meters'], errors='coerce')
    
    output_path = os.path.join(output_dir, f'{base_name}_conf_vs_distance.csv')
    conf_dist_df.to_csv(output_path, index=False)
    print(f"  ✓ Saved: {os.path.basename(output_path)}")
    
    return conf_dist_df


def print_summary(df):
    """Print quick summary to console."""
    print("\n" + "="*70)
    print("QUICK SUMMARY")
    print("="*70)
    
    counts = df['Classification'].value_counts()
    total = len(df)
    tp = counts.get('True_Positive', 0)
    fp = counts.get('False_Positive', 0)
    fn = counts.get('False_Negative', 0)
    tn = counts.get('True_Negative', 0)
    
    accuracy = (tp + tn) / total * 100 if total > 0 else 0
    
    print(f"Total Frames: {total}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"TP: {tp} | FP: {fp} | FN: {fn} | TN: {tn}")


def analyse_single_file(csv_path):
    """Analyse a single CSV file and generate all output CSVs."""
    print(f"\n📊 Analysing: {csv_path}")
    
    # Load data
    df = load_csv(csv_path)
    if df is None:
        return False
    
    print(f"✓ Loaded {len(df)} frames")
    
    # Get output directory and base filename
    output_dir = os.path.dirname(csv_path)
    filename = os.path.basename(csv_path)
    base_name = filename.replace('_CSV_results.csv', '').replace('.csv', '')
    
    print(f"\nGenerating analysis CSV files...")
    
    # Generate all analysis CSV files
    save_confusion_matrix(df, output_dir, base_name)
    save_distance_analysis(df, output_dir, base_name)
    save_confidence_vs_distance(df, output_dir, base_name)
    save_performance_metrics(df, output_dir, base_name)
    save_gaze_analysis(df, output_dir, base_name)
    save_detection_methods(df, output_dir, base_name)
    
    # Print summary
    print_summary(df)
    
    return True


def main():
    """Main analysis function."""
    print("\n" + "="*70)
    print("TEMI DETECTION SYSTEM - RESULTS ANALYSER")
    print("="*70)
    
    # Get CSV file path
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        
        # Check if file exists
        if not os.path.exists(csv_path):
            print(f"❌ File not found: {csv_path}")
            return
        
        # Analyse single file
        success = analyse_single_file(csv_path)
        
    else:
        print("\nUsage: python analyse_results.py <csv_file_path>")
        print("\nExample:")
        print('  python analyse_results.py "Control Tests/Control Result CSV/1324/output_1324_CSV_results.csv"')
        return
    
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE")
    print("="*70)
    print("\nAll CSV tables have been generated and saved!")
    print("You can now import these into your Word document.\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

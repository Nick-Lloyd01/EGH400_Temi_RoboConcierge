"""
Visualization Generator for Temi Detection System

This script creates publication-ready graphs from CSV log files.
Requires: matplotlib, pandas

Usage:
    python create_graphs.py <csv_file_path>

Example:
    python create_graphs.py "Control Tests/Control Result CSV/output_1324_CSV_results.csv"
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os


def load_csv(csv_path):
    """Load CSV file and remove summary rows."""
    try:
        df = pd.read_csv(csv_path)
        df = df[pd.to_numeric(df['Frame_Number'], errors='coerce').notna()]
        df['Frame_Number'] = df['Frame_Number'].astype(int)
        return df
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return None


def plot_fps_over_time(df, output_dir):
    """Create FPS over time graph."""
    plt.figure(figsize=(12, 6))
    
    df['Processing_FPS'] = pd.to_numeric(df['Processing_FPS'], errors='coerce')
    
    plt.plot(df['Frame_Number'], df['Processing_FPS'], linewidth=0.8, alpha=0.7)
    
    # Add moving average
    window = 30
    if len(df) > window:
        rolling_avg = df['Processing_FPS'].rolling(window=window).mean()
        plt.plot(df['Frame_Number'], rolling_avg, 'r-', linewidth=2, 
                label=f'{window}-frame moving average')
    
    plt.xlabel('Frame Number', fontsize=12)
    plt.ylabel('Processing Speed (FPS)', fontsize=12)
    plt.title('Real-Time Processing Performance', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add average line
    avg_fps = df['Processing_FPS'].mean()
    plt.axhline(y=avg_fps, color='g', linestyle='--', 
               label=f'Average: {avg_fps:.1f} FPS')
    plt.legend()
    
    output_path = os.path.join(output_dir, 'fps_performance.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_confidence_vs_distance(df, output_dir):
    """Create confidence vs distance scatter plot."""
    plt.figure(figsize=(12, 6))
    
    df_detected = df[df['Face_Detected'] == 'Yes'].copy()
    df_detected['Face_Confidence'] = pd.to_numeric(df_detected['Face_Confidence'], errors='coerce')
    df_detected['Distance_Meters'] = pd.to_numeric(df_detected['Distance_Meters'], errors='coerce')
    
    # Color by classification
    colors = {
        'True_Positive': 'green',
        'False_Positive': 'red',
        'False_Negative': 'orange'
    }
    
    for classification, color in colors.items():
        subset = df_detected[df_detected['Classification'] == classification]
        if len(subset) > 0:
            plt.scatter(subset['Distance_Meters'], subset['Face_Confidence'], 
                       c=color, alpha=0.5, s=20, label=classification)
    
    plt.xlabel('Distance (meters)', fontsize=12)
    plt.ylabel('Detection Confidence', fontsize=12)
    plt.title('Detection Confidence vs Distance', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim([0, 1.05])
    
    output_path = os.path.join(output_dir, 'confidence_distance.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_confusion_matrix(df, output_dir):
    """Create confusion matrix visualization."""
    from matplotlib.patches import Rectangle
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Count classifications
    counts = df['Classification'].value_counts()
    tp = counts.get('True_Positive', 0)
    fp = counts.get('False_Positive', 0)
    fn = counts.get('False_Negative', 0)
    tn = counts.get('True_Negative', 0)
    
    # Create confusion matrix
    matrix = np.array([[tp, fn], [fp, tn]])
    
    # Plot with colors
    im = ax.imshow(matrix, cmap='Blues', alpha=0.6)
    
    # Add text annotations
    labels = [['True Positive\n(Correct Detection)', 'False Negative\n(Missed Face)'],
              ['False Positive\n(Wrong Detection)', 'True Negative\n(Correct No-Face)']]
    
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, f'{labels[i][j]}\n\n{matrix[i, j]}',
                          ha="center", va="center", fontsize=11, fontweight='bold')
    
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Face Present', 'No Face Present'], fontsize=12)
    ax.set_yticklabels(['System: Face', 'System: No Face'], fontsize=12)
    ax.set_xlabel('Ground Truth', fontsize=13, fontweight='bold')
    ax.set_ylabel('System Prediction', fontsize=13, fontweight='bold')
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold', pad=20)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Count', fontsize=11)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_accuracy_by_distance(df, output_dir):
    """Create bar chart of accuracy by distance range."""
    plt.figure(figsize=(12, 6))
    
    bins = [0, 1, 2, 3, 4, 5, 100]
    labels = ['0-1m', '1-2m', '2-3m', '3-4m', '4-5m', '5m+']
    
    df_detected = df[df['Face_Detected'] == 'Yes'].copy()
    df_detected['Distance_Range'] = pd.cut(df_detected['Distance_Meters'], 
                                           bins=bins, labels=labels, right=False)
    
    accuracies = []
    ranges = []
    
    for distance_range in labels:
        subset = df_detected[df_detected['Distance_Range'] == distance_range]
        if len(subset) > 0:
            tp = len(subset[subset['Classification'] == 'True_Positive'])
            accuracy = tp / len(subset) * 100
            accuracies.append(accuracy)
            ranges.append(distance_range)
    
    colors = ['green' if acc >= 90 else 'orange' if acc >= 75 else 'red' 
              for acc in accuracies]
    
    bars = plt.bar(ranges, accuracies, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.xlabel('Distance Range', fontsize=12)
    plt.ylabel('Detection Accuracy (%)', fontsize=12)
    plt.title('Detection Accuracy by Distance', fontsize=14, fontweight='bold')
    plt.ylim([0, 105])
    plt.grid(axis='y', alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='green', alpha=0.7, label='≥90% (Excellent)'),
                      Patch(facecolor='orange', alpha=0.7, label='75-90% (Good)'),
                      Patch(facecolor='red', alpha=0.7, label='<75% (Needs Improvement)')]
    plt.legend(handles=legend_elements, loc='lower left')
    
    output_path = os.path.join(output_dir, 'accuracy_by_distance.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_gaze_classification_pie(df, output_dir):
    """Create pie chart of gaze classifications."""
    plt.figure(figsize=(10, 8))
    
    df_detected = df[df['Face_Detected'] == 'Yes'].copy()
    
    looking_yes = len(df_detected[df_detected['Looking_At_Camera'] == 'Yes'])
    looking_no = len(df_detected[df_detected['Looking_At_Camera'] == 'No'])
    
    sizes = [looking_yes, looking_no]
    labels = [f'Looking at Camera\n({looking_yes} frames)', 
              f'Not Looking\n({looking_no} frames)']
    colors = ['#4CAF50', '#FF9800']
    explode = (0.05, 0)
    
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90,
            textprops={'fontsize': 12, 'fontweight': 'bold'})
    
    plt.title('Gaze Classification Distribution', fontsize=14, fontweight='bold', pad=20)
    plt.axis('equal')
    
    output_path = os.path.join(output_dir, 'gaze_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_detection_methods(df, output_dir):
    """Create bar chart of detection methods used."""
    plt.figure(figsize=(10, 6))
    
    df_detected = df[df['Face_Detected'] == 'Yes'].copy()
    methods = df_detected['Detection_Method'].value_counts()
    
    colors_map = {
        'yolo': '#FF6B6B',
        'mediapipe': '#4ECDC4',
        'both': '#95E77D'
    }
    
    colors = [colors_map.get(method.lower(), '#95A5A6') for method in methods.index]
    
    bars = plt.bar(methods.index, methods.values, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels
    for bar, value in zip(bars, methods.values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{value}\n({value/len(df_detected)*100:.1f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.xlabel('Detection Method', fontsize=12)
    plt.ylabel('Number of Frames', fontsize=12)
    plt.title('Detection Method Usage', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    output_path = os.path.join(output_dir, 'detection_methods.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def main():
    """Main function to generate all graphs."""
    print("\n" + "="*70)
    print("TEMI DETECTION SYSTEM - GRAPH GENERATOR")
    print("="*70)
    
    # Get CSV file path
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = input("\nEnter CSV file path: ").strip()
    
    if not os.path.exists(csv_path):
        print(f"❌ File not found: {csv_path}")
        return
    
    print(f"\n📊 Analyzing: {csv_path}")
    
    # Load data
    df = load_csv(csv_path)
    if df is None:
        return
    
    print(f"✓ Loaded {len(df)} frames\n")
    
    # Create output directory
    csv_dir = os.path.dirname(csv_path)
    output_dir = os.path.join(csv_dir, 'graphs')
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}\n")
    
    # Generate all graphs
    print("Generating graphs...")
    print("-" * 70)
    
    try:
        plot_fps_over_time(df, output_dir)
        plot_confidence_vs_distance(df, output_dir)
        plot_confusion_matrix(df, output_dir)
        plot_accuracy_by_distance(df, output_dir)
        plot_gaze_classification_pie(df, output_dir)
        plot_detection_methods(df, output_dir)
        
        print("-" * 70)
        print(f"\n✅ All graphs generated successfully!")
        print(f"📂 Saved to: {output_dir}")
        print("\n" + "="*70)
        
    except Exception as e:
        print(f"\n❌ Error generating graphs: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹ Interrupted by user")
    except ImportError as e:
        print(f"\n❌ Missing required library: {e}")
        print("\nInstall required packages with:")
        print("  pip install matplotlib pandas numpy")

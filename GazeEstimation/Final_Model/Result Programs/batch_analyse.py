"""
Batch CSV Analysis Tool for Temi Detection System

This script automatically finds and analyses all CSV result files in the
Control Result CSV folder structure and generates analysis tables for each.

Usage:
    python batch_analyse.py
    
This will process all CSV files found in:
    Control Tests/Control Result CSV/*/output_*_CSV_results.csv
"""

import os
import glob
from pathlib import Path
import analyse_results


def find_all_csv_files(base_path):
    """Find all CSV result files in the directory structure."""
    # Look for CSV files matching the pattern
    pattern = os.path.join(base_path, '*', 'output_*_CSV_results.csv')
    csv_files = glob.glob(pattern)
    return sorted(csv_files)


def main():
    """Main batch processing function."""
    print("\n" + "="*70)
    print("TEMI DETECTION SYSTEM - BATCH ANALYSER")
    print("="*70)
    
    # Base path for control results
    base_path = 'Control Tests/Control Result CSV'
    
    if not os.path.exists(base_path):
        print(f"\n❌ Directory not found: {base_path}")
        print("Please ensure the 'Control Tests/Control Result CSV' directory exists.")
        return
    
    # Find all CSV files
    csv_files = find_all_csv_files(base_path)
    
    if not csv_files:
        print(f"\n⚠️  No CSV result files found in {base_path}")
        print("\nExpected pattern: Control Tests/Control Result CSV/*/output_*_CSV_results.csv")
        return
    
    print(f"\n✓ Found {len(csv_files)} CSV file(s) to process:\n")
    for i, csv_file in enumerate(csv_files, 1):
        rel_path = os.path.relpath(csv_file)
        print(f"  {i}. {rel_path}")
    
    print("\n" + "="*70)
    
    # Process each CSV file
    success_count = 0
    for csv_file in csv_files:
        try:
            success = analyse_results.analyse_single_file(csv_file)
            if success:
                success_count += 1
        except Exception as e:
            print(f"\n❌ Error processing {csv_file}: {e}")
            continue
    
    # Final summary
    print("\n" + "="*70)
    print("BATCH PROCESSING COMPLETE")
    print("="*70)
    print(f"\n✅ Successfully processed {success_count}/{len(csv_files)} file(s)")
    print("\nAll analysis CSV tables have been generated!")
    print("Check each test folder for the new CSV files:\n")
    print("  - *_confusion_matrix.csv")
    print("  - *_distance_analysis.csv")
    print("  - *_conf_vs_distance.csv")
    print("  - *_performance.csv")
    print("  - *_gaze_analysis.csv")
    print("  - *_detection_methods.csv")
    print("\nYou can now import these into your Word document! 📊\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

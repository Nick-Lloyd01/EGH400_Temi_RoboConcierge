#!/usr/bin/env python3
"""
Test Script for Video Mode Feature

This script demonstrates how to easily test both live camera mode
and video processing mode.

Author: Nicholas Lloyd
Date: October 2025
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_live_camera():
    """Test live camera mode."""
    print("\n" + "="*60)
    print("TESTING: LIVE CAMERA MODE")
    print("="*60)
    
    # Temporarily modify config
    import config
    original_mode = config.VIDEO_MODE
    config.VIDEO_MODE = False
    
    print("Starting live camera mode...")
    print("Press 'q' to quit")
    
    try:
        import main
        main.main()
    finally:
        # Restore original setting
        config.VIDEO_MODE = original_mode
    
    print("\n✅ Live camera mode test complete")


def test_video_processing():
    """Test video processing mode."""
    print("\n" + "="*60)
    print("TESTING: VIDEO PROCESSING MODE")
    print("="*60)
    
    # Temporarily modify config
    import config
    original_mode = config.VIDEO_MODE
    original_input = config.INPUT_VIDEO_PATH
    original_output = config.OUTPUT_VIDEO_PATH
    
    config.VIDEO_MODE = True
    config.INPUT_VIDEO_PATH = 'InputVideos/IMG_1322.MOV'
    config.OUTPUT_VIDEO_PATH = 'ResultsVideos/test_output.mp4'
    
    print(f"Input:  {config.INPUT_VIDEO_PATH}")
    print(f"Output: {config.OUTPUT_VIDEO_PATH}")
    print("Press 'q' to stop early, or wait for completion")
    
    try:
        import main
        main.main()
    finally:
        # Restore original settings
        config.VIDEO_MODE = original_mode
        config.INPUT_VIDEO_PATH = original_input
        config.OUTPUT_VIDEO_PATH = original_output
    
    print("\n✅ Video processing mode test complete")


def show_menu():
    """Show test menu."""
    print("\n" + "="*60)
    print("TEMI VIDEO MODE FEATURE TEST MENU")
    print("="*60)
    print("\n1. Test Live Camera Mode")
    print("2. Test Video Processing Mode")
    print("3. Run Both Tests")
    print("4. Show Current Configuration")
    print("q. Quit")
    print("\nEnter choice: ", end="")


def show_current_config():
    """Display current configuration."""
    import config
    
    print("\n" + "="*60)
    print("CURRENT CONFIGURATION")
    print("="*60)
    print(f"\nMode: {'📹 VIDEO PROCESSING' if config.VIDEO_MODE else '📷 LIVE CAMERA'}")
    print(f"\nVIDEO_MODE = {config.VIDEO_MODE}")
    
    if config.VIDEO_MODE:
        print(f"INPUT_VIDEO_PATH = '{config.INPUT_VIDEO_PATH}'")
        print(f"OUTPUT_VIDEO_PATH = '{config.OUTPUT_VIDEO_PATH}'")
        print(f"OUTPUT_VIDEO_CODEC = '{config.OUTPUT_VIDEO_CODEC}'")
        print(f"OUTPUT_VIDEO_FPS = {config.OUTPUT_VIDEO_FPS}")
    else:
        print(f"CAMERA_INDEX = {config.CAMERA_INDEX}")
        print(f"CAMERA_RESOLUTION = {config.CAMERA_RESOLUTION}")
        print(f"CAMERA_FPS = {config.CAMERA_FPS}")
    
    print("\n" + "="*60)


def main():
    """Main test script."""
    while True:
        show_menu()
        choice = input().strip().lower()
        
        if choice == '1':
            test_live_camera()
        elif choice == '2':
            test_video_processing()
        elif choice == '3':
            print("\nRunning both tests sequentially...")
            test_live_camera()
            print("\n" + "="*60)
            input("Press Enter to continue to video processing test...")
            test_video_processing()
        elif choice == '4':
            show_current_config()
        elif choice == 'q':
            print("\n👋 Goodbye!")
            break
        else:
            print("\n⚠️  Invalid choice. Please try again.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

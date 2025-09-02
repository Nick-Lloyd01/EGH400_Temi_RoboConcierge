#!/usr/bin/env python3
"""
yolo_test.py
A minimal YOLOv8 inference script for image or webcam.
"""

import argparse
import cv2
from ultralytics import YOLO

def run_on_image(model, image_path):
    results = model(image_path)
    results[0].print()
    results[0].show()

def run_on_webcam(model, cam_index=0):
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"Error: cannot open camera {cam_index}")
        return

    window_name = 'YOLOv8 Live'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO and draw boxes
        results = model(frame)
        annotated = results[0].plot()

        # Show result
        cv2.imshow(window_name, annotated)

        # Wait 30 ms and check for 'q'
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Inference Script")
    parser.add_argument(
        '--source',
        type=str,
        default='0',
        help="Path to image/video file or '0' for webcam"
    )
    args = parser.parse_args()

    model = YOLO('yolov8n.pt')

    if args.source == '0':
        run_on_webcam(model, cam_index=0)
    else:
        run_on_image(model, args.source)

if __name__ == "__main__":
    main()

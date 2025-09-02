import os
import cv2
from ultralytics import YOLO

# 1. Load model
model = YOLO('yolov8n.pt')

# 2. Path to your image
img_path = '/Users/nicholaslloyd/Desktop/EGH400/TemiCode/TESTTEST/Photo/FriendHeadTest.jpg'

# 3. Run inference
results = model(img_path)          # returns a list
annotated = results[0].plot()      # numpy array with boxes drawn

# 4. Compute save path
base, ext = os.path.splitext(img_path)
save_path = f"{base}_Results{ext}"

# 5. Write annotated image
cv2.imwrite(save_path, annotated)
print(f"Saved results to: {save_path}")

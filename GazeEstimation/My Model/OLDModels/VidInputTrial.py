import cv2
import os
from ultralytics import YOLO
import mediapipe as mp

# === Configuration ===
INPUT_PATH = '/Users/nicholaslloyd/Desktop/EGH400/TemiCode/TEMITEST/VIDEO/InputVideo/ControlTest.mp4'
OUTPUT_PATH = '/Users/nicholaslloyd/Desktop/EGH400/TemiCode/TEMITEST/VIDEO/OutputVideo/ControlTest_Results.mp4'
FRAME_INTERVAL = 3  # Process every 3rd frame
MAX_FACES = 2
KEY_POINTS = {
    1: "Nose",
    33: "Left eye",
    263: "Right eye",
    61: "Mouth L",
    291: "Mouth R",
    199: "Chin"
}

# === Load Models ===
yolo_model = YOLO('yolov8s.pt')  # Person detector
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    static_image_mode=False,
    max_num_faces=MAX_FACES,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# === Prepare Input Video ===
cap = cv2.VideoCapture(INPUT_PATH)
if not cap.isOpened():
    raise IOError(f"Cannot open video: {INPUT_PATH}")

frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# === Prepare Output Folder & Writer ===
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_w, frame_h))

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % FRAME_INTERVAL != 0:
        frame_idx += 1
        continue

    annotated = frame.copy()
    yolo_results = yolo_model(frame, verbose=False)
    boxes = yolo_results[0].boxes.data

    for det in boxes:
        x1, y1, x2, y2, conf, cls = map(float, det)
        if int(cls) != 0:  # Only class 0 (person)
            continue

        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        person_crop = frame[y1:y2, x1:x2]
        if person_crop.size == 0:
            continue

        rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_crop)

        # Draw person box and label
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(annotated, f"Person {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, label in KEY_POINTS.items():
                    lm = face_landmarks.landmark[idx]
                    x_rel = int(lm.x * (x2 - x1))
                    y_rel = int(lm.y * (y2 - y1))
                    abs_x, abs_y = x1 + x_rel, y1 + y_rel

                    cv2.circle(annotated, (abs_x, abs_y), 3, (0, 255, 0), -1)
                    if label in ["Left eye", "Mouth L"]:
                        label_pos = (abs_x - 60, abs_y - 5)
                    else:
                        label_pos = (abs_x + 5, abs_y - 5)
                    cv2.putText(annotated, label, label_pos,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    out.write(annotated)
    frame_idx += 1

# === Cleanup ===
cap.release()
out.release()
print(f"Processing complete. Output saved to {OUTPUT_PATH}")

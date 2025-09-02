import cv2
from ultralytics import YOLO
import mediapipe as mp

# Load models
yolo_model = YOLO('yolov8n.pt')  # Generic YOLOv8 for person
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    static_image_mode=False,
    max_num_faces=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Key landmarks
key_points = {
    1: "Nose",
    33: "Left eye",
    263: "Right eye",
    61: "Mouth L",
    291: "Mouth R",
    199: "Chin"
}

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_h, frame_w = frame.shape[:2]
    annotated = frame.copy()

    # YOLO detection
    yolo_results = yolo_model(frame, verbose=False)
    boxes = yolo_results[0].boxes.data

    for det in boxes:
        x1, y1, x2, y2, conf, cls = map(float, det)
        if int(cls) != 0:  # Class 0 = person
            continue

        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(annotated, f"Person {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Crop the person region
        person_crop = frame[y1:y2, x1:x2]
        if person_crop.size == 0:
            continue

        rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_crop)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, label in key_points.items():
                    lm = face_landmarks.landmark[idx]
                    x_rel = int(lm.x * (x2 - x1))
                    y_rel = int(lm.y * (y2 - y1))
                    abs_x, abs_y = x1 + x_rel, y1 + y_rel

                    cv2.circle(annotated, (abs_x, abs_y), 3, (0, 255, 0), -1)

                    # Offset label left or right
                    if label in ["Left eye", "Mouth L"]:
                        label_pos = (abs_x - 60, abs_y - 5)
                    else:
                        label_pos = (abs_x + 5, abs_y - 5)

                    cv2.putText(annotated, label, label_pos,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("Person > Face > Keypoints", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import cv2
from ultralytics import YOLO
import mediapipe as mp

# Load YOLOv8
yolo_model = YOLO('yolov8n.pt')

# Load MediaPipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_h, frame_w = frame.shape[:2]
    face_detected = False
    person_detected = False

    # YOLO detection
    yolo_results = yolo_model(frame, verbose=False)  # suppress model output
    for det in yolo_results[0].boxes.data:
        x1, y1, x2, y2, conf, cls = map(float, det)
        if int(cls) == 0:  # class 0 = person
            person_detected = True
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)
            cv2.putText(frame, "Person", (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # Run face mesh only if person detected
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                face_detected = True
                for face_landmarks in results.multi_face_landmarks:
                    x_coords = [int(lm.x * frame_w) for lm in face_landmarks.landmark]
                    y_coords = [int(lm.y * frame_h) for lm in face_landmarks.landmark]
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)

                    # Face box
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
                    cv2.putText(frame, "Face", (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                    # Keypoints
                    key_points = {
                        1: "Nose",
                        33: "Left eye",
                        263: "Right eye",
                        61: "Mouth L",
                        291: "Mouth R",
                        199: "Chin"
                    }
                    for idx, label in key_points.items():
                        pt = face_landmarks.landmark[idx]
                        x, y = int(pt.x * frame_w), int(pt.y * frame_h)
                        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
                        cv2.putText(frame, label, (x + 5, y - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # if person_detected and not face_detected:
    #     print("→ Person detected, but no face landmarks → Not looking at Temi")

    cv2.imshow("YOLO + Face Keypoints", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

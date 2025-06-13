import cv2
import numpy as np
import math
import time
from ultralytics import YOLO
import mediapipe as mp

# --- CONFIG ---
flip_gaze_direction = False  # Set to True if the arrow still points the wrong way

# Load YOLOv8
yolo = YOLO('yolov8s.pt')

# Load MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1, min_detection_confidence=0.5)

# 3D model points for solvePnP
MODEL_POINTS = np.array([
    (   0.0,    0.0,    0.0),    # Nose tip        (landmark 1)
    (   0.0, -63.6, -12.5),      # Chin            (landmark 199)
    (-43.3,  32.7, -26.0),       # Left eye left   (landmark 33)
    ( 43.3,  32.7, -26.0),       # Right eye right (landmark 263)
    (-28.9, -28.9, -24.1),       # Mouth left      (landmark 61)
    ( 28.9, -28.9, -24.1)        # Mouth right     (landmark 291)
], dtype=np.float64)
LANDMARK_IDS = [1, 199, 33, 263, 61, 291]

def rotation_to_euler(rvec):
    R, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2,1], R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.degrees(x), np.degrees(y), np.degrees(z)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera"); exit(1)

ret, frame = cap.read()
h, w = frame.shape[:2]
focal = w
cam_matrix = np.array([[focal, 0, w/2],
                       [0, focal, h/2],
                       [0,     0,   1]], dtype=np.float64)
dist_coeffs = np.zeros((4,1), dtype=np.float64)

print("Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    t0 = time.time()
    predictions = []

    # YOLOv8 detection
    yres = yolo(frame, verbose=False)[0]
    for det in yres.boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = det
        if int(cls) != 0:  # class 0 = person
            continue
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        # Draw person box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(frame, "Person", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Run FaceMesh on whole frame (could crop to person for speed)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                lm = face_landmarks.landmark
                face_kps = []
                for idx in LANDMARK_IDS:
                    x, y = int(lm[idx].x * w), int(lm[idx].y * h)
                    face_kps.append({'x': float(x), 'y': float(y)})
                    cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

                # Face bounding box (for Roboflow printout)
                xs = [pt['x'] for pt in face_kps]
                ys = [pt['y'] for pt in face_kps]
                x_min, x_max = int(min(xs)), int(max(xs))
                y_min, y_max = int(min(ys)), int(max(ys))
                face_width = x_max - x_min
                face_height = y_max - y_min
                face_x = x_min + face_width / 2
                face_y = y_min + face_height / 2

                # Prepare for solvePnP
                pts2d = np.array([[pt['x'], pt['y']] for pt in face_kps], dtype=np.float64)
                ok, rvec, tvec = cv2.solvePnP(
                    MODEL_POINTS, pts2d, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
                if not ok:
                    continue

                # Draw head pose axes
                axis = np.float32([[80,0,0],[0,80,0],[0,0,80]])
                imgpts,_ = cv2.projectPoints(axis, rvec, tvec, cam_matrix, dist_coeffs)
                nose = tuple(pts2d[0].astype(int))
                for i,pt in enumerate(imgpts):
                    pt = tuple(pt.ravel().astype(int))
                    color = [(0,0,255),(0,255,0),(255,0,0)][i]
                    cv2.line(frame, nose, pt, color, 2)

                # Draw gaze vector
                # sign = -1 if flip_gaze_direction else 1
                # gaze3D = np.array([[0, 0, sign * 1000.0]], dtype=np.float64)
                # gaze2D, _ = cv2.projectPoints(gaze3D, rvec, tvec, cam_matrix, dist_coeffs)
                # gpt = tuple(gaze2D[0].ravel().astype(int))
                # cv2.arrowedLine(frame, nose, gpt, (0, 0, 255), 2, tipLength=0.2)

                # Euler angles
                pitch, yaw, roll = rotation_to_euler(rvec)
                cv2.putText(frame, f"Yaw {yaw:+.1f}, Pitch {pitch:+.1f}", (x_min, y_min-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)

                # Collect result for printout
                face_dict = {
                    'face': {
                        'x': float(face_x),
                        'y': float(face_y),
                        'width': float(face_width),
                        'height': float(face_height),
                        'confidence': float(conf),
                        'class': 'face',
                        'class_confidence': None,
                        'class_id': 0,
                        'tracker_id': None,
                        'detection_id': None,
                        'parent_id': None,
                        'landmarks': face_kps
                    },
                    'yaw': float(yaw),
                    'pitch': float(pitch)
                }
                predictions.append(face_dict)

    elapsed = time.time() - t0
    results = [{
        'predictions': predictions,
        'time': elapsed,
        'time_face_det': None,
        'time_gaze_det': None
    }]
    print("\n# TRIAL RESULTS:\n# Gaze detection result:\n", results)

    cv2.imshow("YOLO + Face Keypoints + Gaze", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

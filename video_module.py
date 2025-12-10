import cv2
import mediapipe as mp
import numpy as np
import time
from ultralytics import YOLO
from alert_logger import log_event, video_log_file

mp_face = mp.solutions.face_mesh
mp_detect = mp.solutions.face_detection
face_mesh = mp_face.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_detector = mp_detect.FaceDetection(min_detection_confidence=0.5)
yolo = YOLO('yolov8n.pt')

# head estimation helper
def estimate_head_pose(image, landmarks):
    h, w, _ = image.shape
    image_points = np.array([
        landmarks[1], landmarks[199],
        landmarks[33], landmarks[263],
        landmarks[61], landmarks[291]
    ], dtype="double")

    model_points = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0)
    ])
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))
    success, rvec, tvec = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
    if not success:
        return 0, 0, 0

    rmat, _ = cv2.Rodrigues(rvec)
    proj_matrix = np.hstack((rmat, tvec))
    _, _, _, _, _, _, eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)
    yaw, pitch, roll = eulerAngles.flatten()
    return yaw, pitch, roll

def run_video_detection():
    cap = cv2.VideoCapture(0)
    print("Video detection started...")
    looking_away_start = None
    alert_time = 0
    alert_message = ""

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape

        # face detection
        face_results = face_detector.process(frame_rgb)
        face_count = 0
        if face_results.detections:
            for d in face_results.detections:
                face_count += 1
                box = d.location_data.relative_bounding_box
                x, y, wb, hb = int(box.xmin * w), int(box.ymin * h), int(box.width * w), int(box.height * h)
                cv2.rectangle(frame, (x, y), (x + wb, y + hb), (0, 255, 0), 2)

        if face_count == 0:
            log_event(video_log_file, "No face detected")
            alert_message = "No face detected"
            alert_time = time.time()
        elif face_count > 1:
            log_event(video_log_file, "Multiple faces detected")
            alert_message = "Multiple faces detected"
            alert_time = time.time()

        # head pose estimation
        mesh_results = face_mesh.process(frame_rgb)
        if mesh_results.multi_face_landmarks:
            landmarks = mesh_results.multi_face_landmarks[0]
            points = [(int(l.x * w), int(l.y * h)) for l in landmarks.landmark]
            yaw, pitch, roll = estimate_head_pose(frame, points)
            if abs(yaw) > 35 or abs(pitch) > 25:
                if not looking_away_start:
                    looking_away_start = time.time()
                elif time.time() - looking_away_start > 5:
                    log_event(video_log_file, "Suspicious: Looking away too long")
                    alert_message = "Suspicious: Looking away too long"
                    alert_time = time.time()
            else:
                looking_away_start = None

        # Mobile (YOLOv8)
        results = yolo(frame, verbose=False)
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = box.conf[0]
                if yolo.names[cls].lower() == "cell phone" and conf > 0.5:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(frame, "Phone Detected", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    log_event(video_log_file, "Cheating Detected: Mobile Phone")
                    alert_message = "Cheating Detected: Mobile Phone"
                    alert_time = time.time()

        if time.time() - alert_time < 3:
            cv2.putText(frame, f"{alert_message}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        cv2.imshow("Cheating Detection System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Video detection stopped.")

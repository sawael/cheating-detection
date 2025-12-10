
# Deep Learning Project — Online Exam Cheating Detection System

A real-time system using Computer Vision and Deep Learning to detect cheating behaviors during online exams.
Built with Python, OpenCV, MediaPipe, YOLOv8, WebRTC-VAD, and PyTorch/TensorFlow in Google Colab.

## Overview

This project is designed for a Deep Learning university course project to automatically monitor and detect cheating behavior in online exams.
It integrates video analysis, audio analysis, and object detection to identify suspicious actions like:

- Looking away from the screen too often
- Using a mobile phone
- Presence of multiple people in frame
- Talking during the exam

## Features

| Detection Type | Technique | Tool/Library | Description |
|----------------|------------|---------------|--------------|
| Face Detection | Mediapipe Face Detection | mediapipe | Ensures only one student is visible |
| Head Pose Estimation | PnP + Face Mesh | opencv, mediapipe | Detects if student looks away for >5s |
| Mobile Detection | Object Detection | YOLOv8 | Flags if a phone appears in frame |
| Audio Detection | Voice Activity Detection | webrtcvad, sounddevice | Detects speech during the exam |
| Logging System | CSV File Logging | pandas | Records all suspicious events with timestamps in the CSV files created|

## System Pipeline (طريقة تنفيذ المشروع)

### Phase 1: Video Input
- Input is live cam using whatever camera's connected.

### Phase 2: Face Detection
- Mediapipe ensures that only one face is present in the frame.
- If more than one face → Suspicious activity logged.

### Phase 3: Head Pose Estimation
- Uses facial landmarks (nose, eyes, mouth) to estimate head orientation using OpenCV’s solvePnP() library.
- If the head is turned away from screen for >5 seconds → Looking Away Alert.

### Phase 4: Mobile Detection
- YOLOv8 detects cell phones in real-time.
- If a phone is detected → "Cheating Detected: Mobile Phone" is logged on the CSV.

### Phase 5: Audio Monitoring
- WebRTC Voice Activity Detection (VAD) continuously listens.
- If speech is detected → "Suspicious: Talking detected" is logged on the CSV.

### Phase 6: Alert & Logging
- System displays on-screen alerts and logs all detected events in .csv files with timestamps.

## Project Structure

```
cheating_detection_project/
│
├── main.py                   # Starts video + audio modules
├── video_module.py            # Video analysis: face, head pose, YOLO detection
├── audio_module.py            # Audio analysis: talking detection
├── alert_logger.py            # CSV logging system
├── requirements.txt           # Python libraries used and required
└── models/
     └── yolov8n.pt            # Pre-trained YOLOv8 model (Ultralytics)
```

## Detailed Explanation by Module

### 1. alert_logger.py — Event Logging
Handles all logging of suspicious activities into CSV files.

**Functions:**
- log_event(file, event) → Inserts a timestamp and event message to a CSV file.
- Automatically initializes empty logs (cheating_log.csv, audio_log.csv) with headers.

**Purpose:**
Ensures every detected incident (e.g., "Talking", "Multiple Faces") is stored for review and in a suitable metrics for evaluation.

### 2. audio_module.py — Talking Detection
Detects voice activity in real-time using WebRTC-VAD.

**Pipeline:**
1. Records 1-second audio snippets using sounddevice.
2. Splits each snippet into 30ms frames.
3. WebRTC-VAD checks if frames contain speech.
4. Logs if speech detected.

**Key Functions:**
- detect_talking() → Main detection loop.
- start_audio_thread() → Runs in background using Python threading.

### 3. video_module.py — Face, Head Pose & Mobile Detection

#### Face Detection
- Uses mediapipe.solutions.face_detection to locate all faces in each frame.
- If face count is more than one → Logs suspicious behavior.

#### Head Pose Estimation
- Uses 3D facial saved landmarks from mediapipe.solutions.face_mesh.
- If face movement is continous (looking left/right/down too long) → Logs after 5 seconds.

#### Mobile Detection (YOLOv8)
- Loads pre-trained YOLOv8 model using:
  ```python
  from ultralytics import YOLO
  model = YOLO('yolov8n.pt')
  ```
- Detects bounding boxes for phones with confidence > 0.5 (accuracy of the object shown in the image and the real-time video recording).
- Draws red boxes and logs "Cheating Detected: Mobile Phone".

#### Alerts
Displays visual warnings using OpenCV text overlays for a short period of times.

### 4. main.py — Integration Layer
Runs the entire system by launching both modules:

```python
if __name__ == "__main__":
    start_audio_thread()   # Start talking detection
    run_video_detection()  # Start video monitoring
```

- Audio runs in background (in a thread that does not block the other modules).
- Video runs in the main loop with live display.
- Both modules log independently (seperate files).

## How Head Pose Estimation Works

Head pose is determined by mapping 2D face landmarks to 3D reference points using the Perspective-n-Point (PnP) algorithm:

1. Select key points from Mediapipe (nose tip, eyes, mouth corners).
2. Define their 3D coordinates (relative to a face model).
3. Use cv2.solvePnP() to find rotation vectors to be able to detect all facial features wherever the head is pointing.
4. Compute yaw, pitch, roll to determine gaze direction.

If the head is turned away beyond thresholds (5 seconds in other words) → mark as “looking away”.

## Installation

Clone the repository:
```bash
git clone https://github.com/sawael/cheating-detection.git
cd cheating-detection
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Required Libraries:
```
opencv-python
mediapipe
ultralytics
torch
pandas
sounddevice
webrtcvad
numpy
```

## Usage

Run the program:
```bash
python main.py
```

Keyboard shortcuts:
- Q → Quit video feed.
- Logs automatically saved as cheating_log.csv and audio_log.csv.

## Evaluation Metrics

| Metric | Definition | Purpose |
|--------|-------------|----------|
| Accuracy | Correct detections / Total detections | Overall system correctness |
| Precision | TP / (TP + FP) | Avoiding false positives |
| Recall | TP / (TP + FN) | Detecting all cheating cases |
| F1-Score | Harmonic mean of precision and recall | Balanced measure |

## Output Example

During runtime, the system shows:
- Green box around student’s face
- Red box on detected phone
- Warning text overlay (e.g., “Cheating Detected: Mobile Phone”)
- Real-time detection + CSV logging

## Datasets

### For Mobile Detection:
- Open Images Dataset (Google) — includes “Mobile Phone” category
- Or self-recorded videos for fine-tuning

### For Face & Pose:
- Mediapipe Face Mesh — pre-trained on large datasets, no additional training required


## Authors

Developed by:
Deep Learning Project Team:
- [Al-Saeed Wael]
- [Seif Alaa-eldin]
- [Youssif Sherif]
- [Marawan Arafa]
- [Mohamed Esslam]

## License

This project is released for educational and research use only.
Unauthorized use for surveillance or unethical purposes is strictly prohibited.

## References

1. Mediapipe Documentation - https://developers.google.com/mediapipe
2. Ultralytics YOLOv8 Docs - https://docs.ultralytics.com/
3. WebRTC Voice Activity Detection - https://github.com/wiseman/py-webrtcvad
4. OpenCV SolvePnP - https://docs.opencv.org/master/d9/d0c/group__calib3d.html

## Project Summary

This project demonstrates how deep learning and AI can enhance online proctoring systems by combining vision, audio, and object detection in real time.

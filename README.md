# TASK 4 — TrackSight Object Detection & Tracking

A real-time computer-vision web application using OpenCV for video input, a pre-trained YOLOv8 model for object detection, and a custom SORT implementation for persistent object IDs.

## Features
- Live webcam input.
- Video-file upload (MP4/AVI/MOV/MKV/WEBM).
- Pre-trained YOLOv8n detection.
- Configurable confidence threshold.
- SORT tracker built with Kalman filters + Hungarian assignment + IoU matching.
- Bounding boxes, class labels, confidence, and tracking IDs.
- Real-time MJPEG browser stream.
- Clean responsive UI.

## Run
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python app.py
```
Open `http://127.0.0.1:5003` and press **Start tracking**.

The first launch automatically downloads the configured YOLO weights (`yolov8n.pt`) through Ultralytics. A machine with an NVIDIA GPU can use a CUDA-compatible PyTorch/Ultralytics setup for better performance; CPU inference also works but will be slower.

## Pipeline
```text
Webcam / Video file
        ↓
     OpenCV
        ↓
   YOLOv8 detector
        ↓
 x1,y1,x2,y2,score
        ↓
  SORT association
  ├─ Kalman prediction
  ├─ IoU cost matrix
  └─ Hungarian matching
        ↓
  Persistent Track IDs
        ↓
Annotated MJPEG stream
```

## Important implementation detail
The tracker is implemented locally in `sort_tracker.py`, so there is no hidden dependency on a separate tracking repository. The UI creates a fresh tracker for each stream. Track IDs are therefore meaningful within the current video stream.

## Configuration
Set `YOLO_MODEL` to another Ultralytics-compatible model path/name, for example a custom-trained weights file:
```bash
# Windows PowerShell
$env:YOLO_MODEL="path/to/best.pt"
# Linux/macOS
export YOLO_MODEL="path/to/best.pt"
```

## Technologies
Python, Flask, OpenCV, Ultralytics YOLO, SORT, Kalman filtering, SciPy, FilterPy, HTML/CSS/JavaScript.

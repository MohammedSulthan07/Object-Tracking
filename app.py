import os
import time
import uuid
import threading
from pathlib import Path

import cv2
from flask import Flask, jsonify, render_template, request, send_from_directory, Response
from ultralytics import YOLO

from sort_tracker import Sort

BASE = Path(__file__).resolve().parent
UPLOADS = BASE / "uploads"
UPLOADS.mkdir(exist_ok=True)
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 250 * 1024 * 1024

MODEL_NAME = os.getenv("YOLO_MODEL", "yolov8n.pt")
_model = None
_model_lock = threading.Lock()

NAMES = {}


def get_model():
    global _model, NAMES
    with _model_lock:
        if _model is None:
            _model = YOLO(MODEL_NAME)
            NAMES = _model.names
        return _model


def open_source(source):
    if source == "webcam" or source == "":
        cap = cv2.VideoCapture(0)
    else:
        safe = os.path.basename(source)
        cap = cv2.VideoCapture(str(UPLOADS / safe))
    if not cap.isOpened():
        raise RuntimeError("Could not open video source. Check webcam permissions or the uploaded file.")
    return cap


def frames(source="webcam", conf=0.45):
    cap = open_source(source)
    tracker = Sort(max_age=15, min_hits=2, iou_threshold=0.25)
    model = get_model()
    prev = time.perf_counter()
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            results = model.predict(frame, conf=conf, verbose=False)
            detections = []
            labels = []
            for r in results:
                if r.boxes is None:
                    continue
                xyxy = r.boxes.xyxy.cpu().numpy()
                scores = r.boxes.conf.cpu().numpy()
                classes = r.boxes.cls.cpu().numpy().astype(int)
                for box, score, cls in zip(xyxy, scores, classes):
                    detections.append([*box.tolist(), float(score)])
                    labels.append((box, float(score), int(cls)))
            tracks = tracker.update(detections)
            # Associate each track back to the closest current detection.
            for tr in tracks:
                x1,y1,x2,y2,tid = tr.astype(int)
                best=None; best_iou=0
                for box,score,cls in labels:
                    bx1,by1,bx2,by2 = box
                    ix1,iy1=max(x1,bx1),max(y1,by1); ix2,iy2=min(x2,bx2),min(y2,by2)
                    inter=max(0,ix2-ix1)*max(0,iy2-iy1)
                    area=(x2-x1)*(y2-y1)+(bx2-bx1)*(by2-by1)-inter
                    iou=inter/max(area,1e-6)
                    if iou>best_iou: best_iou=iou; best=(score,cls)
                score,cls=best if best else (.0,0)
                name=NAMES.get(cls, str(cls)) if isinstance(NAMES,dict) else str(cls)
                cv2.rectangle(frame,(x1,y1),(x2,y2),(103,224,195),2)
                cv2.rectangle(frame,(x1,y1),(min(x2,x1+210),y1+27),(20,27,38),-1)
                cv2.putText(frame,f"ID {tid}  {name}  {score:.2f}",(x1+7,y1+19),cv2.FONT_HERSHEY_SIMPLEX,.46,(240,244,250),1,cv2.LINE_AA)

            now=time.perf_counter(); fps=1/max(now-prev,1e-6); prev=now
            cv2.rectangle(frame,(12,12),(224,43),(12,17,28),-1)
            cv2.putText(frame,f"YOLO + SORT   {fps:.1f} FPS",(22,33),cv2.FONT_HERSHEY_SIMPLEX,.55,(220,230,240),1,cv2.LINE_AA)
            ok, buffer=cv2.imencode('.jpg',frame,[int(cv2.IMWRITE_JPEG_QUALITY),82])
            if not ok: continue
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+buffer.tobytes()+b'\r\n'
    finally:
        cap.release()


@app.get("/")
def index():
    return render_template("index.html")

@app.post("/api/upload")
def upload():
    file=request.files.get('video')
    if not file or not file.filename:
        return jsonify({"error":"Choose a video file."}),400
    ext=Path(file.filename).suffix.lower()
    if ext not in {'.mp4','.avi','.mov','.mkv','.webm'}:
        return jsonify({"error":"Unsupported video format."}),400
    name=f"{uuid.uuid4().hex}{ext}"
    file.save(UPLOADS/name)
    return jsonify({"filename":name})

@app.get("/video_feed")
def video_feed():
    source=request.args.get('source','webcam')
    conf=max(.05,min(.95,float(request.args.get('conf','0.45'))))
    return Response(frames(source,conf),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.post("/api/reset")
def reset():
    global _model
    return jsonify({"ok":True,"message":"A new SORT tracker is created for every video stream."})

@app.get('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOADS, filename)

if __name__=='__main__':
    app.run(debug=False,host='0.0.0.0',port=5003,threaded=True)

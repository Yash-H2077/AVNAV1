import cv2, time, sys
sys.path.append('../src')

from objects import YOLODetector, annotate
from video import open_capture, read_frame, close_capture

detector = YOLODetector(weights='yolov8n.pt', imgsz=640, conf=0.25)

source = '../data/videos/road.mp4'
cap = open_capture(source)
prev_t = time.time()

while True:
    frame = read_frame(cap)
    if frame is None: break
    t = time.time()
    dt = t - prev_t
    prev_t = t

    obstacles = detector.predict(frame)
    out = annotate(frame, obstacles)
    fps = 1.0 / max(dt, 1e-3)

    cv2.putText(out, f"Detections: {len(obstacles)}  FPS: {fps:.1f}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)
    cv2.imshow("YOLO Detection", out)
    if cv2.waitKey(1) & 0xFF == 27: break

close_capture(cap)

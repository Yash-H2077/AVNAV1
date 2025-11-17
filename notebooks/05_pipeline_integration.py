import cv2, time, sys
sys.path.append('../src')

from lane import detect_lanes, lateral_error
from objects import YOLODetector, annotate
from control import PID, throttle_rule
from viz import put_status
from video import open_capture, read_frame, close_capture

# Choose source: '../data/videos/road.mp4' or 0 for webcam
source = '../data/videos/road.mp4'

detector = YOLODetector(weights='yolov8n.pt', imgsz=640, conf=0.25)
pid = PID(Kp=0.015, Ki=0.0, Kd=0.005, clamp=(-0.8, 0.8))

cap = open_capture(source)
prev_t = time.time()

while True:
    frame = read_frame(cap)
    if frame is None: break
    t = time.time()
    dt = t - prev_t
    prev_t = t

    # Perception: lanes + objects
    lane_overlay, lane_cx = detect_lanes(frame)
    obstacles = detector.predict(frame)
    annotated = annotate(lane_overlay, obstacles)

    # Control
    err = lateral_error(frame.shape[1], lane_cx)
    steer = pid.update(err, dt)
    throttle = throttle_rule(obstacles, frame.shape)

    # Viz
    fps = 1.0 / max(dt, 1e-3)
    out = put_status(annotated, steer, throttle, fps)
    cv2.imshow("AV Navigation Pipeline", out)

    if cv2.waitKey(1) & 0xFF == 27: break

close_capture(cap)

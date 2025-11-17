# PID demo with lane error
import cv2, time, sys
sys.path.append('../src')

from lane import detect_lanes, lateral_error
from control import PID, throttle_rule
from objects import YOLODetector, annotate

source = '../data/videos/road.mp4'
detector = YOLODetector(weights='yolov8n.pt')

pid = PID(Kp=0.015, Ki=0.0, Kd=0.005, clamp=(-0.8, 0.8))

cap = cv2.VideoCapture(source)
prev_t = time.time()

while True:
    ret, frame = cap.read()
    if not ret: break
    t = time.time()
    dt = t - prev_t
    prev_t = t

    lane_overlay, lane_cx = detect_lanes(frame)
    err = lateral_error(frame.shape[1], lane_cx)
    steer = pid.update(err, dt)

    obstacles = detector.predict(frame)
    out = annotate(lane_overlay, obstacles)
    throttle = throttle_rule(obstacles, frame.shape)

    fps = 1.0 / max(dt, 1e-3)
    cv2.putText(out, f"Steer: {steer:+.2f}  Throttle: {throttle:.2f}  FPS: {fps:.1f}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)
    cv2.imshow("Planning + Control", out)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()

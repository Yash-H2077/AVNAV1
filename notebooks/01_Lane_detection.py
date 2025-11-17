import cv2, time
import numpy as np
import sys
sys.path.append(r"C:\Users\yashh\Desktop\avnav\src")

from lane import detect_lanes, lateral_error
from video import open_capture, read_frame, close_capture

# Change to your video path or 0 for webcam
source = '../data/videos/road.mp4'

cap = open_capture(source)
prev_t = time.time()

while True:
    frame = read_frame(cap)
    if frame is None: break
    t = time.time()
    dt = t - prev_t
    prev_t = t

    overlay, lane_cx = detect_lanes(frame)
    err = lateral_error(frame.shape[1], lane_cx)
    fps = 1.0 / max(dt, 1e-3)

    cv2.putText(overlay, f"Lane CX: {lane_cx}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    cv2.putText(overlay, f"Error: {err:+.2f}  FPS: {fps:.1f}", (20,80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    cv2.imshow("Lane Detection", overlay)
    if cv2.waitKey(1) & 0xFF == 27: break

close_capture(cap)

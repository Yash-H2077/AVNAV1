import cv2

def put_status(frame, steer, throttle, fps):
    out = frame.copy()
    cv2.putText(out, f"Steer: {steer:+.2f}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)
    cv2.putText(out, f"Throttle: {throttle:.2f}", (20,80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)
    cv2.putText(out, f"FPS: {fps:.1f}", (20,120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)
    return out

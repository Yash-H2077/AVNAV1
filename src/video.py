import cv2

def open_capture(source=0):
    """
    source: path to video file or int for webcam
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")
    return cap

def read_frame(cap):
    ret, frame = cap.read()
    if not ret:
        return None
    return frame

def close_capture(cap):
    cap.release()
    cv2.destroyAllWindows()

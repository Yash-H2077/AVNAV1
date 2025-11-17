from ultralytics import YOLO
import cv2

class YOLODetector:
    def __init__(self, weights='yolov8n.pt', imgsz=640, conf=0.25):
        self.model = YOLO(weights)
        self.imgsz = imgsz
        self.conf = conf

    def predict(self, frame):
        results = self.model.predict(source=frame, imgsz=self.imgsz, conf=self.conf, verbose=False)
        obstacles = []
        for r in results:
            for box in r.boxes:
                x1,y1,x2,y2 = box.xyxy[0].cpu().numpy().astype(int)
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                obstacles.append(((x1,y1,x2,y2), cls, conf))
        return obstacles

def annotate(frame, obstacles, color=(255,0,0)):
    out = frame.copy()
    for (x1,y1,x2,y2), cls, conf in obstacles:
        cv2.rectangle(out, (x1,y1), (x2,y2), color, 2)
        cv2.putText(out, f"{cls}:{conf:.2f}", (x1, max(0,y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return out

def forward_roi(box, frame_shape):
    h, w = frame_shape[:2]
    x1,y1,x2,y2 = box
    cx = (x1+x2)//2
    return (y2 > int(0.6*h)) and (abs(cx - w//2) < int(0.2*w))

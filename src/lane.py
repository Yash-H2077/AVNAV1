import cv2
import numpy as np

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, [np.array(vertices, dtype=np.int32)], 255)
    return cv2.bitwise_and(img, mask)

def detect_lanes(frame):
    """
    Classical lane detector (Canny + Hough) returning overlay and lane center.
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)

    roi_vertices = [(0, h), (w, h), (w, int(h*0.6)), (0, int(h*0.6))]
    edges_roi = region_of_interest(edges, roi_vertices)

    lines = cv2.HoughLinesP(edges_roi, 1, np.pi/180, threshold=50,
                            minLineLength=40, maxLineGap=100)

    left, right = [], []
    if lines is not None:
        for x1,y1,x2,y2 in lines[:,0]:
            slope = (y2 - y1) / (x2 - x1 + 1e-6)
            if slope < -0.5:
                left.append((x1,y1,x2,y2))
            elif slope > 0.5:
                right.append((x1,y1,x2,y2))

    def average_line(points):
        if not points: return None
        xs = [p[0] for p in points] + [p[2] for p in points]
        ys = [p[1] for p in points] + [p[3] for p in points]
        poly = np.polyfit(xs, ys, 1)  # y = m x + b
        m, b = poly
        y1, y2 = h, int(h*0.6)
        x1, x2 = int((y1 - b)/m), int((y2 - b)/m)
        return (x1, y1, x2, y2)

    left_line = average_line(left)
    right_line = average_line(right)

    overlay = frame.copy()
    if left_line:
        cv2.line(overlay, left_line[:2], left_line[2:], (0,255,0), 6)
    if right_line:
        cv2.line(overlay, right_line[:2], right_line[2:], (0,255,0), 6)

    lane_center_x = w//2
    if left_line and right_line:
        lane_center_x = (left_line[0] + right_line[0]) // 2

    return overlay, lane_center_x

def lateral_error(frame_width, lane_center_x):
    """
    Normalized lateral error in [-1,1], positive means vehicle center is right of lane center.
    """
    center_x = frame_width // 2
    return (center_x - lane_center_x) / (frame_width / 2 + 1e-6)

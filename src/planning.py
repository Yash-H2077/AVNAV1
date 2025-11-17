import numpy as np

def pure_pursuit(cur_pose, path_points, Ld=5.0):
    """
    cur_pose: (x, y, yaw radians)
    path_points: list of (x,y)
    returns curvature kappa and chosen lookahead point
    """
    cx, cy, yaw = cur_pose
    dists = [np.hypot(px - cx, py - cy) for px, py in path_points]
    if not dists:
        return 0.0, (cx, cy)
    idx = int(np.argmin(dists))
    while idx < len(path_points) and np.hypot(path_points[idx][0]-cx, path_points[idx][1]-cy) < Ld:
        idx += 1
    if idx >= len(path_points): idx = len(path_points)-1
    px, py = path_points[idx]

    dx, dy = px - cx, py - cy
    x_v =  np.cos(-yaw)*dx - np.sin(-yaw)*dy
    y_v =  np.sin(-yaw)*dx + np.cos(-yaw)*dy

    kappa = 2.0 * y_v / (Ld**2 + 1e-6)
    return kappa, (px, py)

def steer_from_curvature(kappa, wheelbase=2.7):
    """
    Simple mapping from curvature to steering angle (radians) for demo.
    """
    return np.arctan(kappa * wheelbase)

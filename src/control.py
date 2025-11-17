class PID:
    def __init__(self, Kp=0.015, Ki=0.0, Kd=0.005, clamp=(-0.8, 0.8)):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.integral = 0.0
        self.prev_error = 0.0
        self.min_out, self.max_out = clamp

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, error, dt):
        self.integral += error * dt
        deriv = (error - self.prev_error) / (dt + 1e-6)
        self.prev_error = error
        out = self.Kp * error + self.Ki * self.integral + self.Kd * deriv
        return max(self.min_out, min(self.max_out, out))

def throttle_rule(obstacles, frame_shape, base=0.4, slow=0.1):
    from .objects import forward_roi
    danger = any(forward_roi(b, frame_shape) for b, cls, conf in obstacles)
    return slow if danger else base

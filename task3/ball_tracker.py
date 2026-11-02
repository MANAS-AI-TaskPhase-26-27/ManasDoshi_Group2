import argparse
import csv
import glob
import os
import re
import sys
from collections import deque

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def find_file(given, patterns, what):
    candidates = []
    if given:
        candidates += [given, os.path.join(BASE_DIR, given)]
    for pattern in patterns:
        candidates += sorted(glob.glob(os.path.join(BASE_DIR, pattern)))
        candidates += sorted(glob.glob(pattern))
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError("Could not find the " + what + " file. Put it in the same folder as ball_tracker.py or pass its path.")


def read_number(text, key):
    match = re.search(r"^\s*" + key + r"\s*:\s*([-+0-9.eE]+)", text, re.M)
    if match is None:
        raise ValueError("Missing '" + key + "' in the calibration file")
    return float(match.group(1))


def read_matrix(text, key):
    pattern = key + r"\s*:\s*!!opencv-matrix\s+rows:\s*(\d+)\s+cols:\s*(\d+)\s+dt:\s*\S+\s+data:\s*\[(.*?)\]"
    match = re.search(pattern, text, re.S)
    if match is None:
        return None
    rows, cols = int(match.group(1)), int(match.group(2))
    values = [float(v) for v in match.group(3).replace("\n", " ").split(",") if v.strip()]
    if len(values) != rows * cols:
        raise ValueError("Matrix '" + key + "' has " + str(len(values)) + " numbers, expected " + str(rows * cols))
    return np.array(values, dtype=np.float64).reshape(rows, cols)


def load_calibration(path):
    with open(path, "r") as f:
        text = f.read()
    K = read_matrix(text, "camera_matrix")
    D = read_matrix(text, "distortion_coefficients")
    R = read_matrix(text, "rectification_matrix")
    if K is None or D is None:
        raise ValueError("camera_matrix or distortion_coefficients not found in " + path)
    if R is None:
        R = np.eye(3)
    size = (int(read_number(text, "image_width")), int(read_number(text, "image_height")))
    return K, D, R, size


def build_undistort_maps(K, D, R, calib_size, frame_size):
    K = K.copy()
    K[0, :] *= frame_size[0] / calib_size[0]
    K[1, :] *= frame_size[1] / calib_size[1]
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, frame_size, alpha=0)
    return cv2.initUndistortRectifyMap(K, D, R, new_K, frame_size, cv2.CV_16SC2)


BALL_LO = np.array([11, 90, 185])
BALL_HI = np.array([22, 255, 255])
STRICT_V = 232
BODY_LO = np.array([10, 90, 100])
BODY_HI = np.array([30, 255, 255])
BALL_AREA = (35, 600)
BODY_MIN_AREA = 900


def ball_candidates(frame, prev_gray):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    loose = cv2.inRange(hsv, BALL_LO, BALL_HI)
    loose = cv2.morphologyEx(loose, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    strict = cv2.inRange(hsv, (12, 100, STRICT_V), (21, 255, 255))

    body = cv2.inRange(hsv, BODY_LO, BODY_HI)
    body = cv2.morphologyEx(body, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(body)
    big_ids = [i for i in range(1, n) if st[i, 4] > BODY_MIN_AREA]
    players = np.isin(lab, big_ids).astype(np.uint8) * 255
    players = cv2.dilate(players, np.ones((9, 9), np.uint8))

    gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    motion = cv2.absdiff(gray, prev_gray)
    _, motion = cv2.threshold(motion, 12, 255, cv2.THRESH_BINARY)
    motion = cv2.dilate(motion, np.ones((9, 9), np.uint8))

    contours, _ = cv2.findContours(loose, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        area = cv2.contourArea(c)
        if not (BALL_AREA[0] <= area <= BALL_AREA[1]):
            continue
        x, y, w, h = cv2.boundingRect(c)
        if not (0.45 <= w / float(h) <= 2.2) or area / float(w * h) < 0.45:
            continue
        if cv2.countNonZero(motion[y:y + h, x:x + w]) < 0.3 * w * h:
            continue
        overlap = players[y:y + h, x:x + w].mean() / 255.0
        is_strict = cv2.countNonZero(strict[y:y + h, x:x + w]) > 0.25 * area
        (cx, cy), r = cv2.minEnclosingCircle(c)
        perim = cv2.arcLength(c, True)
        circ = 4 * np.pi * area / (perim * perim + 1e-6)
        out.append((cx, cy, r, circ, overlap, is_strict))
    return out, gray


class BallTracker:
    MAX_SPEED = 90
    MAX_MISSES = 12

    def __init__(self):
        kf = cv2.KalmanFilter(4, 2)
        kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                        [0, 1, 0, 1],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], np.float32)
        kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                         [0, 1, 0, 0]], np.float32)
        kf.processNoiseCov = np.diag([1, 1, 8, 8]).astype(np.float32)
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 4
        kf.errorCovPost = np.eye(4, dtype=np.float32) * 50
        self.kf = kf
        self.active = False
        self.misses = 0
        self.age = 0
        self.history = deque(maxlen=10)
        self.pending = []
        self.overlap_run = 0
        self.radius = 10

    def _drop(self):
        self.active = False
        self.pending = []

    def update(self, cands):
        if not self.active:
            return self._try_acquire(cands)

        pred = self.kf.predict()
        px, py = float(pred[0, 0]), float(pred[1, 0])
        gate = min(40 + 15 * self.misses, 130)

        best, best_cost = None, 1e9
        for cx, cy, r, circ, overlap, strict in cands:
            d = np.hypot(cx - px, cy - py)
            if overlap > 0.15:
                if self.misses > 3 or d > 22:
                    continue
            elif d >= gate:
                continue
            cost = d - 8 * circ + 25 * overlap - (6 if strict else 0)
            if cost < best_cost:
                best, best_cost = (cx, cy, r, overlap), cost
        self.age += 1
        if best is not None:
            self.kf.correct(np.array([[best[0]], [best[1]]], np.float32))
            self.misses, self.radius = 0, best[2]
            self.history.append(1)
            self.overlap_run = self.overlap_run + 1 if best[3] > 0.15 else 0
            if self.overlap_run > 4:
                self._drop()
                return None
            return (float(self.kf.statePost[0, 0]), float(self.kf.statePost[1, 0]), "detected")

        self.misses += 1
        self.history.append(0)
        if self.misses > self.MAX_MISSES or (self.age < 10 and sum(self.history) < 0.5 * len(self.history)):
            self._drop()
            return None
        return (px, py, "predicted")

    def _try_acquire(self, cands):
        cands = [c for c in cands if c[5] and c[4] < 0.15]
        if not cands:
            self.pending = []
            return None
        if self.pending:
            lx, ly = self.pending[-1][:2]
            near = [c for c in cands if 3 <= np.hypot(c[0] - lx, c[1] - ly) <= self.MAX_SPEED]
            if len(self.pending) >= 2:
                (x0, y0), (x1, y1) = self.pending[-2][:2], self.pending[-1][:2]
                vx, vy = x1 - x0, y1 - y0
                near = [c for c in near if np.hypot(c[0] - (x1 + vx), c[1] - (y1 + vy)) < 30]
            if near:
                self.pending.append(min(near, key=lambda c: np.hypot(c[0] - lx, c[1] - ly)))
            else:
                self.pending = [max(cands, key=lambda t: t[3])]
        else:
            self.pending = [max(cands, key=lambda t: t[3])]

        if len(self.pending) >= 3:
            (x0, y0), (x1, y1), (x2, y2) = [p[:2] for p in self.pending[-3:]]
            self.kf.statePost = np.array([[x2], [y2], [x2 - x1], [y2 - y1]], np.float32)
            self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 10
            self.active, self.misses, self.age, self.overlap_run = True, 0, 0, 0
            self.history.clear()
            self.radius = self.pending[-1][2]
            self.pending = []
            return (x2, y2, "detected")
        return None


def can_show():
    if sys.platform.startswith("linux"):
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return True


def draw_frame(frame, positions, idx, trail_len):
    H, W = frame.shape[:2]
    vis = frame.copy()
    bar = vis.copy()
    cv2.rectangle(bar, (0, 48), (W, 84), (0, 0, 0), -1)
    vis = cv2.addWeighted(bar, 0.5, vis, 0.5, 0)
    cv2.putText(vis, "Team 1: 6 players        Team 2: 6 players", (W // 2 - 270, 74),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    start = max(0, idx - trail_len + 1)
    for i in range(start + 1, idx + 1):
        if i in positions and (i - 1) in positions:
            t = (i - start) / float(idx - start + 1)
            p0 = (int(positions[i - 1][0]), int(positions[i - 1][1]))
            p1 = (int(positions[i][0]), int(positions[i][1]))
            cv2.line(vis, p0, p1, (0, int(255 * t), 255), max(1, int(1 + 3 * t)), cv2.LINE_AA)
    if idx in positions:
        x, y, status, radius = positions[idx]
        colour = (0, 255, 0) if status == "detected" else (0, 200, 255)
        cv2.circle(vis, (int(x), int(y)), int(max(radius, 8) + 6), colour, 2, cv2.LINE_AA)
    return vis


def track_video(video_path, map1, map2, H, W, trail_len, show):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("OpenCV could not open the video: " + video_path)
    tracker = BallTracker()
    positions = {}
    candidates = []
    prev_gray = None
    idx = 0
    while True:
        ok, raw = cap.read()
        if not ok:
            break
        frame = cv2.remap(raw, map1, map2, cv2.INTER_LINEAR)
        if prev_gray is None:
            prev_gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        cands, gray = ball_candidates(frame, prev_gray)
        prev_gray = gray
        candidates.append(cands)
        state = tracker.update(cands)
        if state is not None and -20 <= state[1] <= H and 0 <= state[0] <= W:
            positions[idx] = (state[0], state[1], state[2], tracker.radius)
        if show:
            try:
                cv2.imshow("Live ball tracking (press q to close preview)", draw_frame(frame, positions, idx, trail_len))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    show = False
                    cv2.destroyAllWindows()
            except cv2.error:
                show = False
        idx += 1
        if idx % 300 == 0:
            print("tracking: processed", idx, "frames")
    cap.release()
    if show:
        cv2.destroyAllWindows()
    return positions, candidates, idx


def velocity(positions, frame, direction):
    for k in range(1, 5):
        other = frame + direction * k
        if other in positions:
            return ((positions[other][0] - positions[frame][0]) / (direction * k),
                    (positions[other][1] - positions[frame][1]) / (direction * k))
    return None


def plan_gaps(positions, max_gap=30):
    plan = {}
    frames = sorted(positions)
    for a, b in zip(frames, frames[1:]):
        gap = b - a - 1
        if gap < 1 or gap > max_gap:
            continue
        x0, y0, _, r0 = positions[a]
        x1, y1, _, r1 = positions[b]
        span = float(b - a)
        if np.hypot(x1 - x0, y1 - y0) > BallTracker.MAX_SPEED * span:
            continue
        v0 = velocity(positions, a, -1) or ((x1 - x0) / span, (y1 - y0) / span)
        v1 = velocity(positions, b, 1) or ((x1 - x0) / span, (y1 - y0) / span)
        for i in range(a + 1, b):
            t = (i - a) / span
            h00, h10 = 2 * t ** 3 - 3 * t ** 2 + 1, t ** 3 - 2 * t ** 2 + t
            h01, h11 = -2 * t ** 3 + 3 * t ** 2, t ** 3 - t ** 2
            x = h00 * x0 + h10 * span * v0[0] + h01 * x1 + h11 * span * v1[0]
            y = h00 * y0 + h10 * span * v0[1] + h01 * y1 + h11 * span * v1[1]
            lx, ly = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
            if np.hypot(x - lx, y - ly) > 120:
                x, y = lx, ly
            plan[i] = (x, y, r0 + t * (r1 - r0), gap)
    return plan


def local_ball(frame, x, y, search=28):
    H, W = frame.shape[:2]
    if not (0 <= x < W and 0 <= y < H):
        return None
    x0, y0 = max(0, int(x) - 45), max(0, int(y) - 45)
    x1, y1 = min(W, int(x) + 46), min(H, int(y) + 46)
    hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (11, 90, 200), (22, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_d = None, float(search)
    for c in contours:
        area = cv2.contourArea(c)
        if not (BALL_AREA[0] <= area <= BALL_AREA[1]):
            continue
        bx, by, bw, bh = cv2.boundingRect(c)
        if not (0.45 <= bw / float(bh) <= 2.2) or area / float(bw * bh) < 0.45:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(c)
        d = np.hypot(cx + x0 - x, cy + y0 - y)
        if d < best_d:
            best, best_d = (cx + x0, cy + y0, r), d
    return best


def render_video(video_path, out_video, map1, map2, positions, plan, fps, W, H, trail_len):
    cap = cv2.VideoCapture(video_path)
    writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    if not writer.isOpened():
        raise IOError("OpenCV could not create the output video: " + out_video)
    idx = 0
    while True:
        ok, raw = cap.read()
        if not ok:
            break
        frame = cv2.remap(raw, map1, map2, cv2.INTER_LINEAR)
        if idx in plan:
            px, py, pr, gap = plan[idx]
            found = local_ball(frame, px, py)
            if found is not None:
                positions[idx] = (found[0], found[1], "interpolated", found[2])
            elif gap <= 3 and -20 <= py <= H and 0 <= px <= W:
                positions[idx] = (px, py, "interpolated", pr)
        writer.write(draw_frame(frame, positions, idx, trail_len))
        idx += 1
        if idx % 600 == 0:
            print("writing video:", idx, "frames")
    cap.release()
    writer.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=None)
    ap.add_argument("--calib", default=None)
    ap.add_argument("--out", default="output")
    ap.add_argument("--trail", type=int, default=45)
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    video_path = find_file(args.video, ["Volleyball.mp4", "*.mp4"], "video")
    calib_path = find_file(args.calib, ["camera_calibration*.yaml", "*.yaml", "*.yml"], "calibration")
    out_dir = args.out if os.path.isabs(args.out) else os.path.join(BASE_DIR, args.out)
    os.makedirs(out_dir, exist_ok=True)
    print("Video:", video_path)
    print("Calibration:", calib_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("OpenCV could not open the video: " + video_path)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    K, D, R, csize = load_calibration(calib_path)
    map1, map2 = build_undistort_maps(K, D, R, csize, (W, H))

    show = (not args.no_show) and can_show()
    if not show:
        print("Live preview is off (no display or --no-show). The output video is still created.")
    positions, candidates, total = track_video(video_path, map1, map2, H, W, args.trail, show)
    plan = plan_gaps(positions)

    out_video = os.path.join(out_dir, "tracked_output.mp4")
    render_video(video_path, out_video, map1, map2, positions, plan, fps, W, H, args.trail)

    with open(os.path.join(out_dir, "ball_trajectory.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "x", "y", "status"])
        for i in sorted(positions):
            w.writerow([i, round(positions[i][0], 1), round(positions[i][1], 1), positions[i][2]])
    counts = {}
    for p in positions.values():
        counts[p[2]] = counts.get(p[2], 0) + 1
    print("Saved:", out_video)
    print("frames:", total, "| frames with a circle:", len(positions), "|", counts)


if __name__ == "__main__":
    main()

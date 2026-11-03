import argparse
import csv
import glob
import os
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


def track_video(video_path, H, W, trail_len):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("OpenCV could not open the video: " + video_path)
    tracker = BallTracker()
    positions = {}
    prev_gray = None
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if prev_gray is None:
            prev_gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        cands, gray = ball_candidates(frame, prev_gray)
        prev_gray = gray
        state = tracker.update(cands)
        if state is not None and -20 <= state[1] <= H and 0 <= state[0] <= W:
            positions[idx] = (state[0], state[1], state[2], tracker.radius)
        cv2.imshow("Live ball tracking (press q to close)", draw_frame(frame, positions, idx, trail_len))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        idx += 1
    cap.release()
    cv2.destroyAllWindows()
    return positions, idx


def save_csv(positions, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "x", "y", "status"])
        for frame in sorted(positions):
            x, y, status, _ = positions[frame]
            writer.writerow([frame, round(x, 1), round(y, 1), status])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=None)
    ap.add_argument("--trail", type=int, default=45)
    args = ap.parse_args()

    if not can_show():
        print("No display was found, so the live preview cannot be shown on this computer.")
        return

    video_path = find_file(args.video, ["Volleyball.mp4", "*.mp4"], "video")
    print("Video:", video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("OpenCV could not open the video: " + video_path)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    positions, total = track_video(video_path, H, W, args.trail)
    csv_path = os.path.join(BASE_DIR, "ball_trajectory.csv")
    save_csv(positions, csv_path)
    print("Trajectory CSV saved:", csv_path)
    detected = sum(1 for p in positions.values() if p[2] == "detected")
    print("frames shown:", total, "| ball marked in:", len(positions),
          "| detected:", detected, "| predicted:", len(positions) - detected)


if __name__ == "__main__":
    main()

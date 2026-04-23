"""
openpilot-style YOLOP 320×320 Live Cam UI — Enhanced v4.1
=========================================================
Fix v4.1:
  - cv2.setMouseCallback moved after window is fully initialized
  - Added cv2.waitKey(1) to flush window creation before callback
  - Added fallback for Qt / headless environments
  - Graceful handling of window creation failure

INSTALL:  pip install opencv-python numpy onnxruntime onnx
RUN:      python openpilot_yolop_v4.py
          python openpilot_yolop_v4.py --source road.mp4
KEYS:
  Q/ESC  quit          E  engage        D  drivable area
  L  lane lines        B  boxes         P  path
  M  bird-eye map      W  warnings      N  night mode
  +/-  brightness      [/]  confidence
  C  calibration mode  R  record        S  screenshot
"""

import argparse
import collections
import csv
import datetime
import json
import math
import os
import platform
import sys
import time

import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
MODEL_FILE    = "yolop-320-320.onnx"
INF_SIZE      = 320
DISPLAY_W     = 1280
DISPLAY_H     = 720
ORT_THREADS   = 2
INF_EVERY     = 2
MIN_LANE_PIX  = 80
PATH_EMA      = 0.20
PATH_TOP_FRAC = 0.70
PATH_N_ROWS   = 20

# Bird-eye view panel size
BEV_W = 280
BEV_H = 360

# Colours (BGR)
C_GREEN   = (0,   195,  55)
C_WHITE   = (220, 220, 220)
C_BOX     = (0,   175, 255)
C_ENGAGE  = (0,   215,  75)
C_STBY    = (175, 175, 175)
C_WARN    = (0,    80, 255)
C_PATH    = (0,   230, 255)
C_HORIZON = (180,  80, 200)
C_LEAD    = (0,    60, 255)
C_DARK    = (15,   15,  15)
C_GRID    = (40,   40,  40)

_MEAN     = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD      = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_DILATE_K = np.ones((3, 3), np.uint8)

_CURVE_HIST = 14
_STEER_HIST = 12
_SPEED_HIST = 20

# Multi-class detection  (COCO id → name, BGR, icon)
CLASS_META = {
    0: ("person",     (100, 120, 255), "P"),
    1: ("bicycle",    (180, 220,   0), "B"),
    2: ("car",        (  0, 175, 255), "C"),
    3: ("motorcycle", (180,  80, 255), "M"),
    5: ("bus",        (  0, 100, 255), "U"),
    7: ("truck",      (  0,  50, 200), "T"),
}
DANGER_CLASSES = {0, 1}

# IPM default trapezoid (fraction of frame W×H)
_DEFAULT_IPM_SRC_FRAC = np.float32([
    [0.12, 0.95],   # bottom-left
    [0.88, 0.95],   # bottom-right
    [0.60, 0.58],   # top-right
    [0.40, 0.58],   # top-left
])


# ─────────────────────────────────────────────────────────────────────────────
# Stub builder
# ─────────────────────────────────────────────────────────────────────────────
def _build_stub(path: str) -> bool:
    try:
        import onnx
        from onnx import helper, TensorProto
        specs = [
            ("det_out",        [1, 3, 20, 20, 6]),
            ("drive_area_seg", [1, 2, INF_SIZE, INF_SIZE]),
            ("lane_line_seg",  [1, 2, INF_SIZE, INF_SIZE]),
        ]
        nodes = []
        for name, shape in specs:
            vals = np.zeros(shape, dtype=np.float32)
            t = helper.make_tensor(name + "_val", TensorProto.FLOAT,
                                   shape, vals.flatten().tolist())
            nodes.append(
                helper.make_node("Constant", inputs=[], outputs=[name], value=t)
            )
        graph = helper.make_graph(
            nodes, "yolop_stub",
            [helper.make_tensor_value_info(
                "images", TensorProto.FLOAT, [1, 3, INF_SIZE, INF_SIZE])],
            [helper.make_tensor_value_info(n, TensorProto.FLOAT, s)
             for n, s in specs],
        )
        m = helper.make_model(graph,
                              opset_imports=[helper.make_opsetid("", 12)])
        onnx.save(m, path)
        print(f"[+] Stub: {path}")
        return True
    except Exception as e:
        print(f"[!] Stub failed: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────
def load_session(path: str):
    import onnxruntime as ort
    is_stub = False
    if not os.path.exists(path):
        print(f"[!] {path} not found — building stub.")
        stub = path.replace(".onnx", "_stub.onnx")
        if not _build_stub(stub):
            sys.exit(1)
        path, is_stub = stub, True
    else:
        print(f"[+] Model: {path}")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads     = ORT_THREADS
    opts.inter_op_num_threads     = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.execution_mode           = ort.ExecutionMode.ORT_SEQUENTIAL
    sess      = ort.InferenceSession(path, sess_options=opts,
                                     providers=["CPUExecutionProvider"])
    out_names = [o.name for o in sess.get_outputs()]
    in_name   = sess.get_inputs()[0].name
    print(f"[+] Outputs: {out_names}  stub={is_stub}")
    return sess, out_names, in_name, is_stub


# ─────────────────────────────────────────────────────────────────────────────
# Pre / post processing
# ─────────────────────────────────────────────────────────────────────────────
def preprocess(frame_bgr: np.ndarray) -> np.ndarray:
    img = cv2.resize(frame_bgr, (INF_SIZE, INF_SIZE),
                     interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - _MEAN) / _STD
    return img.transpose(2, 0, 1)[np.newaxis].astype(np.float32)


def seg_mask(tensor: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    m = np.argmax(tensor[0], axis=0).astype(np.uint8)
    if m.shape != (out_h, out_w):
        m = cv2.resize(m, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    return m


def decode_det(det, fh, fw, conf=0.40, iou=0.45):
    """Returns list of (x1,y1,x2,y2,score,cls_id)."""
    cands = []

    def _collect(x):
        if isinstance(x, np.ndarray):
            cands.append(x)
        elif isinstance(x, (list, tuple)):
            for i in x:
                _collect(i)

    _collect(det)
    pred = None
    for t in cands:
        if t.ndim == 5:
            t = t.reshape(t.shape[0], -1, t.shape[-1])[0]
        elif t.ndim == 3:
            t = t[0]
        if t.ndim == 2 and t.shape[-1] >= 6 and t.shape[0] > 0:
            pred = t
            break
    if pred is None or len(pred) == 0:
        return []
    scores = pred[:, 4] * pred[:, 5]
    keep   = scores > conf
    if not keep.any():
        return []
    p      = pred[keep]
    scores = scores[keep]
    sx, sy = fw / INF_SIZE, fh / INF_SIZE
    cx_    = p[:, 0] * sx
    cy_    = p[:, 1] * sy
    bw_    = p[:, 2] * sx
    bh_    = p[:, 3] * sy
    x1 = np.clip((cx_ - bw_ / 2).astype(int), 0, fw - 1)
    y1 = np.clip((cy_ - bh_ / 2).astype(int), 0, fh - 1)
    x2 = np.clip((cx_ + bw_ / 2).astype(int), 0, fw - 1)
    y2 = np.clip((cy_ + bh_ / 2).astype(int), 0, fh - 1)
    xywh = np.stack([x1, y1, x2-x1, y2-y1], 1).astype(float).tolist()
    idxs = cv2.dnn.NMSBoxes(xywh, scores.tolist(), conf, iou)
    if len(idxs) == 0:
        return []
    idxs = np.array(idxs).flatten()
    return [(int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i]),
             float(scores[i]), 2) for i in idxs]


# ─────────────────────────────────────────────────────────────────────────────
# Kalman Lane Tracker
# ─────────────────────────────────────────────────────────────────────────────
class LaneKalman:
    def __init__(self):
        self.kf = cv2.KalmanFilter(6, 3)
        dt = 1.0
        self.kf.transitionMatrix = np.array([
            [1,0,0,dt,0,0],
            [0,1,0,0,dt,0],
            [0,0,1,0,0,dt],
            [0,0,0,1,0,0],
            [0,0,0,0,1,0],
            [0,0,0,0,0,1],
        ], dtype=np.float32)
        self.kf.measurementMatrix   = np.eye(3, 6, dtype=np.float32)
        self.kf.processNoiseCov     = np.eye(6, dtype=np.float32) * 1e-4
        self.kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 1e-2
        self.kf.errorCovPost        = np.eye(6, dtype=np.float32)
        self.initialized = False
        self.frames_lost = 0

    def update(self, fit):
        meas = fit.astype(np.float32).reshape(3, 1)
        if not self.initialized:
            self.kf.statePost[:3] = meas
            self.initialized = True
        self.kf.predict()
        self.kf.correct(meas)
        self.frames_lost = 0
        return self.kf.statePost[:3].flatten()

    def predict_only(self):
        self.frames_lost += 1
        self.kf.predict()
        return self.kf.statePost[:3].flatten()

    @property
    def valid(self):
        return self.initialized and self.frames_lost < 8


# ─────────────────────────────────────────────────────────────────────────────
# IPM Transform
# ─────────────────────────────────────────────────────────────────────────────
class IPMTransform:
    CAL_FILE = "ipm_calibration.json"

    def __init__(self, w: int, h: int):
        self._w, self._h = w, h
        self.src = (_DEFAULT_IPM_SRC_FRAC *
                    np.float32([w, h])).astype(np.float32)
        self._load()
        self._build()

    def _build(self):
        dst = np.float32([
            [0,     BEV_H],
            [BEV_W, BEV_H],
            [BEV_W, 0    ],
            [0,     0    ],
        ])
        self.M     = cv2.getPerspectiveTransform(self.src, dst)
        self.M_inv = cv2.getPerspectiveTransform(dst, self.src)

    def warp(self, frame: np.ndarray) -> np.ndarray:
        return cv2.warpPerspective(frame, self.M, (BEV_W, BEV_H),
                                   flags=cv2.INTER_LINEAR)

    def warp_mask(self, mask: np.ndarray) -> np.ndarray:
        return cv2.warpPerspective(mask, self.M, (BEV_W, BEV_H),
                                   flags=cv2.INTER_NEAREST)

    def cam_to_bev(self, pts: np.ndarray) -> np.ndarray:
        if len(pts) == 0:
            return pts
        p = pts.reshape(-1, 1, 2).astype(np.float32)
        return cv2.perspectiveTransform(p, self.M).reshape(-1, 2)

    def save(self):
        data = {"src": self.src.tolist(), "w": self._w, "h": self._h}
        with open(self.CAL_FILE, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[CAL] Saved {self.CAL_FILE}")

    def _load(self):
        if not os.path.exists(self.CAL_FILE):
            return
        try:
            with open(self.CAL_FILE) as f:
                d = json.load(f)
            if d["w"] == self._w and d["h"] == self._h:
                self.src = np.array(d["src"], dtype=np.float32)
                print(f"[CAL] Loaded {self.CAL_FILE}")
        except Exception as e:
            print(f"[CAL] Load failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Calibration UI
# ─────────────────────────────────────────────────────────────────────────────
class CalibrationUI:
    def __init__(self, ipm: IPMTransform):
        self._ipm      = ipm
        self._drag_idx = None
        self.active    = False

    def toggle(self):
        self.active = not self.active

    def mouse(self, event, x, y, flags, param):
        if not self.active:
            return
        pts = self._ipm.src
        if event == cv2.EVENT_LBUTTONDOWN:
            dists = np.linalg.norm(pts - [x, y], axis=1)
            idx   = int(np.argmin(dists))
            if dists[idx] < 25:
                self._drag_idx = idx
        elif event == cv2.EVENT_MOUSEMOVE and self._drag_idx is not None:
            pts[self._drag_idx] = [x, y]
            self._ipm._build()
        elif event == cv2.EVENT_LBUTTONUP:
            self._drag_idx = None

    def draw(self, canvas: np.ndarray):
        if not self.active:
            return
        pts    = self._ipm.src.astype(np.int32)
        order  = [0, 1, 2, 3, 0]
        labels = ["BL", "BR", "TR", "TL"]
        for i in range(4):
            cv2.line(canvas,
                     tuple(pts[order[i]]),
                     tuple(pts[order[i+1]]),
                     (0, 255, 80), 1, cv2.LINE_AA)
        for i, (x, y) in enumerate(pts):
            cv2.circle(canvas, (x, y), 10, (0, 255, 80), 2, cv2.LINE_AA)
            cv2.putText(canvas, labels[i], (x+12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 255, 80), 1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            "CALIBRATION — drag corners  |  ENTER = save  |  C = exit",
            (canvas.shape[1]//2 - 270, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 80), 2, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# Speed estimator
# ─────────────────────────────────────────────────────────────────────────────
class SpeedEstimator:
    def __init__(self):
        self._prev = None
        self._hist = collections.deque(maxlen=_SPEED_HIST)
        self.kmh   = 0.0

    def update(self, frame: np.ndarray) -> float:
        h, w = frame.shape[:2]
        roi  = frame[h//3: 2*h//3, w//4: 3*w//4]
        gray = cv2.cvtColor(cv2.resize(roi, (80, 45)), cv2.COLOR_BGR2GRAY)
        if self._prev is not None:
            flow = cv2.calcOpticalFlowFarneback(
                self._prev, gray, None, 0.5, 2, 8, 2, 5, 1.1, 0)
            mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            self._hist.append(float(np.mean(mag)) * 18.0)
        self._prev = gray
        if self._hist:
            self.kmh = float(np.mean(self._hist))
        return self.kmh


# ─────────────────────────────────────────────────────────────────────────────
# Lane Analyser with Kalman
# ─────────────────────────────────────────────────────────────────────────────
class LaneAnalyser:
    def __init__(self, w: int, h: int):
        self._w, self._h  = w, h
        self._smooth_path = []
        self._left_fit    = None
        self._right_fit   = None
        self._left_kf     = LaneKalman()
        self._right_kf    = LaneKalman()
        self.lane_count   = 0
        self.curve_rad    = 9999.0
        self.lat_offset   = 0.0
        self.departure    = False
        self._curve_hist  = collections.deque(maxlen=_CURVE_HIST)
        self._offset_hist = collections.deque(maxlen=_CURVE_HIST)
        self._half_lane_w = w * 0.22

    @staticmethod
    def _fit_poly(ys, xs):
        if len(ys) < MIN_LANE_PIX:
            return None
        try:
            return np.polyfit(ys, xs, 2)
        except Exception:
            return None

    @staticmethod
    def _poly_x(fit, y):
        return float(np.polyval(fit, y))

    @staticmethod
    def _curve_radius(fit, y_eval, ym=30/720, xm=3.7/700):
        if fit is None:
            return 9999.0
        A = fit[0] * (xm / ym**2)
        B = fit[1] * (xm / ym)
        return float(min(
            ((1 + (2*A*y_eval*ym + B)**2)**1.5) / abs(2*A + 1e-6),
            9999.0))

    def _split(self, ll_mask: np.ndarray):
        cx = self._w / 2.0
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(
            ll_mask.astype(np.uint8), connectivity=8)
        ly, lx, ry, rx = [], [], [], []
        for lbl in range(1, n):
            if stats[lbl, cv2.CC_STAT_AREA] < 20:
                continue
            ys, xs = np.where(labels == lbl)
            if centroids[lbl, 0] < cx:
                ly.extend(ys.tolist()); lx.extend(xs.tolist())
            else:
                ry.extend(ys.tolist()); rx.extend(xs.tolist())
        return (np.array(ly), np.array(lx),
                np.array(ry), np.array(rx))

    def _da_path(self, da_mask: np.ndarray):
        h, w  = da_mask.shape
        y_top = int(h * PATH_TOP_FRAC)
        ys    = np.linspace(y_top, h-1, PATH_N_ROWS).astype(int)
        pts   = []
        for y in ys:
            xs = np.where(da_mask[y] > 0)[0]
            if len(xs) >= 4:
                pts.append((int(np.mean(xs)), int(y)))
        return pts

    def _poly_path(self):
        h     = self._h
        y_top = int(h * PATH_TOP_FRAC)
        ys    = np.linspace(y_top, h-1, PATH_N_ROWS)
        pts   = []
        for y in ys:
            lx = (self._poly_x(self._left_fit, y)
                  if self._left_fit  is not None else None)
            rx = (self._poly_x(self._right_fit, y)
                  if self._right_fit is not None else None)
            if lx is not None and rx is not None:
                lo, hi = (lx, rx) if lx < rx else (rx, lx)
                cx     = max(lo+2, min(hi-2, (lo+hi)/2.0))
            elif lx is not None:
                cx = lx + self._half_lane_w
            elif rx is not None:
                cx = rx - self._half_lane_w
            else:
                continue
            pts.append((int(np.clip(cx, 0, self._w-1)), int(y)))
        return pts

    def _smooth(self, new_pts):
        if not new_pts:
            return self._smooth_path
        if len(self._smooth_path) != len(new_pts):
            self._smooth_path = new_pts
            return self._smooth_path
        smoothed = []
        for (nx, ny), (ox, _) in zip(new_pts, self._smooth_path):
            smoothed.append((int(ox + PATH_EMA*(nx-ox)), ny))
        self._smooth_path = smoothed
        return smoothed

    def update(self, ll_mask: np.ndarray, da_mask):
        h, w         = ll_mask.shape
        self._w, self._h = w, h
        ly, lx, ry, rx = self._split(ll_mask)

        raw_left  = self._fit_poly(ly, lx)
        raw_right = self._fit_poly(ry, rx)

        if raw_left is not None:
            kf_left = self._left_kf.update(raw_left)
        elif self._left_kf.valid:
            kf_left = self._left_kf.predict_only()
        else:
            kf_left = None

        if raw_right is not None:
            kf_right = self._right_kf.update(raw_right)
        elif self._right_kf.valid:
            kf_right = self._right_kf.predict_only()
        else:
            kf_right = None

        self._left_fit  = kf_left
        self._right_fit = kf_right
        has_left  = self._left_fit  is not None
        has_right = self._right_fit is not None
        self.lane_count = int(has_left) + int(has_right)

        if has_left and has_right:
            y_ref = h * 0.85
            lx_r  = self._poly_x(self._left_fit,  y_ref)
            rx_r  = self._poly_x(self._right_fit, y_ref)
            if rx_r > lx_r + 20:
                self._half_lane_w = (rx_r - lx_r) / 2.0

        y_eval = h * 0.80
        rads   = [self._curve_radius(f, y_eval)
                  for f in [self._left_fit, self._right_fit]
                  if f is not None]
        self._curve_hist.append(float(np.mean(rads)) if rads else 9999.0)
        self.curve_rad = float(np.mean(self._curve_hist))

        cx_img = w / 2.0
        if has_left and has_right:
            lx_r    = self._poly_x(self._left_fit,  y_eval)
            rx_r    = self._poly_x(self._right_fit, y_eval)
            raw_off = ((lx_r+rx_r)/2.0 - cx_img) / (w/2.0)
        elif has_left:
            lx_r    = self._poly_x(self._left_fit, y_eval)
            raw_off = (lx_r + self._half_lane_w - cx_img) / (w/2.0)
        elif has_right:
            rx_r    = self._poly_x(self._right_fit, y_eval)
            raw_off = (rx_r - self._half_lane_w - cx_img) / (w/2.0)
        else:
            raw_off = 0.0

        self._offset_hist.append(raw_off)
        self.lat_offset = float(np.mean(self._offset_hist))
        self.departure  = abs(self.lat_offset) > 0.40

        if self.lane_count >= 1:
            raw_pts = self._poly_path()
        else:
            raw_pts = self._da_path(da_mask) if da_mask is not None else []
        self._smooth_path = self._smooth(raw_pts)

    @property
    def path_points(self):
        return self._smooth_path

    def lane_edge_points(self, n=20):
        h, y_top = self._h, int(self._h * PATH_TOP_FRAC)
        ys = np.linspace(y_top, h-1, n)
        lpts, rpts = [], []
        for y in ys:
            if self._left_fit is not None:
                lpts.append((int(self._poly_x(self._left_fit,  y)), int(y)))
            if self._right_fit is not None:
                rpts.append((int(self._poly_x(self._right_fit, y)), int(y)))
        return lpts, rpts


# ─────────────────────────────────────────────────────────────────────────────
# Steering model
# ─────────────────────────────────────────────────────────────────────────────
class SteeringModel:
    def __init__(self):
        self._hist  = collections.deque(maxlen=_STEER_HIST)
        self.angle  = 0.0
        self.torque = 0.0

    def update(self, curve_rad, lat_offset,
               left_fit=None, right_fit=None) -> float:
        curve_steer = 0.0
        if curve_rad < 9000:
            fits = [f for f in [left_fit, right_fit] if f is not None]
            if fits:
                avg_a = float(np.mean([f[0] for f in fits]))
                mag   = math.degrees(math.atan(3.0/max(curve_rad, 1.0))) * 8.0
                curve_steer = -np.sign(avg_a) * mag
        raw = max(-45.0, min(45.0, curve_steer + lat_offset*18.0))
        self._hist.append(raw)
        self.angle  = float(np.mean(self._hist))
        self.torque = self.angle / 45.0
        return self.angle


# ─────────────────────────────────────────────────────────────────────────────
# Lead tracker
# ─────────────────────────────────────────────────────────────────────────────
class LeadTracker:
    def __init__(self):
        self.box    = None
        self.dist_m = 0.0
        self.ttc_s  = 99.0

    def update(self, dets, fw, fh, speed_kmh):
        if not dets:
            self.box = None; self.dist_m = 0.0; self.ttc_s = 99.0
            return
        cx   = fw / 2
        best, best_s = None, -1.0
        for d in dets:
            x1, y1, x2, y2, sc, _ = d
            area = (x2-x1)*(y2-y1)
            cent = 1.0 - abs(((x1+x2)/2) - cx) / cx
            rank = area * cent * (y2/fh) * sc
            if rank > best_s:
                best, best_s = d, rank
        self.box = best
        if best:
            bh          = max(best[3]-best[1], 1)
            self.dist_m = max(1.5*700.0/bh, 1.0)
            rel_v       = max(speed_kmh*0.8/3.6, 0.1)
            self.ttc_s  = self.dist_m / rel_v
        else:
            self.dist_m = 0.0; self.ttc_s = 99.0


# ─────────────────────────────────────────────────────────────────────────────
# Night mode
# ─────────────────────────────────────────────────────────────────────────────
class AdaptiveDisplay:
    def __init__(self):
        self.auto   = True
        self.night  = False
        self._clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

    def process(self, frame: np.ndarray) -> np.ndarray:
        lum = float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())
        if self.auto:
            self.night = lum < 60
        if not self.night:
            return frame.copy()
        lab     = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        lab_eq  = cv2.merge([self._clahe.apply(l), a, b])
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


# ─────────────────────────────────────────────────────────────────────────────
# Dashcam recorder
# ─────────────────────────────────────────────────────────────────────────────
class DashcamRecorder:
    SPLIT_SEC = 600
    MAX_FILES = 6
    FPS_OUT   = 25

    def __init__(self, out_dir="dashcam"):
        self._dir    = out_dir
        self._writer = None
        self._start  = None
        self._files  = collections.deque()
        self.active  = False
        os.makedirs(out_dir, exist_ok=True)

    def toggle(self):
        self.active = not self.active
        if not self.active and self._writer:
            self._writer.release()
            self._writer = None
        print(f"[REC] {'ON' if self.active else 'OFF'}")

    def write(self, frame: np.ndarray):
        if not self.active:
            return
        now = time.time()
        if self._writer is None or now - self._start > self.SPLIT_SEC:
            self._rotate(frame.shape[1], frame.shape[0])
        self._writer.write(frame)

    def _rotate(self, w, h):
        if self._writer:
            self._writer.release()
        ts   = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self._dir, f"dash_{ts}.mp4")
        self._writer = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc(*"mp4v"), self.FPS_OUT, (w, h))
        self._start = time.time()
        self._files.append(path)
        while len(self._files) > self.MAX_FILES:
            old = self._files.popleft()
            try:    os.remove(old)
            except: pass
        print(f"[REC] {path}")

    def stop(self):
        if self._writer:
            self._writer.release()


# ─────────────────────────────────────────────────────────────────────────────
# Event logger
# ─────────────────────────────────────────────────────────────────────────────
class EventLogger:
    COOLDOWN = 5.0

    def __init__(self, path="events.csv"):
        self._last = {}
        self._f    = open(path, "a", newline="")
        self._csv  = csv.writer(self._f)

    def log(self, kind: str, detail: str = ""):
        now = time.time()
        if now - self._last.get(kind, 0) < self.COOLDOWN:
            return
        self._last[kind] = now
        ts = datetime.datetime.now().isoformat(timespec="seconds")
        self._csv.writerow([ts, kind, detail])
        self._f.flush()

    def close(self):
        self._f.close()


# ─────────────────────────────────────────────────────────────────────────────
# Bird's-Eye View panel
# ─────────────────────────────────────────────────────────────────────────────
class BEVPanel:
    M_PER_PIX = 40.0 / BEV_H

    def __init__(self, ipm: IPMTransform, fw: int, fh: int):
        self._ipm = ipm
        self._fw  = fw
        self._fh  = fh

    @staticmethod
    def _draw_grid(canvas: np.ndarray):
        pix_per_10m = max(1, int(10.0 / BEVPanel.M_PER_PIX))
        y = BEV_H - pix_per_10m
        dist_m = 10
        while y > 0:
            cv2.line(canvas, (0, y), (BEV_W, y), C_GRID, 1, cv2.LINE_AA)
            cv2.putText(canvas, f"{dist_m}m", (4, y-2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28,
                        (70, 70, 70), 1, cv2.LINE_AA)
            y -= pix_per_10m
            dist_m += 10
        cx = BEV_W // 2
        cv2.line(canvas, (cx, 0), (cx, BEV_H), (40, 40, 40), 1, cv2.LINE_AA)
        half_lane_pix = int(1.8 / max(3.7 / (BEV_W * 0.60), 1e-6))
        for dx in [-half_lane_pix, half_lane_pix]:
            x = cx + dx
            if 0 <= x < BEV_W:
                cv2.line(canvas, (x, 0), (x, BEV_H),
                         (35, 55, 35), 1, cv2.LINE_AA)

    @staticmethod
    def _draw_ego(canvas: np.ndarray):
        cx  = BEV_W // 2
        cy  = BEV_H - 18
        cw2 = 12
        cl  = 22
        body = np.array([
            [cx-cw2, cy], [cx+cw2, cy],
            [cx+cw2, cy-cl], [cx-cw2, cy-cl],
        ], np.int32)
        cv2.fillPoly(canvas, [body], (60, 130, 60))
        cv2.polylines(canvas, [body], True, (0, 220, 80), 2, cv2.LINE_AA)
        ws = np.array([
            [cx-cw2+3, cy-cl+4], [cx+cw2-3, cy-cl+4],
            [cx+cw2-3, cy-cl+10], [cx-cw2+3, cy-cl+10],
        ], np.int32)
        cv2.fillPoly(canvas, [ws], (140, 200, 140))
        for dx in [-cw2+3, cw2-3]:
            cv2.circle(canvas, (cx+dx, cy-cl+2), 3,
                       (220, 220, 100), -1, cv2.LINE_AA)
        cv2.putText(canvas, "EGO", (cx-10, cy+12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28,
                    (0, 200, 80), 1, cv2.LINE_AA)

    @staticmethod
    def _draw_bev_object(canvas, bx, by, cls_id, score, is_lead):
        name, col, icon = CLASS_META.get(cls_id, ("obj", C_BOX, "?"))
        bx = int(np.clip(bx, 4, BEV_W-4))
        by = int(np.clip(by, 4, BEV_H-4))
        dist_frac = by / BEV_H
        box_w = max(8,  int(22 * dist_frac))
        box_h = max(10, int(30 * dist_frac))
        x1 = bx - box_w//2
        y1 = by - box_h
        x2 = bx + box_w//2
        y2 = by
        # Semi-transparent fill
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2),
                      tuple(int(c*0.4) for c in col), -1)
        cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0, canvas)
        border = (0, 0, 220) if is_lead else col
        cv2.rectangle(canvas, (x1, y1), (x2, y2), border,
                      3 if is_lead else 2, cv2.LINE_AA)
        font_scale = max(0.22, 0.40 * dist_frac)
        (tw, th), _ = cv2.getTextSize(
            icon, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.putText(canvas, icon,
                    (bx - tw//2, y1 + box_h//2 + th//2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), 1, cv2.LINE_AA)
        if is_lead:
            cv2.circle(canvas, (bx, by - box_h//2),
                       box_w+4, (0, 50, 200), 1, cv2.LINE_AA)

    def render(self, da_mask, ll_mask, lane_an,
               dets, lead_box, canvas_out, ox, oy):
        bev = np.zeros((BEV_H, BEV_W, 3), dtype=np.uint8)
        self._draw_grid(bev)

        # Drivable area
        if da_mask is not None:
            try:
                da_bev = self._ipm.warp_mask(da_mask)
                da_col = np.zeros_like(bev)
                da_col[da_bev == 1] = (0, 80, 0)
                cv2.addWeighted(bev, 1.0, da_col, 0.85, 0, bev)
                contours, _ = cv2.findContours(
                    da_bev, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(bev, contours, -1, (0, 160, 50), 1)
            except Exception:
                pass

        # Lane lines
        if ll_mask is not None:
            try:
                ll_bev = self._ipm.warp_mask(ll_mask)
                bev[ll_bev == 1] = (200, 200, 200)
            except Exception:
                pass

        # Planned path
        pts = lane_an.path_points
        if len(pts) >= 2:
            try:
                cam_pts = np.array(pts, dtype=np.float32)
                bev_pts = self._ipm.cam_to_bev(cam_pts)
                valid   = ((bev_pts[:,0] >= 0) & (bev_pts[:,0] < BEV_W) &
                           (bev_pts[:,1] >= 0) & (bev_pts[:,1] < BEV_H))
                bev_pts = bev_pts[valid].astype(np.int32)
                if len(bev_pts) >= 2:
                    for i in range(0, len(bev_pts)-1, 2):
                        cv2.line(bev, tuple(bev_pts[i]),
                                 tuple(bev_pts[i+1]), C_PATH, 2, cv2.LINE_AA)
                    for i, pt in enumerate(bev_pts):
                        r = max(2, int(5*(1 - i/max(len(bev_pts)-1,1))))
                        cv2.circle(bev, tuple(pt), r, C_PATH, -1, cv2.LINE_AA)
            except Exception:
                pass

        # Lane edges
        lpts, rpts = lane_an.lane_edge_points(n=16)
        for side, col in [(lpts, (0, 200, 80)), (rpts, (0, 200, 80))]:
            if len(side) >= 2:
                try:
                    cam_arr = np.array(side, dtype=np.float32)
                    bev_arr = self._ipm.cam_to_bev(cam_arr).astype(np.int32)
                    valid   = ((bev_arr[:,0]>=0)&(bev_arr[:,0]<BEV_W)&
                               (bev_arr[:,1]>=0)&(bev_arr[:,1]<BEV_H))
                    bev_arr = bev_arr[valid]
                    if len(bev_arr) >= 2:
                        cv2.polylines(bev, [bev_arr], False, col,
                                      2, cv2.LINE_AA)
                except Exception:
                    pass

        # Objects
        for det in dets:
            x1, y1, x2, y2, sc, cls_id = det
            try:
                foot  = np.array([[(x1+x2)/2.0, float(y2)]], np.float32)
                bev_pt = self._ipm.cam_to_bev(foot)[0]
                is_lead = (lead_box is not None and
                           lead_box[0] == x1 and lead_box[1] == y1)
                self._draw_bev_object(bev, int(bev_pt[0]), int(bev_pt[1]),
                                      cls_id, sc, is_lead)
            except Exception:
                pass

        self._draw_ego(bev)

        # Border + title
        cv2.rectangle(bev, (0, 0), (BEV_W-1, BEV_H-1), (60, 60, 60), 1)
        cv2.putText(bev, "BIRD'S-EYE VIEW",
                    (BEV_W//2 - 56, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (100, 100, 100), 1, cv2.LINE_AA)

        # Paste into canvas
        ch, cw = canvas_out.shape[:2]
        ey = min(oy + BEV_H, ch)
        ex = min(ox + BEV_W, cw)
        ph = ey - oy
        pw = ex - ox
        if ph > 0 and pw > 0:
            canvas_out[oy:ey, ox:ex] = bev[:ph, :pw]


# ─────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────────────────────
def apply_brightness(frame, offset):
    if offset == 0.0:
        return frame.copy()
    return np.clip(frame.astype(np.int16) + int(offset), 0, 255).astype(np.uint8)


def overlay_da(canvas, mask, alpha=0.28):
    ys, xs = np.where(mask == 1)
    if len(ys) == 0:
        return
    roi = canvas[ys, xs].astype(np.float32)
    canvas[ys, xs] = np.clip(
        roi*(1-alpha) + np.array(C_GREEN, np.float32)*alpha, 0, 255
    ).astype(np.uint8)


def overlay_ll(canvas, mask):
    dilated = cv2.dilate(mask, _DILATE_K, iterations=2)
    ys, xs  = np.where(dilated == 1)
    if len(ys):
        canvas[ys, xs] = C_WHITE


def draw_boxes(canvas, dets, lead_box=None):
    for x1, y1, x2, y2, sc, cls_id in dets:
        name, col, icon = CLASS_META.get(cls_id, (f"cls{cls_id}", C_BOX, "?"))
        is_lead   = (lead_box is not None and
                     lead_box[0] == x1 and lead_box[1] == y1)
        is_danger = cls_id in DANGER_CLASSES
        border    = C_LEAD if is_lead else col
        thick     = 3 if (is_lead or is_danger) else 2
        cv2.rectangle(canvas, (x1, y1), (x2, y2), border, thick, cv2.LINE_AA)
        label = (f"{'⚠ ' if is_danger else ''}"
                 f"{'LEAD ' if is_lead else ''}{name} {sc:.0%}")
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
        cv2.rectangle(canvas, (x1, y1-th-6), (x1+tw+6, y1), border, -1)
        cv2.putText(canvas, label, (x1+3, y1-3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                    (255, 255, 255), 1, cv2.LINE_AA)


def draw_path_plan(canvas, lane_an, show):
    if not show:
        return
    h, w = canvas.shape[:2]
    pts  = lane_an.path_points
    lc   = lane_an.lane_count
    if lc == 2:
        lpts, rpts = lane_an.lane_edge_points(n=20)
        if len(lpts) >= 2 and len(rpts) >= 2:
            poly = np.array(lpts + list(reversed(rpts)), np.int32
                            ).reshape(-1, 1, 2)
            ov = canvas.copy()
            cv2.fillPoly(ov, [poly], (0, 70, 0))
            cv2.addWeighted(ov, 0.22, canvas, 0.78, 0, canvas)
        for side in [lpts, rpts]:
            if len(side) >= 2:
                cv2.polylines(canvas, [np.array(side, np.int32)],
                              False, (0, 210, 100), 2, cv2.LINE_AA)
    mode_txt = {2: "2-LANE", 1: "1-LANE", 0: "DA PATH"}
    mode_col = {2: (0,230,100), 1: (0,200,255), 0: (200,130,0)}
    cv2.putText(canvas, mode_txt[lc], (w//2-35, h-62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, mode_col[lc], 1, cv2.LINE_AA)
    if len(pts) < 2:
        return
    col = {2: C_PATH, 1: (0,200,255), 0: (200,140,0)}[lc]
    for i in range(0, len(pts)-1, 2):
        cv2.line(canvas, pts[i], pts[i+1], col, 3, cv2.LINE_AA)
    n = len(pts)
    for i, pt in enumerate(pts):
        r = max(2, int(6 * i / max(n-1, 1)))
        cv2.circle(canvas, pt, r, col, -1, cv2.LINE_AA)


def draw_horizon(canvas, da_mask):
    if da_mask is None:
        return
    h, w = canvas.shape[:2]
    rows = np.where(da_mask.sum(axis=1) > w*0.05)[0]
    if len(rows) == 0:
        return
    hy = int(rows.min())
    for x in range(0, w, 20):
        cv2.line(canvas, (x, hy), (x+10, hy), C_HORIZON, 1, cv2.LINE_AA)


def curvature_banner(canvas, curve_rad, lat_offset):
    h, w = canvas.shape[:2]
    if curve_rad > 2000:
        txt, col = "STRAIGHT  \u2191", (0, 220, 80)
    elif curve_rad > 800:
        txt = f"GENTLE {'-->' if lat_offset>0 else '<--'}  {curve_rad:.0f}m"
        col = (0, 180, 255)
    elif curve_rad > 300:
        txt = f"MODERATE {'-->' if lat_offset>0 else '<--'}  {curve_rad:.0f}m"
        col = (0, 120, 255)
    else:
        txt = f"SHARP {'-->' if lat_offset>0 else '<--'}  {curve_rad:.0f}m"
        col = (0, 60, 255)
    (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
    cv2.putText(canvas, txt, (w//2 - tw//2, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 2, cv2.LINE_AA)


def draw_steering_gauge(canvas, angle_deg, r=72):
    h, w = canvas.shape[:2]
    cx   = w - r - 20
    cy   = h - r - 20
    cv2.circle(canvas, (cx, cy), r+6, C_DARK,       -1, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), r+6, (60, 60, 60),  2, cv2.LINE_AA)
    for deg in range(-45, 46, 15):
        rad = math.radians(90+deg)
        x1_ = int(cx + (r-10)*math.cos(rad))
        y1_ = int(cy - (r-10)*math.sin(rad))
        x2_ = int(cx + (r-2) *math.cos(rad))
        y2_ = int(cy - (r-2) *math.sin(rad))
        cv2.line(canvas, (x1_, y1_), (x2_, y2_),
                 (200,200,200) if deg==0 else (100,100,100), 1, cv2.LINE_AA)
    ac = max(-45.0, min(45.0, angle_deg))
    arc_col = ((0,200,80) if abs(ac)<10 else
               (0,160,255) if abs(ac)<25 else (0,60,255))
    sa, ea = 90, int(90+ac)
    if sa != ea:
        cv2.ellipse(canvas, (cx, cy), (r-6, r-6), 0,
                    -max(sa,ea), -min(sa,ea), arc_col, 5, cv2.LINE_AA)
    nr = math.radians(90+ac)
    cv2.line(canvas, (cx, cy),
             (int(cx+(r-10)*math.cos(nr)), int(cy-(r-10)*math.sin(nr))),
             (255,255,255), 3, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), 6, (255,255,255), -1, cv2.LINE_AA)
    cv2.putText(canvas, f"{ac:+.1f}deg",
                (cx-28, cy+r+16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200,200,200), 1, cv2.LINE_AA)
    cv2.putText(canvas, "STEER",
                (cx-18, cy-r-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (120,120,120), 1, cv2.LINE_AA)


def draw_speed_panel(canvas, kmh, x=20, pw=90, ph=60):
    y = canvas.shape[0] - ph - 20
    cv2.rectangle(canvas, (x, y), (x+pw, y+ph), C_DARK, -1, cv2.LINE_AA)
    cv2.rectangle(canvas, (x, y), (x+pw, y+ph), (60,60,60), 1, cv2.LINE_AA)
    kmh_c   = max(0.0, min(kmh, 200.0))
    bar_w   = int(kmh_c/200.0*(pw-10))
    bar_col = ((0,200,80) if kmh_c<80 else
               (0,160,255) if kmh_c<120 else (0,60,255))
    if bar_w > 0:
        cv2.rectangle(canvas, (x+5, y+ph-14),
                      (x+5+bar_w, y+ph-6), bar_col, -1, cv2.LINE_AA)
    cv2.putText(canvas, f"{kmh_c:.0f}", (x+8, y+32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.90, (230,230,230), 2, cv2.LINE_AA)
    cv2.putText(canvas, "km/h", (x+8, y+50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, (140,140,140), 1, cv2.LINE_AA)


def draw_lane_panel(canvas, lane_an, x=20, y=120, pw=150):
    rows = [
        ("LANES",   str(lane_an.lane_count)),
        ("CURVE R", f"{min(lane_an.curve_rad,9999):.0f} m"),
        ("OFFSET",  f"{lane_an.lat_offset:+.2f}"),
        ("DEPART",  "! YES" if lane_an.departure else "OK"),
    ]
    lh = 28
    ph = lh*len(rows)+16
    cv2.rectangle(canvas, (x, y), (x+pw, y+ph), C_DARK, -1, cv2.LINE_AA)
    cv2.rectangle(canvas, (x, y), (x+pw, y+ph), (60,60,60), 1, cv2.LINE_AA)
    for i, (label, val) in enumerate(rows):
        ry  = y + 18 + i*lh
        col = (C_WARN if label == "DEPART" and lane_an.departure
               else (220,220,220))
        cv2.putText(canvas, label, (x+6, ry),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (120,120,120), 1,
                    cv2.LINE_AA)
        cv2.putText(canvas, val, (x+6, ry+14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, col, 1, cv2.LINE_AA)


def draw_lead_panel(canvas, lead, x=20, y=310, pw=150):
    rows = [
        ("LEAD", f"{lead.dist_m:.1f} m" if lead.box else "--"),
        ("TTC",  f"{lead.ttc_s:.1f} s"  if lead.box else "--"),
    ]
    lh = 28
    ph = lh*len(rows)+16
    cv2.rectangle(canvas, (x, y), (x+pw, y+ph), C_DARK, -1, cv2.LINE_AA)
    cv2.rectangle(canvas, (x, y), (x+pw, y+ph), (60,60,60), 1, cv2.LINE_AA)
    for i, (label, val) in enumerate(rows):
        ry  = y + 18 + i*lh
        col = (C_WARN if label=="TTC" and lead.ttc_s < 3.0
               else (220,220,220))
        cv2.putText(canvas, label, (x+6, ry),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (120,120,120), 1,
                    cv2.LINE_AA)
        cv2.putText(canvas, val, (x+6, ry+14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, col, 1, cv2.LINE_AA)


def draw_warnings(canvas, lane_an, lead, engaged):
    h, w  = canvas.shape[:2]
    warns = []
    if lane_an.departure:
        warns.append(("LANE DEPARTURE", C_WARN))
    if lead.ttc_s < 3.0:
        warns.append(("COLLISION RISK", (0, 0, 255)))
    elif lead.ttc_s < 6.0:
        warns.append(("FOLLOW CLOSE",   C_WARN))
    if not engaged:
        warns.append(("SYSTEM STANDBY", C_STBY))
    for i, (txt, col) in enumerate(warns):
        y       = h//2 - 40 + i*34
        overlay = canvas.copy()
        cv2.rectangle(overlay, (w//2-140, y-18), (w//2+140, y+8), col, -1)
        cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0, canvas)
        cv2.putText(canvas, txt, (w//2-120, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60,
                    (255,255,255), 2, cv2.LINE_AA)


def draw_hud(canvas, fps, engaged, stub, brt, conf,
             show_da, show_ll, show_box, show_path, show_bev,
             recording, night):
    h, w = canvas.shape[:2]
    col  = C_ENGAGE if engaged else C_STBY
    cv2.putText(canvas, "ENGAGED" if engaged else "STANDBY",
                (w//2-60, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.80, col, 2, cv2.LINE_AA)
    fps_col = ((0,210,70) if fps>=15 else
               (0,160,255) if fps>=8 else (0,80,255))
    cv2.putText(canvas, f"{fps:.0f} FPS", (w-115, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, fps_col, 1, cv2.LINE_AA)
    cv2.putText(canvas, "YOLOP v4.1", (w-120, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (100,100,100), 1,
                cv2.LINE_AA)
    if stub:
        cv2.putText(canvas, "STUB — place yolop-320-320.onnx here",
                    (w//2-160, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0,100,255), 1,
                    cv2.LINE_AA)
    if recording:
        cv2.circle(canvas, (w-14, 14), 6, (0,0,220), -1, cv2.LINE_AA)
        cv2.putText(canvas, "REC", (w-38, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0,0,220), 1, cv2.LINE_AA)
    if night:
        cv2.putText(canvas, "NIGHT", (w-72, 64),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180,180,50), 1,
                    cv2.LINE_AA)
    tags = " ".join(t for t, on in [
        ("DA", show_da), ("LL", show_ll), ("DET", show_box),
        ("PATH", show_path), ("BEV", show_bev),
    ] if on)
    cv2.putText(canvas, tags or "--", (w-240, h-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (110,110,110), 1,
                cv2.LINE_AA)
    cv2.putText(canvas,
                f"BRT {'+' if brt>=0 else ''}{int(brt)}  CONF {conf:.0%}",
                (8, h-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, (110,110,110), 1,
                cv2.LINE_AA)
    cv2.putText(
        canvas,
        "[E]engage [D]da [L]ll [B]box [P]path [M]bev "
        "[N]night [R]rec [C]cal [+/-]brt [[]conf [S]save [Q]quit",
        (8, h-26),
        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (70,70,70), 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# Window helper  — create + verify before registering callbacks
# ─────────────────────────────────────────────────────────────────────────────
def _init_window(name: str, w: int, h: int) -> bool:
    """
    Create the named window and verify it is ready.
    Returns True on success, False if the display is unavailable.
    """
    try:
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, w, h)

        # Draw a black frame and flush it so the window handle is live
        blank = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.imshow(name, blank)
        cv2.waitKey(1)          # ← this forces the Qt/GTK event loop to
                                #   actually create the native window handle
        return True
    except Exception as e:
        print(f"[!] Window init failed: {e}")
        return False


def _safe_set_mouse_callback(win: str, fn) -> bool:
    """Register mouse callback only when the window handle exists."""
    try:
        cv2.setMouseCallback(win, fn)
        return True
    except cv2.error as e:
        print(f"[!] Mouse callback failed (non-fatal): {e}")
        print("    Calibration drag will be unavailable.")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--model",  default=MODEL_FILE)
    ap.add_argument("--skip",   type=int,   default=INF_EVERY)
    ap.add_argument("--conf",   type=float, default=0.40)
    args = ap.parse_args()

    sess, out_names, in_name, is_stub = load_session(args.model)

    src     = int(args.source) if args.source.isdigit() else args.source
    is_file = isinstance(src, str) and os.path.isfile(src)
    cap     = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[!] Cannot open: {args.source}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  DISPLAY_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_H)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if platform.system() == "Linux" and not is_file:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)

    WIN = "openpilot · YOLOP v4.1"

    # ── Create window FIRST, flush it, THEN register callback ─────────────────
    if not _init_window(WIN, DISPLAY_W, DISPLAY_H):
        print("[!] Cannot create display window — exiting.")
        sys.exit(1)

    # ── Subsystems ────────────────────────────────────────────────────────────
    speed_est    = SpeedEstimator()
    ipm          = IPMTransform(DISPLAY_W, DISPLAY_H)
    lane_an      = LaneAnalyser(DISPLAY_W, DISPLAY_H)
    steer_model  = SteeringModel()
    lead_tracker = LeadTracker()
    bev_panel    = BEVPanel(ipm, DISPLAY_W, DISPLAY_H)
    adapt_disp   = AdaptiveDisplay()
    recorder     = DashcamRecorder()
    evt_log      = EventLogger()
    cal_ui       = CalibrationUI(ipm)

    # ── Register mouse callback AFTER window is confirmed alive ────────────────
    _safe_set_mouse_callback(WIN, cal_ui.mouse)

    # ── UI state ──────────────────────────────────────────────────────────────
    engaged    = True
    show_da    = True
    show_ll    = True
    show_box   = True
    show_path  = True
    show_bev   = True
    show_warn  = True
    brt_offset = 0.0
    conf       = args.conf

    fps_t   = time.time()
    fps     = 0.0
    frame_n = 0
    inf_n   = 0
    da_mask = ll_mask = None
    dets    = []
    fails   = 0

    print(f"[*] Source : {src}  {DISPLAY_W}x{DISPLAY_H}")
    print(f"[*] BEV    : {BEV_W}x{BEV_H}  bottom-right corner")
    print("[*] Q / ESC to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            if is_file:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            fails += 1
            if fails > 30:
                print("[!] Camera failed — exiting.")
                break
            time.sleep(0.05)
            continue
        fails = 0

        frame = cv2.resize(frame, (DISPLAY_W, DISPLAY_H),
                           interpolation=cv2.INTER_LINEAR)
        h, w  = frame.shape[:2]
        kmh   = speed_est.update(frame)

        # ── Inference ─────────────────────────────────────────────────────────
        if inf_n % args.skip == 0:
            blob = preprocess(frame)
            outs = sess.run(out_names, {in_name: blob})
            om   = dict(zip(out_names, outs))
            try:
                det_raw = om["det_out"]
                da_raw  = om["drive_area_seg"]
                ll_raw  = om["lane_line_seg"]
            except KeyError:
                det_raw, da_raw, ll_raw = outs[0], outs[1], outs[2]

            da_mask = seg_mask(da_raw, h, w)
            ll_mask = seg_mask(ll_raw, h, w)
            dets    = decode_det(det_raw, h, w, conf=conf)
            lane_an.update(ll_mask, da_mask)
        inf_n += 1

        steer_angle = steer_model.update(
            lane_an.curve_rad, lane_an.lat_offset,
            lane_an._left_fit, lane_an._right_fit)
        lead_tracker.update(dets, w, h, kmh)

        # ── Events ────────────────────────────────────────────────────────────
        if lane_an.departure:
            evt_log.log("departure", f"off={lane_an.lat_offset:+.2f}")
        if lead_tracker.ttc_s < 2.0:
            evt_log.log("near_miss",
                        f"d={lead_tracker.dist_m:.1f}m "
                        f"ttc={lead_tracker.ttc_s:.1f}s")

        # ── Render ────────────────────────────────────────────────────────────
        canvas = adapt_disp.process(frame)
        canvas = apply_brightness(canvas, brt_offset)

        draw_horizon(canvas, da_mask)
        if show_da and da_mask is not None:
            overlay_da(canvas, da_mask)
        if show_ll and ll_mask is not None:
            overlay_ll(canvas, ll_mask)

        draw_path_plan(canvas, lane_an, show_path)

        if show_box:
            draw_boxes(canvas, dets, lead_tracker.box)

        curvature_banner(canvas, lane_an.curve_rad, lane_an.lat_offset)

        if show_warn:
            draw_warnings(canvas, lane_an, lead_tracker, engaged)

        draw_steering_gauge(canvas, steer_angle)
        draw_speed_panel(canvas, kmh)
        draw_lane_panel(canvas, lane_an)
        draw_lead_panel(canvas, lead_tracker)

        # ── BEV panel ─────────────────────────────────────────────────────────
        if show_bev:
            bev_panel.render(
                da_mask, ll_mask, lane_an,
                dets, lead_tracker.box,
                canvas,
                w - BEV_W - 10,   # ox — right edge
                h - BEV_H - 10,   # oy — bottom edge
            )

        cal_ui.draw(canvas)

        draw_hud(canvas, fps, engaged, is_stub, brt_offset, conf,
                 show_da, show_ll, show_box, show_path, show_bev,
                 recorder.active, adapt_disp.night)

        recorder.write(canvas)

        # ── FPS ───────────────────────────────────────────────────────────────
        frame_n += 1
        now = time.time()
        if now - fps_t >= 0.5:
            fps     = frame_n / (now - fps_t)
            frame_n = 0
            fps_t   = now

        cv2.imshow(WIN, canvas)
        key = cv2.waitKey(1) & 0xFF

        if   key in (ord("q"), 27): break
        elif key == ord("e"):  engaged    = not engaged
        elif key == ord("d"):  show_da    = not show_da
        elif key == ord("l"):  show_ll    = not show_ll
        elif key == ord("b"):  show_box   = not show_box
        elif key == ord("p"):  show_path  = not show_path
        elif key == ord("m"):  show_bev   = not show_bev
        elif key == ord("w"):  show_warn  = not show_warn
        elif key == ord("n"):
            adapt_disp.auto  = False
            adapt_disp.night = not adapt_disp.night
        elif key == ord("r"):  recorder.toggle()
        elif key == ord("c"):  cal_ui.toggle()
        elif key == 13:        ipm.save()          # ENTER
        elif key in (ord("+"), ord("=")):
            brt_offset = min(brt_offset + 5,  80)
        elif key == ord("-"):
            brt_offset = max(brt_offset - 5, -80)
        elif key == ord("]"):  conf = min(conf + 0.05, 0.95)
        elif key == ord("["):  conf = max(conf - 0.05, 0.05)
        elif key == ord("s"):
            fname = f"screenshot_{int(time.time())}.jpg"
            cv2.imwrite(fname, canvas)
            print(f"[+] {fname}")

    recorder.stop()
    evt_log.close()
    cap.release()
    cv2.destroyAllWindows()
    print("[*] Done.")


if __name__ == "__main__":
    main()

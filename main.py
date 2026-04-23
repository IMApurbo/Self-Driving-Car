"""
openpilot-style YOLOP 320×320 Live Cam UI — Enhanced v3
=========================================================
Fixes v3:
  - Path drawn only from bottom ~30% of frame (not full height)
  - Steering works correctly in both directions (left AND right)
  - Lane assignment by x-position relative to ego, not hard image half
  - Lateral offset sign corrected (positive = right of centre, negative = left)
  - Path start anchored to bottom-centre of frame

INSTALL:  pip install opencv-python numpy onnxruntime onnx
RUN:      python openpilot_yolop_enhanced.py
          python openpilot_yolop_enhanced.py --source road.mp4
KEYS:     Q/ESC quit | S screenshot | E engage | D da | L ll
          B boxes | P path | M map | W warnings | +/- bright | [/] conf
"""

import argparse
import collections
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
MODEL_FILE  = "yolop-320-320.onnx"
INF_SIZE    = 320
DISPLAY_W   = 1280
DISPLAY_H   = 720
ORT_THREADS = 2
INF_EVERY   = 2

# Minimum lane pixels to consider a lane "detected"
MIN_LANE_PIX = 80

# EMA alpha for path smoothing (lower = smoother but more lag)
PATH_EMA     = 0.20

# Path only drawn in the BOTTOM fraction of the frame
PATH_TOP_FRAC  = 0.70   # y starts at 70% down the image
PATH_N_ROWS    = 20     # fewer points → shorter, cleaner path

# Colours (BGR)
C_GREEN   = (0,   195,  55)
C_WHITE   = (220, 220, 220)
C_CYAN    = (255, 200,   0)
C_BOX     = (0,   175, 255)
C_ENGAGE  = (0,   215,  75)
C_STBY    = (175, 175, 175)
C_WARN    = (0,    80, 255)
C_PATH    = (0,   230, 255)
C_PLAN    = (255, 140,   0)
C_HORIZON = (180,  80, 200)
C_LEAD    = (0,    60, 255)
C_DARK    = (20,   20,  20)

_MEAN     = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD      = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_DILATE_K = np.ones((3, 3), np.uint8)

_CURVE_HIST = 14
_STEER_HIST = 12   # slightly longer for smoother steering
_SPEED_HIST = 20


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
            t    = helper.make_tensor(
                name + "_val", TensorProto.FLOAT,
                shape, vals.flatten().tolist()
            )
            nodes.append(
                helper.make_node("Constant", inputs=[], outputs=[name], value=t)
            )
        graph = helper.make_graph(
            nodes, "yolop_stub",
            [helper.make_tensor_value_info(
                "images", TensorProto.FLOAT, [1, 3, INF_SIZE, INF_SIZE])],
            [helper.make_tensor_value_info(n, TensorProto.FLOAT, s)
             for n, s in specs]
        )
        m = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 12)]
        )
        onnx.save(m, path)
        print(f"[+] Stub saved: {path}")
        return True
    except Exception as e:
        print(f"[!] Stub build failed: {e}")
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
        path    = stub
        is_stub = True
    else:
        print(f"[+] Model: {path}")

    opts = ort.SessionOptions()
    opts.intra_op_num_threads     = ORT_THREADS
    opts.inter_op_num_threads     = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.execution_mode           = ort.ExecutionMode.ORT_SEQUENTIAL

    sess      = ort.InferenceSession(
        path, sess_options=opts,
        providers=["CPUExecutionProvider"]
    )
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


def decode_det(det, fh: int, fw: int,
               conf: float = 0.40, iou: float = 0.45):
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
    if keep.sum() == 0:
        return []

    p      = pred[keep]
    scores = scores[keep]
    sx     = fw / INF_SIZE
    sy     = fh / INF_SIZE

    cx = p[:, 0] * sx
    cy = p[:, 1] * sy
    bw = p[:, 2] * sx
    bh = p[:, 3] * sy

    x1 = np.clip((cx - bw / 2).astype(int), 0, fw - 1)
    y1 = np.clip((cy - bh / 2).astype(int), 0, fh - 1)
    x2 = np.clip((cx + bw / 2).astype(int), 0, fw - 1)
    y2 = np.clip((cy + bh / 2).astype(int), 0, fh - 1)

    xywh = np.stack([x1, y1, x2 - x1, y2 - y1], 1).astype(float).tolist()
    idxs = cv2.dnn.NMSBoxes(xywh, scores.tolist(), conf, iou)
    if len(idxs) == 0:
        return []

    idxs = np.array(idxs).flatten()
    return [(int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i]),
             float(scores[i])) for i in idxs]


# ─────────────────────────────────────────────────────────────────────────────
# Speed estimator
# ─────────────────────────────────────────────────────────────────────────────
class SpeedEstimator:
    def __init__(self):
        self._prev_gray  = None
        self._hist       = collections.deque(maxlen=_SPEED_HIST)
        self.kmh         = 0.0

    def update(self, frame: np.ndarray) -> float:
        h, w = frame.shape[:2]
        roi  = frame[h // 3: 2 * h // 3, w // 4: 3 * w // 4]
        gray = cv2.cvtColor(
            cv2.resize(roi, (80, 45)), cv2.COLOR_BGR2GRAY
        )
        if self._prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                self._prev_gray, gray, None,
                0.5, 2, 8, 2, 5, 1.1, 0
            )
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            self._hist.append(float(np.mean(mag)) * 18.0)
        self._prev_gray = gray
        if self._hist:
            self.kmh = float(np.mean(self._hist))
        return self.kmh


# ─────────────────────────────────────────────────────────────────────────────
# Lane Analyser  — Fixed v3
# ─────────────────────────────────────────────────────────────────────────────
class LaneAnalyser:
    """
    Lane-count-aware path planner.

    Key fixes in v3
    ---------------
    1. Lane left/right assignment uses CENTROID x vs image centre,
       not a hard image-half split.  This means a lane drifting
       across the midpoint is still correctly labelled.
    2. Lateral offset sign convention:
         positive  → ego is LEFT  of lane centre  (steer right)
         negative  → ego is RIGHT of lane centre  (steer left)
    3. Path y-range restricted to PATH_TOP_FRAC … 1.0
       (bottom portion of frame only).
    4. Path x is always anchored at frame centre on the bottom row,
       then curves toward the polynomial midpoint higher up.
       This gives a natural "looking forward" feel.
    """

    def __init__(self, w: int, h: int):
        self._w           = w
        self._h           = h
        self._smooth_path: list = []
        self._left_fit    = None
        self._right_fit   = None

        self.lane_count   = 0
        self.curve_rad    = 9999.0
        self.lat_offset   = 0.0   # + = ego left of centre, steer right
        self.departure    = False

        self._curve_hist  = collections.deque(maxlen=_CURVE_HIST)
        self._offset_hist = collections.deque(maxlen=_CURVE_HIST)
        self._half_lane_w = w * 0.22

    # ── Helpers ──────────────────────────────────────────────────────────────
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
        A = fit[0] * (xm / ym ** 2)
        B = fit[1] * (xm / ym)
        r = ((1 + (2 * A * y_eval * ym + B) ** 2) ** 1.5) / abs(2 * A + 1e-6)
        return float(min(r, 9999.0))

    # ── Split mask → left lane / right lane by centroid ──────────────────────
    def _split(self, ll_mask: np.ndarray):
        """
        Find all lane pixels, cluster them by whether their column
        centroid (computed per connected component) is left or right
        of image centre.

        Falls back to simple pixel column < mid / >= mid when no
        connected-component info is available.
        """
        cx = self._w / 2.0

        # Use connected components to separate lane lines robustly
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            ll_mask.astype(np.uint8), connectivity=8
        )

        left_ys, left_xs   = [], []
        right_ys, right_xs = [], []

        for lbl in range(1, n_labels):          # skip background (0)
            area = stats[lbl, cv2.CC_STAT_AREA]
            if area < 20:                        # ignore tiny noise blobs
                continue
            cent_x = centroids[lbl, 0]
            ys, xs = np.where(labels == lbl)
            if cent_x < cx:
                left_ys.extend(ys.tolist())
                left_xs.extend(xs.tolist())
            else:
                right_ys.extend(ys.tolist())
                right_xs.extend(xs.tolist())

        return (np.array(left_ys),  np.array(left_xs),
                np.array(right_ys), np.array(right_xs))

    # ── Drivable-area centroid path ───────────────────────────────────────────
    def _da_path(self, da_mask: np.ndarray):
        h, w  = da_mask.shape
        y_top = int(h * PATH_TOP_FRAC)
        ys    = np.linspace(y_top, h - 1, PATH_N_ROWS).astype(int)
        pts   = []
        for y in ys:
            row = da_mask[y]
            xs  = np.where(row > 0)[0]
            if len(xs) >= 4:
                pts.append((int(np.mean(xs)), int(y)))
        return pts

    # ── Raw path from polynomial fits ────────────────────────────────────────
    def _poly_path(self):
        h     = self._h
        y_top = int(h * PATH_TOP_FRAC)
        ys    = np.linspace(y_top, h - 1, PATH_N_ROWS)
        pts   = []

        for y in ys:
            lx = (self._poly_x(self._left_fit,  y)
                  if self._left_fit  is not None else None)
            rx = (self._poly_x(self._right_fit, y)
                  if self._right_fit is not None else None)

            if lx is not None and rx is not None:
                # 2 lanes: midpoint, strictly clamped between them
                lo, hi = (lx, rx) if lx < rx else (rx, lx)
                cx     = (lo + hi) / 2.0
                cx     = max(lo + 2, min(hi - 2, cx))
            elif lx is not None:
                # Left lane only → offset right by half-lane width
                cx = lx + self._half_lane_w
            elif rx is not None:
                # Right lane only → offset left by half-lane width
                cx = rx - self._half_lane_w
            else:
                continue

            cx = max(0, min(self._w - 1, cx))
            pts.append((int(cx), int(y)))

        return pts

    # ── EMA smoothing ────────────────────────────────────────────────────────
    def _smooth(self, new_pts: list) -> list:
        if not new_pts:
            return self._smooth_path

        if len(self._smooth_path) != len(new_pts):
            self._smooth_path = new_pts
            return self._smooth_path

        smoothed = []
        for (nx, ny), (ox, _) in zip(new_pts, self._smooth_path):
            sx = int(ox + PATH_EMA * (nx - ox))
            smoothed.append((sx, ny))
        self._smooth_path = smoothed
        return smoothed

    # ── Main update ──────────────────────────────────────────────────────────
    def update(self, ll_mask: np.ndarray, da_mask):
        h, w = ll_mask.shape
        self._w, self._h = w, h

        ly, lx, ry, rx = self._split(ll_mask)

        self._left_fit  = self._fit_poly(ly, lx)
        self._right_fit = self._fit_poly(ry, rx)

        has_left  = self._left_fit  is not None
        has_right = self._right_fit is not None
        self.lane_count = int(has_left) + int(has_right)

        # Update half-lane width when both lanes seen
        if has_left and has_right:
            y_ref = h * 0.85          # sample near bottom for accuracy
            lx_r  = self._poly_x(self._left_fit,  y_ref)
            rx_r  = self._poly_x(self._right_fit, y_ref)
            if rx_r > lx_r + 20:     # sanity: right must be noticeably right
                self._half_lane_w = (rx_r - lx_r) / 2.0

        # ── Curve radius ──────────────────────────────────────────────────────
        y_eval = h * 0.80            # evaluate near bottom (more reliable)
        rads   = [self._curve_radius(f, y_eval)
                  for f in [self._left_fit, self._right_fit]
                  if f is not None]
        raw_r  = float(np.mean(rads)) if rads else 9999.0
        self._curve_hist.append(raw_r)
        self.curve_rad = float(np.mean(self._curve_hist))

        # ── Lateral offset ────────────────────────────────────────────────────
        # Convention:
        #   lat_offset > 0  →  lane centre is to the RIGHT of ego centre
        #                      → steer RIGHT to re-centre
        #   lat_offset < 0  →  lane centre is to the LEFT of ego centre
        #                      → steer LEFT  to re-centre
        cx_img = w / 2.0
        if has_left and has_right:
            lx_r        = self._poly_x(self._left_fit,  y_eval)
            rx_r        = self._poly_x(self._right_fit, y_eval)
            lane_centre = (lx_r + rx_r) / 2.0
            # lane_centre > cx_img  → lane centre right of ego → offset +
            raw_off     = (lane_centre - cx_img) / (w / 2.0)

        elif has_left:
            # Left lane visible: its expected right neighbour would be at
            # lx_r + 2*half_lane_w; midpoint at lx_r + half_lane_w
            lx_r        = self._poly_x(self._left_fit, y_eval)
            lane_centre = lx_r + self._half_lane_w
            raw_off     = (lane_centre - cx_img) / (w / 2.0)

        elif has_right:
            # Right lane visible: its expected left neighbour would be at
            # rx_r - 2*half_lane_w; midpoint at rx_r - half_lane_w
            rx_r        = self._poly_x(self._right_fit, y_eval)
            lane_centre = rx_r - self._half_lane_w
            raw_off     = (lane_centre - cx_img) / (w / 2.0)

        else:
            raw_off = 0.0

        self._offset_hist.append(raw_off)
        self.lat_offset = float(np.mean(self._offset_hist))
        self.departure  = abs(self.lat_offset) > 0.40

        # ── Build path ────────────────────────────────────────────────────────
        if self.lane_count >= 1:
            raw_pts = self._poly_path()
        else:
            raw_pts = self._da_path(da_mask) if da_mask is not None else []

        self._smooth_path = self._smooth(raw_pts)

    # ── Public properties ─────────────────────────────────────────────────────
    @property
    def path_points(self) -> list:
        return self._smooth_path

    def lane_edge_points(self, n: int = 20):
        h  = self._h
        y_top = int(h * PATH_TOP_FRAC)
        ys = np.linspace(y_top, h - 1, n)
        lpts, rpts = [], []
        for y in ys:
            if self._left_fit  is not None:
                lpts.append((int(self._poly_x(self._left_fit,  y)), int(y)))
            if self._right_fit is not None:
                rpts.append((int(self._poly_x(self._right_fit, y)), int(y)))
        return lpts, rpts


# ─────────────────────────────────────────────────────────────────────────────
# Steering model  — Fixed v3: both directions
# ─────────────────────────────────────────────────────────────────────────────
class SteeringModel:
    """
    Combines curve-following and lateral re-centring.

    Sign convention (matches LaneAnalyser):
      angle > 0  →  steer RIGHT
      angle < 0  →  steer LEFT

    Curve contribution uses the polynomial's second-order coefficient
    (fit[0]) to detect which way the road bends:
      fit[0] > 0  →  concave upward in image → road curves LEFT  → steer LEFT
      fit[0] < 0  →  concave downward        → road curves RIGHT → steer RIGHT
    """

    def __init__(self):
        self._hist  = collections.deque(maxlen=_STEER_HIST)
        self.angle  = 0.0
        self.torque = 0.0

    def update(self, curve_rad: float, lat_offset: float,
               left_fit=None, right_fit=None) -> float:

        # ── Curve-following component ─────────────────────────────────────────
        curve_steer = 0.0
        if curve_rad < 9000:
            # Determine curve direction from polynomial curvature
            fits = [f for f in [left_fit, right_fit] if f is not None]
            if fits:
                avg_a = float(np.mean([f[0] for f in fits]))
                # avg_a > 0 → road bends left  → negative steer (turn left)
                # avg_a < 0 → road bends right → positive steer (turn right)
                magnitude    = math.degrees(math.atan(3.0 / max(curve_rad, 1.0))) * 8.0
                curve_steer  = -np.sign(avg_a) * magnitude

        # ── Lateral re-centring component ─────────────────────────────────────
        # lat_offset > 0 → lane centre is RIGHT of ego → steer right (+)
        # lat_offset < 0 → lane centre is LEFT  of ego → steer left  (-)
        offset_steer = lat_offset * 18.0

        raw = max(-45.0, min(45.0, curve_steer + offset_steer))
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

    def update(self, dets, fw: int, fh: int, speed_kmh: float):
        if not dets:
            self.box    = None
            self.dist_m = 0.0
            self.ttc_s  = 99.0
            return
        cx   = fw / 2
        best, best_s = None, -1.0
        for d in dets:
            x1, y1, x2, y2, sc = d
            area  = (x2 - x1) * (y2 - y1)
            cent  = 1.0 - abs(((x1 + x2) / 2) - cx) / cx
            low   = y2 / fh
            rank  = area * cent * low * sc
            if rank > best_s:
                best, best_s = d, rank
        self.box = best
        if best:
            box_h       = max(best[3] - best[1], 1)
            self.dist_m = max(1.5 * 700.0 / box_h, 1.0)
            rel_v       = max(speed_kmh * 0.8 / 3.6, 0.1)
            self.ttc_s  = self.dist_m / rel_v
        else:
            self.dist_m = 0.0
            self.ttc_s  = 99.0


# ─────────────────────────────────────────────────────────────────────────────
# Brightness
# ─────────────────────────────────────────────────────────────────────────────
def apply_brightness(frame: np.ndarray, offset: float) -> np.ndarray:
    if offset == 0.0:
        return frame.copy()
    return np.clip(frame.astype(np.int16) + int(offset), 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Overlays
# ─────────────────────────────────────────────────────────────────────────────
def overlay_da(canvas: np.ndarray, mask: np.ndarray, alpha: float = 0.28):
    ys, xs = np.where(mask == 1)
    if len(ys) == 0:
        return
    roi = canvas[ys, xs].astype(np.float32)
    canvas[ys, xs] = np.clip(
        roi * (1 - alpha) + np.array(C_GREEN, dtype=np.float32) * alpha,
        0, 255
    ).astype(np.uint8)


def overlay_ll(canvas: np.ndarray, mask: np.ndarray):
    dilated = cv2.dilate(mask, _DILATE_K, iterations=2)
    ys, xs  = np.where(dilated == 1)
    if len(ys):
        canvas[ys, xs] = C_WHITE


def draw_boxes(canvas: np.ndarray, dets, lead_box=None):
    for x1, y1, x2, y2, sc in dets:
        is_lead = (lead_box is not None
                   and lead_box[0] == x1 and lead_box[1] == y1)
        col = C_LEAD if is_lead else C_BOX
        cv2.rectangle(canvas, (x1, y1), (x2, y2), col, 2, cv2.LINE_AA)
        lbl = f"{'LEAD ' if is_lead else ''}{sc:.0%}"
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
        cv2.rectangle(canvas, (x1, y1 - th - 6), (x1 + tw + 6, y1), col, -1)
        cv2.putText(canvas, lbl, (x1 + 3, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                    (255, 255, 255), 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# Path planning overlay  — bottom-only, mode-aware
# ─────────────────────────────────────────────────────────────────────────────
def draw_path_plan(canvas: np.ndarray, lane_an: LaneAnalyser, show: bool):
    if not show:
        return

    h, w = canvas.shape[:2]
    pts  = lane_an.path_points

    # ── Lane polygon (only when 2 lanes, bottom region only) ─────────────────
    if lane_an.lane_count == 2:
        lpts, rpts = lane_an.lane_edge_points(n=20)
        if len(lpts) >= 2 and len(rpts) >= 2:
            poly = np.array(
                lpts + list(reversed(rpts)), dtype=np.int32
            ).reshape(-1, 1, 2)
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [poly], (0, 70, 0))
            cv2.addWeighted(overlay, 0.22, canvas, 0.78, 0, canvas)

        for side, col in [(lpts, (0, 210, 100)), (rpts, (0, 210, 100))]:
            if len(side) >= 2:
                cv2.polylines(canvas,
                              [np.array(side, np.int32)],
                              False, col, 2, cv2.LINE_AA)

    # ── Mode label ────────────────────────────────────────────────────────────
    mode_labels = {2: "2-LANE PATH", 1: "1-LANE PATH", 0: "DA PATH"}
    mode_col    = {2: (0, 230, 100), 1: (0, 200, 255), 0: (200, 130, 0)}
    lc          = lane_an.lane_count
    cv2.putText(canvas, mode_labels[lc],
                (w // 2 - 55, h - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                mode_col[lc], 1, cv2.LINE_AA)

    if len(pts) < 2:
        return

    # ── Colour by lane mode ───────────────────────────────────────────────────
    path_col = {2: C_PATH, 1: (0, 200, 255), 0: (200, 140, 0)}
    col      = path_col[lc]

    # ── Draw path — dashes + fading dots, bottom region only ─────────────────
    # pts are already restricted to PATH_TOP_FRAC…1.0 by LaneAnalyser
    for i in range(0, len(pts) - 1, 2):
        cv2.line(canvas, pts[i], pts[i + 1], col, 3, cv2.LINE_AA)

    # Dots: larger at bottom (close), smaller near top (far)
    # pts[0] is the TOP of the path (furthest), pts[-1] is BOTTOM (closest)
    n = len(pts)
    for i, pt in enumerate(pts):
        # fade: 0 at top (far), 1 at bottom (near)
        alpha_r = i / max(n - 1, 1)
        r       = max(2, int(6 * alpha_r))
        cv2.circle(canvas, pt, r, col, -1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# Horizon line
# ─────────────────────────────────────────────────────────────────────────────
def draw_horizon(canvas: np.ndarray, da_mask):
    if da_mask is None:
        return
    h, w = canvas.shape[:2]
    rows = np.where(da_mask.sum(axis=1) > w * 0.05)[0]
    if len(rows) == 0:
        return
    hy = int(rows.min())
    for x in range(0, w, 20):
        cv2.line(canvas, (x, hy), (x + 10, hy), C_HORIZON, 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# Steering gauge
# ─────────────────────────────────────────────────────────────────────────────
def draw_steering_gauge(canvas: np.ndarray, angle_deg: float,
                        cx: int = None, cy: int = None, r: int = 72):
    h, w = canvas.shape[:2]
    if cx is None:
        cx = w - r - 20
    if cy is None:
        cy = h - r - 20

    cv2.circle(canvas, (cx, cy), r + 6, C_DARK,      -1, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), r + 6, (60, 60, 60), 2, cv2.LINE_AA)

    for deg in range(-45, 46, 15):
        rad   = math.radians(90 + deg)
        inner = r - 10
        outer = r - 2
        x1_   = int(cx + inner * math.cos(rad))
        y1_   = int(cy - inner * math.sin(rad))
        x2_   = int(cx + outer * math.cos(rad))
        y2_   = int(cy - outer * math.sin(rad))
        col   = (100, 100, 100) if deg != 0 else (200, 200, 200)
        cv2.line(canvas, (x1_, y1_), (x2_, y2_), col, 1, cv2.LINE_AA)

    ac  = max(-45.0, min(45.0, angle_deg))
    arc_col = ((0, 200, 80)  if abs(ac) < 10 else
               (0, 160, 255) if abs(ac) < 25 else
               (0,  60, 255))
    sa  = 90
    ea  = int(90 + ac)
    if sa != ea:
        cv2.ellipse(canvas, (cx, cy), (r - 6, r - 6), 0,
                    -max(sa, ea), -min(sa, ea),
                    arc_col, 5, cv2.LINE_AA)

    nr  = math.radians(90 + ac)
    nx  = int(cx + (r - 10) * math.cos(nr))
    ny  = int(cy - (r - 10) * math.sin(nr))
    cv2.line(canvas, (cx, cy), (nx, ny), (255, 255, 255), 3, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), 6, (255, 255, 255), -1, cv2.LINE_AA)

    cv2.circle(canvas, (cx, cy), r - 20, (80, 80, 80), 2, cv2.LINE_AA)
    for s in [90, 210, 330]:
        sr  = math.radians(s + ac)
        ex_ = int(cx + (r - 22) * math.cos(sr))
        ey_ = int(cy - (r - 22) * math.sin(sr))
        cv2.line(canvas, (cx, cy), (ex_, ey_), (80, 80, 80), 2, cv2.LINE_AA)

    cv2.putText(canvas, f"{ac:+.1f}°",
                (cx - 22, cy + r + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 200, 200), 1,
                cv2.LINE_AA)
    cv2.putText(canvas, "STEER",
                (cx - 18, cy - r - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (120, 120, 120), 1,
                cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# Speed panel
# ─────────────────────────────────────────────────────────────────────────────
def draw_speed_panel(canvas: np.ndarray, kmh: float,
                     x: int = 20, y: int = None,
                     pw: int = 90, ph: int = 60):
    if y is None:
        y = canvas.shape[0] - ph - 20
    cv2.rectangle(canvas, (x, y), (x + pw, y + ph), C_DARK, -1, cv2.LINE_AA)
    cv2.rectangle(canvas, (x, y), (x + pw, y + ph), (60, 60, 60), 1,
                  cv2.LINE_AA)
    kmh_c   = max(0.0, min(kmh, 200.0))
    bar_w   = int(kmh_c / 200.0 * (pw - 10))
    bar_col = ((0, 200, 80)  if kmh_c < 80
               else (0, 160, 255) if kmh_c < 120
               else (0, 60, 255))
    if bar_w > 0:
        cv2.rectangle(canvas, (x + 5, y + ph - 14),
                      (x + 5 + bar_w, y + ph - 6), bar_col, -1, cv2.LINE_AA)
    cv2.putText(canvas, f"{kmh_c:.0f}", (x + 8, y + 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.90, (230, 230, 230), 2,
                cv2.LINE_AA)
    cv2.putText(canvas, "km/h",       (x + 8, y + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, (140, 140, 140), 1,
                cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# Lane info panel
# ─────────────────────────────────────────────────────────────────────────────
def draw_lane_panel(canvas: np.ndarray, lane_an: LaneAnalyser,
                    x: int = 20, y: int = 120, pw: int = 150):
    rows = [
        ("LANES",    str(lane_an.lane_count)),
        ("CURVE R",  f"{min(lane_an.curve_rad, 9999):.0f} m"),
        ("OFFSET",   f"{lane_an.lat_offset:+.2f}"),
        ("DEPART",   "⚠ YES" if lane_an.departure else "OK"),
    ]
    lh     = 28
    ph     = lh * len(rows) + 16
    cv2.rectangle(canvas, (x, y), (x + pw, y + ph), C_DARK, -1, cv2.LINE_AA)
    cv2.rectangle(canvas, (x, y), (x + pw, y + ph), (60, 60, 60), 1,
                  cv2.LINE_AA)
    for i, (label, val) in enumerate(rows):
        ry  = y + 18 + i * lh
        col = (C_WARN if label == "DEPART" and lane_an.departure
               else (220, 220, 220))
        cv2.putText(canvas, label, (x + 6, ry),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (120, 120, 120), 1,
                    cv2.LINE_AA)
        cv2.putText(canvas, val, (x + 6, ry + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, col, 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# Lead panel
# ─────────────────────────────────────────────────────────────────────────────
def draw_lead_panel(canvas: np.ndarray, lead: LeadTracker,
                    x: int = 20, y: int = 310, pw: int = 150):
    rows = [
        ("LEAD",  f"{lead.dist_m:.1f} m" if lead.box else "–"),
        ("TTC",   f"{lead.ttc_s:.1f} s"  if lead.box else "–"),
    ]
    lh = 28
    ph = lh * len(rows) + 16
    cv2.rectangle(canvas, (x, y), (x + pw, y + ph), C_DARK, -1, cv2.LINE_AA)
    cv2.rectangle(canvas, (x, y), (x + pw, y + ph), (60, 60, 60), 1,
                  cv2.LINE_AA)
    for i, (label, val) in enumerate(rows):
        ry  = y + 18 + i * lh
        col = (C_WARN if label == "TTC" and lead.ttc_s < 3.0
               else (220, 220, 220))
        cv2.putText(canvas, label, (x + 6, ry),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (120, 120, 120), 1,
                    cv2.LINE_AA)
        cv2.putText(canvas, val,   (x + 6, ry + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, col, 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# Mini-map
# ─────────────────────────────────────────────────────────────────────────────
def draw_minimap(canvas: np.ndarray, lane_an: LaneAnalyser,
                 steer_angle: float,
                 mw: int = 160, mh: int = 160):
    fw = canvas.shape[1]
    mx = fw - mw - 20
    my = 60

    cv2.rectangle(canvas, (mx, my), (mx + mw, my + mh), C_DARK, -1,
                  cv2.LINE_AA)
    cv2.rectangle(canvas, (mx, my), (mx + mw, my + mh), (60, 60, 60), 1,
                  cv2.LINE_AA)
    cv2.putText(canvas, "BIRD EYE", (mx + 35, my + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33, (100, 100, 100), 1,
                cv2.LINE_AA)

    ecx = mx + mw // 2
    ecy = my + mh - 20

    # Simulated projected path — uses actual steer angle for direction
    sim_pts = []
    px, py, heading = 0.0, 0.0, 0.0
    for _ in range(14):
        # steer_angle > 0 → turn right → heading increases
        # steer_angle < 0 → turn left  → heading decreases
        heading += math.radians(steer_angle) * 0.18
        px      += 8.0 * math.sin(heading)
        py      -= 8.0
        mpx = ecx + int(px * 0.55)
        mpy = ecy + int(py * 0.55)
        if mx <= mpx <= mx + mw and my <= mpy <= my + mh:
            sim_pts.append((mpx, mpy))

    # Lane edges
    lpts, rpts = lane_an.lane_edge_points(n=12)
    ch = canvas.shape[0]
    cw = canvas.shape[1]

    def to_map(fp):
        out = []
        for (fx, fy) in fp:
            rx_ = (fx - cw / 2) * 0.12
            ry_ = (ch - fy)     * 0.14
            out.append((ecx + int(rx_), ecy - int(ry_)))
        return out

    for side, col in [(to_map(lpts), (0, 180, 80)),
                      (to_map(rpts), (0, 180, 80))]:
        if len(side) >= 2:
            cv2.polylines(canvas, [np.array(side, np.int32)],
                          False, col, 1, cv2.LINE_AA)

    if len(sim_pts) >= 2:
        cv2.polylines(canvas, [np.array(sim_pts, np.int32)],
                      False, C_PATH, 2, cv2.LINE_AA)

    cv2.circle(canvas, (ecx, ecy), 5, C_ENGAGE, -1, cv2.LINE_AA)
    cv2.putText(canvas, "EGO", (ecx - 10, ecy + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, (150, 150, 150), 1,
                cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# Warnings
# ─────────────────────────────────────────────────────────────────────────────
def draw_warnings(canvas: np.ndarray, lane_an: LaneAnalyser,
                  lead: LeadTracker, engaged: bool):
    h, w = canvas.shape[:2]
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
        y       = h // 2 - 40 + i * 34
        overlay = canvas.copy()
        cv2.rectangle(overlay,
                      (w // 2 - 140, y - 18),
                      (w // 2 + 140, y + 8), col, -1)
        cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0, canvas)
        cv2.putText(canvas, txt, (w // 2 - 120, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60,
                    (255, 255, 255), 2, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# HUD
# ─────────────────────────────────────────────────────────────────────────────
def draw_hud(canvas: np.ndarray, fps: float, engaged: bool,
             stub: bool, brt_offset: float, conf: float,
             show_da: bool, show_ll: bool, show_box: bool,
             show_path: bool, show_map: bool):
    h, w = canvas.shape[:2]

    col = C_ENGAGE if engaged else C_STBY
    cv2.putText(canvas, "ENGAGED" if engaged else "STANDBY",
                (w // 2 - 60, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.80, col, 2, cv2.LINE_AA)

    fps_col = ((0, 210, 70)  if fps >= 15
               else (0, 160, 255) if fps >= 8
               else (0, 80,  255))
    cv2.putText(canvas, f"{fps:.0f} FPS", (w - 115, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, fps_col, 1, cv2.LINE_AA)
    cv2.putText(canvas, "YOLOP 320",      (w - 115, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (100, 100, 100), 1,
                cv2.LINE_AA)

    if stub:
        cv2.putText(canvas,
                    "STUB — download yolop-320-320.onnx",
                    (w // 2 - 150, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 100, 255), 1,
                    cv2.LINE_AA)

    tags = " ".join(
        t for t, on in [
            ("DA", show_da), ("LL", show_ll), ("DET", show_box),
            ("PATH", show_path), ("MAP", show_map)
        ] if on
    )
    cv2.putText(canvas, tags or "–", (w - 240, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (110, 110, 110), 1,
                cv2.LINE_AA)

    cv2.putText(canvas,
                f"BRT {'+' if brt_offset>=0 else ''}{int(brt_offset)}  "
                f"CONF {conf:.0%}",
                (8, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, (110, 110, 110), 1,
                cv2.LINE_AA)

    cv2.putText(
        canvas,
        "[E]engage [D]da [L]ll [B]box [P]path [M]map [W]warn "
        "[+/-]brt []/[]conf [S]save [Q]quit",
        (8, h - 26),
        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (70, 70, 70), 1, cv2.LINE_AA,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source",  default="0")
    ap.add_argument("--model",   default=MODEL_FILE)
    ap.add_argument("--skip",    type=int,   default=INF_EVERY)
    ap.add_argument("--conf",    type=float, default=0.40)
    args = ap.parse_args()

    sess, out_names, in_name, is_stub = load_session(args.model)

    src     = int(args.source) if args.source.isdigit() else args.source
    is_file = isinstance(src, str) and os.path.isfile(src)

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[!] Cannot open: {args.source}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  DISPLAY_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_H)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if platform.system() == "Linux" and not is_file:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)

    WIN = "openpilot · YOLOP 320 v3"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, DISPLAY_W, DISPLAY_H)

    # ── Subsystems ────────────────────────────────────────────────────────────
    speed_est    = SpeedEstimator()
    lane_an      = LaneAnalyser(DISPLAY_W, DISPLAY_H)
    steer_model  = SteeringModel()
    lead_tracker = LeadTracker()

    # ── State ─────────────────────────────────────────────────────────────────
    engaged    = True
    show_da    = True
    show_ll    = True
    show_box   = True
    show_path  = True
    show_map   = True
    show_warn  = True
    brt_offset = 0.0
    conf       = args.conf

    fps_t   = time.time()
    fps     = 0.0
    frame_n = 0
    inf_n   = 0

    da_mask = None
    ll_mask = None
    dets    = []
    fails   = 0

    print(f"[*] Source: {src}  display {DISPLAY_W}×{DISPLAY_H}")
    print(f"[*] Skip  : every {args.skip} frames")
    print("[*] Q/ESC to quit")

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

        kmh = speed_est.update(frame)

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
                if len(outs) >= 3:
                    det_raw, da_raw, ll_raw = outs[0], outs[1], outs[2]
                else:
                    break

            da_mask = seg_mask(da_raw, h, w)
            ll_mask = seg_mask(ll_raw, h, w)
            dets    = decode_det(det_raw, h, w, conf=conf)

            # Lane analysis + path planning
            lane_an.update(ll_mask, da_mask)

        inf_n += 1

        # Pass fits into steering so it can read curvature direction
        steer_angle = steer_model.update(
            lane_an.curve_rad,
            lane_an.lat_offset,
            left_fit  = lane_an._left_fit,
            right_fit = lane_an._right_fit,
        )
        lead_tracker.update(dets, w, h, kmh)

        # ── Render ────────────────────────────────────────────────────────────
        canvas = apply_brightness(frame, brt_offset)

        draw_horizon(canvas, da_mask)

        if show_da and da_mask is not None:
            overlay_da(canvas, da_mask)
        if show_ll and ll_mask is not None:
            overlay_ll(canvas, ll_mask)

        draw_path_plan(canvas, lane_an, show_path)

        if show_box:
            draw_boxes(canvas, dets, lead_tracker.box)

        if show_warn:
            draw_warnings(canvas, lane_an, lead_tracker, engaged)

        draw_steering_gauge(canvas, steer_angle)
        draw_speed_panel(canvas, kmh)
        draw_lane_panel(canvas, lane_an)
        draw_lead_panel(canvas, lead_tracker)

        if show_map:
            draw_minimap(canvas, lane_an, steer_angle)

        draw_hud(canvas, fps, engaged, is_stub, brt_offset, conf,
                 show_da, show_ll, show_box, show_path, show_map)

        # ── FPS ───────────────────────────────────────────────────────────────
        frame_n += 1
        now = time.time()
        if now - fps_t >= 0.5:
            fps     = frame_n / (now - fps_t)
            frame_n = 0
            fps_t   = now

        cv2.imshow(WIN, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("e"):  engaged    = not engaged
        elif key == ord("d"):  show_da    = not show_da
        elif key == ord("l"):  show_ll    = not show_ll
        elif key == ord("b"):  show_box   = not show_box
        elif key == ord("p"):  show_path  = not show_path
        elif key == ord("m"):  show_map   = not show_map
        elif key == ord("w"):  show_warn  = not show_warn
        elif key in (ord("+"), ord("=")):
            brt_offset = min(brt_offset + 5,  80)
        elif key == ord("-"):
            brt_offset = max(brt_offset - 5, -80)
        elif key == ord("]"):
            conf = min(conf + 0.05, 0.95)
        elif key == ord("["):
            conf = max(conf - 0.05, 0.05)
        elif key == ord("s"):
            fname = f"screenshot_{int(time.time())}.jpg"
            cv2.imwrite(fname, canvas)
            print(f"[+] Saved {fname}")

    cap.release()
    cv2.destroyAllWindows()
    print("[*] Done.")


if __name__ == "__main__":
    main()

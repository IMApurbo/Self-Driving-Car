"""
openpilot-style YOLOP 320×320 Live Cam UI — Enhanced v4.7
=========================================================
Fix v4.7:
  - _poly_x / _curve_radius: guard against 0-d Kalman arrays (TypeError fix)
  - LaneKalman.update: always returns 1-D array
  - Steering works in DA mode: uses path centroid deviation instead of fits
  - Steering in 2-lane mode: uses polynomial curvature + lateral offset
  - All previous fixes from v4.6 retained
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
MIN_LANE_PIX  = 100

PATH_EMA_FAST  = 0.35
PATH_EMA_SLOW  = 0.15
PATH_HIST      = 3
PATH_TOP_FRAC  = 0.70
PATH_N_ROWS    = 24

STEER_TAU      = 0.10
STEER_DEADBAND = 0.5
STEER_MAX      = 45.0
OFFSET_DEADBAND = 0.02

LANE_CONFIRM_FRAMES = 6
LANE_LOSE_FRAMES    = 10
LANE_COUNT_DEBOUNCE = 8
MIN_LANE_WIDTH_FRAC = 0.15
MAX_LANE_WIDTH_FRAC = 0.70

DET_VOTE_FRAMES = 10
DET_VOTE_THRESH = 0.40

BEV_W = 280
BEV_H = 360
SW_R  = 72
SW_CX = SW_R + 20
SW_CY = DISPLAY_H - SW_R - 20
BEV_OX = DISPLAY_W - BEV_W - 10
BEV_OY = 10
SIDE_X  = 8
SIDE_W  = SW_CX + SW_R
PANEL_W = SIDE_W - SIDE_X - 4

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
C_SIDEBAR = (18,   18,  18)
C_GRID    = (40,   40,  40)

_MEAN     = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD      = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_DILATE_K = np.ones((3, 3), np.uint8)

_CURVE_HIST = 20
_SPEED_HIST = 20

CLASS_META = {
    0: ("person",     (100, 120, 255), "P"),
    1: ("bicycle",    (180, 220,   0), "B"),
    2: ("car",        (  0, 175, 255), "C"),
    3: ("motorcycle", (180,  80, 255), "M"),
    5: ("bus",        (  0, 100, 255), "U"),
    7: ("truck",      (  0,  50, 200), "T"),
}
DANGER_CLASSES = {0, 1}

_DEFAULT_IPM_SRC_FRAC = np.float32([
    [0.12, 0.95], [0.88, 0.95],
    [0.60, 0.58], [0.40, 0.58],
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
            nodes.append(helper.make_node(
                "Constant", inputs=[], outputs=[name], value=t))
        graph = helper.make_graph(
            nodes, "yolop_stub",
            [helper.make_tensor_value_info(
                "images", TensorProto.FLOAT, [1, 3, INF_SIZE, INF_SIZE])],
            [helper.make_tensor_value_info(n, TensorProto.FLOAT, s)
             for n, s in specs])
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
    p = pred[keep]; scores = scores[keep]
    sx, sy = fw / INF_SIZE, fh / INF_SIZE
    cx_ = p[:, 0]*sx; cy_ = p[:, 1]*sy
    bw_ = p[:, 2]*sx; bh_ = p[:, 3]*sy
    x1 = np.clip((cx_ - bw_/2).astype(int), 0, fw-1)
    y1 = np.clip((cy_ - bh_/2).astype(int), 0, fh-1)
    x2 = np.clip((cx_ + bw_/2).astype(int), 0, fw-1)
    y2 = np.clip((cy_ + bh_/2).astype(int), 0, fh-1)
    xywh = np.stack([x1, y1, x2-x1, y2-y1], 1).astype(float).tolist()
    idxs = cv2.dnn.NMSBoxes(xywh, scores.tolist(), conf, iou)
    if len(idxs) == 0:
        return []
    idxs = np.array(idxs).flatten()
    return [(int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i]),
             float(scores[i]), 2) for i in idxs]


# ─────────────────────────────────────────────────────────────────────────────
# Detection Stabiliser
# ─────────────────────────────────────────────────────────────────────────────
class DetectionStabiliser:
    IOU_MATCH = 0.35

    def __init__(self, n=DET_VOTE_FRAMES, thresh=DET_VOTE_THRESH):
        self._buf   = collections.deque(maxlen=n)
        self._thresh = thresh
        self.stable  = []

    @staticmethod
    def _iou(a, b):
        ax1,ay1,ax2,ay2 = a[:4]
        bx1,by1,bx2,by2 = b[:4]
        ix1=max(ax1,bx1); iy1=max(ay1,by1)
        ix2=min(ax2,bx2); iy2=min(ay2,by2)
        iw=max(0,ix2-ix1); ih=max(0,iy2-iy1)
        inter=iw*ih
        ua=(ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter
        return inter/max(ua,1)

    def update(self, dets):
        self._buf.append(dets)
        if len(self._buf) < 2:
            self.stable = dets
            return self.stable
        latest = self._buf[-1]
        stable = []
        past   = list(self._buf)[:-1]
        for det in latest:
            votes = sum(
                1 for pf in past
                if any(self._iou(det, pd) >= self.IOU_MATCH for pd in pf)
            )
            if votes / max(len(past), 1) >= self._thresh:
                stable.append(det)
        self.stable = stable
        return stable


# ─────────────────────────────────────────────────────────────────────────────
# Kalman Lane Tracker  — always returns clean 1-D array
# ─────────────────────────────────────────────────────────────────────────────
class LaneKalman:
    def __init__(self):
        self.kf = cv2.KalmanFilter(6, 3)
        dt = 1.0
        self.kf.transitionMatrix = np.array([
            [1,0,0,dt,0, 0],[0,1,0,0,dt,0],[0,0,1,0,0,dt],
            [0,0,0,1, 0, 0],[0,0,0,0,1, 0],[0,0,0,0,0, 1],
        ], dtype=np.float32)
        self.kf.measurementMatrix   = np.eye(3, 6, dtype=np.float32)
        self.kf.processNoiseCov     = np.eye(6, dtype=np.float32) * 5e-3
        self.kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 1e-2
        self.kf.errorCovPost        = np.eye(6, dtype=np.float32)
        self.initialized  = False
        self.frames_seen  = 0
        self.frames_lost  = 0

    def _safe_state(self):
        """Always return a clean (3,) float64 array — never 0-d."""
        raw = self.kf.statePost[:3]                  # shape (3,1) or (3,)
        return np.asarray(raw, dtype=np.float64).flatten()[:3]

    def update(self, fit):
        fit  = np.asarray(fit, dtype=np.float32).flatten()[:3]
        meas = fit.reshape(3, 1)
        if not self.initialized:
            self.kf.statePost = np.zeros((6,1), dtype=np.float32)
            self.kf.statePost[:3] = meas
            self.initialized = True
        self.kf.predict()
        self.kf.correct(meas)
        self.frames_seen += 1
        self.frames_lost  = 0
        return self._safe_state()

    def predict_only(self):
        self.frames_lost += 1
        self.kf.predict()
        return self._safe_state()

    def reset(self):
        self.initialized = False
        self.frames_seen = 0
        self.frames_lost = 0

    @property
    def confirmed(self):
        return self.initialized and self.frames_seen >= LANE_CONFIRM_FRAMES

    @property
    def valid(self):
        return self.initialized and self.frames_lost < LANE_LOSE_FRAMES


# ─────────────────────────────────────────────────────────────────────────────
# IPM Transform
# ─────────────────────────────────────────────────────────────────────────────
class IPMTransform:
    CAL_FILE = "ipm_calibration.json"

    def __init__(self, w, h):
        self._w, self._h = w, h
        self.src = (_DEFAULT_IPM_SRC_FRAC * np.float32([w,h])).astype(np.float32)
        self._load(); self._build()

    def _build(self):
        dst = np.float32([[0,BEV_H],[BEV_W,BEV_H],[BEV_W,0],[0,0]])
        self.M     = cv2.getPerspectiveTransform(self.src, dst)
        self.M_inv = cv2.getPerspectiveTransform(dst, self.src)

    def warp(self, f):
        return cv2.warpPerspective(f,self.M,(BEV_W,BEV_H),flags=cv2.INTER_LINEAR)

    def warp_mask(self, m):
        return cv2.warpPerspective(m,self.M,(BEV_W,BEV_H),flags=cv2.INTER_NEAREST)

    def cam_to_bev(self, pts):
        if len(pts)==0: return pts
        return cv2.perspectiveTransform(
            pts.reshape(-1,1,2).astype(np.float32),self.M).reshape(-1,2)

    def save(self):
        with open(self.CAL_FILE,"w") as f:
            json.dump({"src":self.src.tolist(),"w":self._w,"h":self._h},
                      f, indent=2)
        print(f"[CAL] Saved {self.CAL_FILE}")

    def _load(self):
        if not os.path.exists(self.CAL_FILE): return
        try:
            with open(self.CAL_FILE) as f: d=json.load(f)
            if d["w"]==self._w and d["h"]==self._h:
                self.src=np.array(d["src"],dtype=np.float32)
                print(f"[CAL] Loaded {self.CAL_FILE}")
        except Exception as e:
            print(f"[CAL] Load failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Calibration UI
# ─────────────────────────────────────────────────────────────────────────────
class CalibrationUI:
    def __init__(self, ipm):
        self._ipm=ipm; self._drag_idx=None; self.active=False

    def toggle(self): self.active=not self.active

    def mouse(self, event, x, y, flags, param):
        if not self.active: return
        pts=self._ipm.src
        if event==cv2.EVENT_LBUTTONDOWN:
            dists=np.linalg.norm(pts-[x,y],axis=1); idx=int(np.argmin(dists))
            if dists[idx]<25: self._drag_idx=idx
        elif event==cv2.EVENT_MOUSEMOVE and self._drag_idx is not None:
            pts[self._drag_idx]=[x,y]; self._ipm._build()
        elif event==cv2.EVENT_LBUTTONUP:
            self._drag_idx=None

    def draw(self, canvas):
        if not self.active: return
        pts=self._ipm.src.astype(np.int32)
        for i in range(4):
            cv2.line(canvas,tuple(pts[i]),tuple(pts[(i+1)%4]),(0,255,80),1,cv2.LINE_AA)
        for i,(x,y) in enumerate(pts):
            cv2.circle(canvas,(x,y),10,(0,255,80),2,cv2.LINE_AA)
            cv2.putText(canvas,["BL","BR","TR","TL"][i],(x+12,y),
                        cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,255,80),1,cv2.LINE_AA)
        cv2.putText(canvas,
            "CALIBRATION — drag corners  |  ENTER=save  |  C=exit",
            (canvas.shape[1]//2-250,80),
            cv2.FONT_HERSHEY_SIMPLEX,0.50,(0,255,80),2,cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# Speed estimator
# ─────────────────────────────────────────────────────────────────────────────
class SpeedEstimator:
    def __init__(self):
        self._prev=None; self._hist=collections.deque(maxlen=_SPEED_HIST)
        self.kmh=0.0

    def update(self, frame):
        h,w=frame.shape[:2]
        roi=frame[h//3:2*h//3,w//4:3*w//4]
        gray=cv2.cvtColor(cv2.resize(roi,(80,45)),cv2.COLOR_BGR2GRAY)
        if self._prev is not None:
            flow=cv2.calcOpticalFlowFarneback(
                self._prev,gray,None,0.5,2,8,2,5,1.1,0)
            self._hist.append(float(np.mean(
                np.sqrt(flow[...,0]**2+flow[...,1]**2)))*18.0)
        self._prev=gray
        if self._hist: self.kmh=float(np.mean(self._hist))
        return self.kmh


# ─────────────────────────────────────────────────────────────────────────────
# Lane Analyser  v4.7
# ─────────────────────────────────────────────────────────────────────────────
class LaneAnalyser:
    def __init__(self, w, h):
        self._w, self._h  = w, h
        self._ema1        = []
        self._ema2        = []
        self._path_hist   = collections.deque(maxlen=PATH_HIST)
        self._left_kf     = LaneKalman()
        self._right_kf    = LaneKalman()
        self._left_fit    = None    # confirmed 1-D (3,) array or None
        self._right_fit   = None
        self._pending_count  = 0
        self._pending_frames = 0
        self.lane_count      = 0
        self.curve_rad       = 9999.0
        self.lat_offset      = 0.0
        self.departure       = False
        # DA-path steering: deviation of path midpoint from frame centre
        self.da_steer_offset = 0.0
        self._curve_hist  = collections.deque(maxlen=_CURVE_HIST)
        self._offset_hist = collections.deque(maxlen=8)
        self._half_lane_w = w * 0.22
        self._final       = []

    # ── Safe poly helpers — never fail on bad array shape ─────────────────
    @staticmethod
    def _poly_x(fit, y):
        """Evaluate polynomial fit at y.  fit must be (3,) float array."""
        if fit is None:
            return 0.0
        fit = np.atleast_1d(np.asarray(fit, dtype=np.float64)).flatten()
        if len(fit) < 1:
            return 0.0
        return float(np.polyval(fit, float(y)))

    @staticmethod
    def _curve_radius(fit, y_eval, ym=30/720, xm=3.7/700):
        if fit is None:
            return 9999.0
        fit = np.atleast_1d(np.asarray(fit, dtype=np.float64)).flatten()
        if len(fit) < 3:
            return 9999.0
        A = fit[0] * (xm / ym**2)
        B = fit[1] * (xm / ym)
        return float(min(
            ((1 + (2*A*y_eval*ym + B)**2)**1.5) / abs(2*A + 1e-6),
            9999.0))

    @staticmethod
    def _fit_poly(ys, xs):
        if len(ys) < MIN_LANE_PIX:
            return None
        try:
            result = np.polyfit(ys, xs, 2)
            # Always return clean 1-D (3,) float64
            return np.asarray(result, dtype=np.float64).flatten()[:3]
        except Exception:
            return None

    # ── CC lane split ──────────────────────────────────────────────────────
    def _split(self, ll_mask):
        cx_img = self._w / 2.0
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(
            ll_mask.astype(np.uint8), connectivity=8)
        left_blobs=[]; right_blobs=[]
        for lbl in range(1,n):
            area=stats[lbl,cv2.CC_STAT_AREA]
            if area<20: continue
            ys,xs=np.where(labels==lbl)
            cx=float(centroids[lbl,0])
            if cx < cx_img: left_blobs.append((area,cx,ys,xs))
            else:           right_blobs.append((area,cx,ys,xs))

        def _merge(blobs, k=2):
            blobs=sorted(blobs,key=lambda b:-b[0])[:k]
            if not blobs: return np.array([]),np.array([]),None
            ys=np.concatenate([b[2] for b in blobs])
            xs=np.concatenate([b[3] for b in blobs])
            cx=float(np.mean([b[1] for b in blobs]))
            return ys,xs,cx

        ly,lx,l_cx=_merge(left_blobs)
        ry,rx,r_cx=_merge(right_blobs)
        return ly,lx,ry,rx,l_cx,r_cx

    # ── 2-lane width sanity check ──────────────────────────────────────────
    def _validate_two_lanes(self, lf, rf):
        if lf is None or rf is None: return False
        lf=np.atleast_1d(np.asarray(lf,dtype=np.float64)).flatten()
        rf=np.atleast_1d(np.asarray(rf,dtype=np.float64)).flatten()
        if len(lf)<3 or len(rf)<3: return False
        y_ref=self._h*0.85
        lx=self._poly_x(lf,y_ref); rx=self._poly_x(rf,y_ref)
        if rx<=lx: return False
        w_frac=(rx-lx)/self._w
        return MIN_LANE_WIDTH_FRAC <= w_frac <= MAX_LANE_WIDTH_FRAC

    # ── DA centroid path ───────────────────────────────────────────────────
    def _da_path(self, da_mask):
        if da_mask is None: return []
        h,w=da_mask.shape
        y_top=int(h*PATH_TOP_FRAC)
        ys=np.linspace(y_top,h-1,PATH_N_ROWS).astype(int)
        pts=[]
        for y in ys:
            xs=np.where(da_mask[y]>0)[0]
            if len(xs)>=4: pts.append((int(np.mean(xs)),int(y)))
        return pts

    # ── 2-lane midline path ────────────────────────────────────────────────
    def _poly_path_from_fits(self, lf, rf):
        h=self._h; y_top=int(h*PATH_TOP_FRAC)
        ys=np.linspace(y_top,h-1,PATH_N_ROWS)
        pts=[]
        for y in ys:
            lx=self._poly_x(lf,y); rx=self._poly_x(rf,y)
            lo,hi=(lx,rx) if lx<rx else (rx,lx)
            if hi-lo<10: continue
            pts.append((int(np.clip((lo+hi)/2,0,self._w-1)),int(y)))
        return pts

    # ── Dual EMA + history blend ───────────────────────────────────────────
    def _dual_ema(self, new_pts):
        if not new_pts: return self._ema2 if self._ema2 else []
        n=len(new_pts)
        if len(self._ema1)!=n: self._ema1=list(new_pts)
        else:
            a=PATH_EMA_FAST
            self._ema1=[(int(o[0]+a*(p[0]-o[0])),p[1])
                        for o,p in zip(self._ema1,new_pts)]
        if len(self._ema2)!=n: self._ema2=list(self._ema1)
        else:
            a=PATH_EMA_SLOW
            self._ema2=[(int(o[0]+a*(p[0]-o[0])),p[1])
                        for o,p in zip(self._ema2,self._ema1)]
        return self._ema2

    def _blend_history(self, pts):
        self._path_hist.append(pts)
        hlen=len(self._path_hist)
        if hlen==0 or not pts: return pts
        weights=np.linspace(0.2,1.0,hlen); weights/=weights.sum()
        n=len(pts)
        return [(int(sum((p[i][0] if i<len(p) else pts[i][0])*w
                        for p,w in zip(self._path_hist,weights))),
                 pts[i][1]) for i in range(n)]

    # ── Main update ────────────────────────────────────────────────────────
    def update(self, ll_mask, da_mask):
        h,w=ll_mask.shape; self._w,self._h=w,h

        ly,lx,ry,rx,l_cx,r_cx=self._split(ll_mask)

        raw_left =self._fit_poly(ly,lx)
        raw_right=self._fit_poly(ry,rx)

        # Kalman
        kf_l = (self._left_kf.update(raw_left)   if raw_left  is not None else
                self._left_kf.predict_only()       if self._left_kf.valid  else None)
        kf_r = (self._right_kf.update(raw_right)  if raw_right is not None else
                self._right_kf.predict_only()      if self._right_kf.valid else None)

        # 2-lane confirmation
        two_valid = (
            self._left_kf.confirmed and
            self._right_kf.confirmed and
            self._validate_two_lanes(kf_l, kf_r)
        )
        raw_count = 2 if two_valid else 0

        # Debounce
        if raw_count != self._pending_count:
            self._pending_count=raw_count; self._pending_frames=0
        else:
            self._pending_frames+=1
        committed = (self._pending_count
                     if self._pending_frames >= LANE_COUNT_DEBOUNCE
                     else self.lane_count)
        self.lane_count = committed

        # Store fits
        if self.lane_count == 2:
            self._left_fit  = kf_l
            self._right_fit = kf_r
            y_ref=h*0.85
            lx_r=self._poly_x(self._left_fit, y_ref)
            rx_r=self._poly_x(self._right_fit,y_ref)
            self._half_lane_w=(rx_r-lx_r)/2.0
        else:
            if not self._left_kf.valid:
                self._left_fit=None; self._left_kf.reset()
            if not self._right_kf.valid:
                self._right_fit=None; self._right_kf.reset()

        # Curvature (only in 2-lane)
        y_eval=h*0.80
        if self.lane_count==2:
            rads=[self._curve_radius(f,y_eval)
                  for f in [self._left_fit,self._right_fit] if f is not None]
            self._curve_hist.append(float(np.mean(rads)) if rads else 9999.0)
        else:
            self._curve_hist.append(9999.0)
        self.curve_rad=float(np.mean(self._curve_hist))

        # Lateral offset (2-lane) / DA steering offset
        cx_img=w/2.0
        if self.lane_count==2:
            lx_r=self._poly_x(self._left_fit, y_eval)
            rx_r=self._poly_x(self._right_fit,y_eval)
            raw_off=((lx_r+rx_r)/2.0-cx_img)/(w/2.0)
            if abs(raw_off)<OFFSET_DEADBAND: raw_off=0.0
            self._offset_hist.append(raw_off)
            self.lat_offset=float(np.mean(self._offset_hist))
            self.da_steer_offset=0.0
        else:
            self.lat_offset=0.0
            # DA steering: measure bottom-third centroid of drivable area
            self.da_steer_offset=self._da_steer(da_mask,cx_img,h,w)

        self.departure=abs(self.lat_offset)>0.40

        # Path
        if self.lane_count==2:
            raw_pts=self._poly_path_from_fits(self._left_fit,self._right_fit)
        else:
            raw_pts=self._da_path(da_mask)

        ema_pts    =self._dual_ema(raw_pts)
        self._final=self._blend_history(ema_pts)

    # ── DA steering helper ─────────────────────────────────────────────────
    @staticmethod
    def _da_steer(da_mask, cx_img, h, w):
        """
        Compute lateral offset from DA mask bottom-third centroid.
        Returns value in [-1, 1] (negative = left, positive = right).
        """
        if da_mask is None:
            return 0.0
        y_start=int(h*0.65)
        roi=da_mask[y_start:,:]
        ys,xs=np.where(roi>0)
        if len(xs)<50:
            return 0.0
        da_cx=float(np.mean(xs))
        raw=(da_cx-cx_img)/(cx_img)
        # Deadband
        if abs(raw)<OFFSET_DEADBAND: raw=0.0
        return float(np.clip(raw,-1.0,1.0))

    # ── Live path for render ───────────────────────────────────────────────
    def render_path(self):
        if (self.lane_count==2 and
                self._validate_two_lanes(self._left_fit,self._right_fit)):
            return self._poly_path_from_fits(self._left_fit,self._right_fit)
        return self._final   # DA path

    @property
    def path_points(self):
        return self._final

    def lane_edge_points(self, n=20):
        if self.lane_count!=2: return [],[]
        h,y_top=self._h,int(self._h*PATH_TOP_FRAC)
        ys=np.linspace(y_top,h-1,n)
        lpts=[(int(self._poly_x(self._left_fit, y)),int(y)) for y in ys]
        rpts=[(int(self._poly_x(self._right_fit,y)),int(y)) for y in ys]
        return lpts,rpts


# ─────────────────────────────────────────────────────────────────────────────
# Steering model  v4.7 — works in both 2-lane and DA mode
# ─────────────────────────────────────────────────────────────────────────────
class SteeringModel:
    """
    2-lane mode : curvature from polynomial A-coefficient + lateral offset
    DA mode     : lateral offset from DA centroid deviation only
                  (da_steer_offset from LaneAnalyser)
    """
    def __init__(self):
        self._smooth=0.0; self.angle=0.0; self.torque=0.0

    def update(self, lane_an: "LaneAnalyser") -> float:
        if lane_an.lane_count == 2:
            # ── Poly-based steering ───────────────────────────────────────
            curve_steer=0.0
            if lane_an.curve_rad < 9000:
                fits=[f for f in [lane_an._left_fit,lane_an._right_fit]
                      if f is not None]
                if fits:
                    fits_arr=[np.atleast_1d(np.asarray(f,dtype=np.float64)
                                            ).flatten() for f in fits]
                    avg_a=float(np.mean([f[0] for f in fits_arr if len(f)>=1]))
                    mag=math.degrees(
                        math.atan(3.0/max(lane_an.curve_rad,1.0)))*8.0
                    curve_steer=-np.sign(avg_a)*mag
            raw=max(-STEER_MAX,min(STEER_MAX,
                    curve_steer + lane_an.lat_offset*18.0))
        else:
            # ── DA-based steering: follow path centroid ───────────────────
            # da_steer_offset in [-1,1]; scale to degrees
            raw=max(-STEER_MAX,min(STEER_MAX,
                    lane_an.da_steer_offset * STEER_MAX * 0.6))

        # Low-pass filter
        self._smooth += (1.0-STEER_TAU)*(raw-self._smooth)
        out=self._smooth
        if abs(out)<STEER_DEADBAND: out=0.0
        self.angle =float(np.clip(out,-STEER_MAX,STEER_MAX))
        self.torque=self.angle/STEER_MAX
        return self.angle


# ─────────────────────────────────────────────────────────────────────────────
# Lead tracker
# ─────────────────────────────────────────────────────────────────────────────
class LeadTracker:
    def __init__(self):
        self.box=None; self.dist_m=0.0; self.ttc_s=99.0

    def update(self, dets, fw, fh, speed_kmh):
        if not dets:
            self.box=None; self.dist_m=0.0; self.ttc_s=99.0; return
        cx=fw/2; best,best_s=None,-1.0
        for d in dets:
            x1,y1,x2,y2,sc,_=d
            rank=(x2-x1)*(y2-y1)*(1-abs(((x1+x2)/2-cx)/cx))*(y2/fh)*sc
            if rank>best_s: best,best_s=d,rank
        self.box=best
        if best:
            bh=max(best[3]-best[1],1)
            self.dist_m=max(1.5*700.0/bh,1.0)
            self.ttc_s =self.dist_m/max(speed_kmh*0.8/3.6,0.1)
        else:
            self.dist_m=0.0; self.ttc_s=99.0


# ─────────────────────────────────────────────────────────────────────────────
# Night mode
# ─────────────────────────────────────────────────────────────────────────────
class AdaptiveDisplay:
    def __init__(self):
        self.auto=True; self.night=False
        self._clahe=cv2.createCLAHE(clipLimit=2.5,tileGridSize=(8,8))

    def process(self, frame):
        lum=float(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY).mean())
        if self.auto: self.night=lum<60
        if not self.night: return frame.copy()
        lab=cv2.cvtColor(frame,cv2.COLOR_BGR2LAB); l,a,b=cv2.split(lab)
        return cv2.cvtColor(cv2.merge([self._clahe.apply(l),a,b]),
                            cv2.COLOR_LAB2BGR)


# ─────────────────────────────────────────────────────────────────────────────
# Dashcam recorder
# ─────────────────────────────────────────────────────────────────────────────
class DashcamRecorder:
    SPLIT_SEC=600; MAX_FILES=6; FPS_OUT=25

    def __init__(self, out_dir="dashcam"):
        self._dir=out_dir; self._writer=None; self._start=None
        self._files=collections.deque(); self.active=False
        os.makedirs(out_dir,exist_ok=True)

    def toggle(self):
        self.active=not self.active
        if not self.active and self._writer:
            self._writer.release(); self._writer=None
        print(f"[REC] {'ON' if self.active else 'OFF'}")

    def write(self, frame):
        if not self.active: return
        now=time.time()
        if self._writer is None or now-self._start>self.SPLIT_SEC:
            self._rotate(frame.shape[1],frame.shape[0])
        self._writer.write(frame)

    def _rotate(self, w, h):
        if self._writer: self._writer.release()
        ts=time.strftime("%Y%m%d_%H%M%S")
        path=os.path.join(self._dir,f"dash_{ts}.mp4")
        self._writer=cv2.VideoWriter(
            path,cv2.VideoWriter_fourcc(*"mp4v"),self.FPS_OUT,(w,h))
        self._start=time.time(); self._files.append(path)
        while len(self._files)>self.MAX_FILES:
            old=self._files.popleft()
            try: os.remove(old)
            except: pass
        print(f"[REC] {path}")

    def stop(self):
        if self._writer: self._writer.release()


# ─────────────────────────────────────────────────────────────────────────────
# Event logger
# ─────────────────────────────────────────────────────────────────────────────
class EventLogger:
    COOLDOWN=5.0
    def __init__(self, path="events.csv"):
        self._last={}; self._f=open(path,"a",newline="")
        self._csv=csv.writer(self._f)
    def log(self,kind,detail=""):
        now=time.time()
        if now-self._last.get(kind,0)<self.COOLDOWN: return
        self._last[kind]=now
        self._csv.writerow([datetime.datetime.now().isoformat(
            timespec="seconds"),kind,detail])
        self._f.flush()
    def close(self): self._f.close()


# ─────────────────────────────────────────────────────────────────────────────
# Left sidebar
# ─────────────────────────────────────────────────────────────────────────────
def draw_left_sidebar(canvas, kmh, lane_an, lead, engaged):
    pad=6; lh=22
    col_l=(110,110,110); col_v=(220,220,220)
    dep_col=C_WARN if lane_an.departure else (80,200,80)
    ttc_col=((0,0,255) if lead.ttc_s<3.0 else
             C_WARN    if lead.ttc_s<6.0 else (80,200,80))
    eng_col=C_ENGAGE if engaged else C_STBY
    lc_str={0:"DA MODE",2:"2-LANE"}.get(lane_an.lane_count,"--")

    sections=[
        ("SPEED", [("",f"{max(0.0,min(kmh,200.0)):.0f} km/h",(230,230,230))]),
        ("LANE",  [("Mode",   lc_str,                                  col_v),
                   ("Curve R",f"{min(lane_an.curve_rad,9999):.0f} m",  col_v),
                   ("Offset", f"{lane_an.lat_offset:+.2f}",            col_v),
                   ("Depart", "! YES" if lane_an.departure else "OK",  dep_col)]),
        ("LEAD",  [("Dist",f"{lead.dist_m:.1f} m" if lead.box else "--",col_v),
                   ("TTC", f"{lead.ttc_s:.1f} s"  if lead.box else "--",ttc_col)]),
        ("SYSTEM",[("Mode","ENGAGED" if engaged else "STANDBY",         eng_col)]),
    ]

    total_rows=sum(1+len(r) for _,r in sections)
    total_h=total_rows*lh+len(sections)*8+pad*2
    wheel_top=SW_CY-SW_R-28
    sb_y=max(4,wheel_top-total_h); sb_x=SIDE_X
    sb_w=PANEL_W+pad*2; sb_h=total_h

    overlay=canvas.copy()
    cv2.rectangle(overlay,(sb_x,sb_y),(sb_x+sb_w,sb_y+sb_h),C_SIDEBAR,-1)
    cv2.addWeighted(overlay,0.82,canvas,0.18,0,canvas)
    cv2.rectangle(canvas,(sb_x,sb_y),(sb_x+3,sb_y+sb_h),(0,180,80),-1)
    cv2.rectangle(canvas,(sb_x,sb_y),(sb_x+sb_w,sb_y+sb_h),(50,50,50),1,cv2.LINE_AA)

    cursor_y=sb_y+pad; tx=sb_x+pad+5
    for sec_title,rows in sections:
        cv2.rectangle(canvas,(sb_x+4,cursor_y),(sb_x+sb_w-4,cursor_y+lh-2),
                      (30,30,30),-1)
        cv2.putText(canvas,sec_title,(tx,cursor_y+lh-6),
                    cv2.FONT_HERSHEY_SIMPLEX,0.34,(140,140,140),1,cv2.LINE_AA)
        cursor_y+=lh+2
        for label,val,vc in rows:
            if label:
                cv2.putText(canvas,label+":",(tx,cursor_y+lh-6),
                            cv2.FONT_HERSHEY_SIMPLEX,0.30,col_l,1,cv2.LINE_AA)
                (tw,_),_=cv2.getTextSize(val,cv2.FONT_HERSHEY_SIMPLEX,0.38,1)
                cv2.putText(canvas,val,(sb_x+sb_w-tw-pad-4,cursor_y+lh-6),
                            cv2.FONT_HERSHEY_SIMPLEX,0.38,vc,1,cv2.LINE_AA)
            else:
                (tw,_),_=cv2.getTextSize(val,cv2.FONT_HERSHEY_SIMPLEX,0.72,2)
                cv2.putText(canvas,val,(sb_x+(sb_w-tw)//2,cursor_y+lh-2),
                            cv2.FONT_HERSHEY_SIMPLEX,0.72,vc,2,cv2.LINE_AA)
            cursor_y+=lh
        cursor_y+=8

    bar_x=sb_x+4; bar_y=sb_y+sb_h-7; bar_W=sb_w-8
    frac=max(0.0,min(kmh,200.0))/200.0
    bc=((0,200,80) if kmh<80 else (0,160,255) if kmh<120 else (0,60,255))
    cv2.rectangle(canvas,(bar_x,bar_y),(bar_x+bar_W,bar_y+4),(40,40,40),-1)
    if frac>0:
        cv2.rectangle(canvas,(bar_x,bar_y),
                      (bar_x+int(bar_W*frac),bar_y+4),bc,-1)


# ─────────────────────────────────────────────────────────────────────────────
# BEV Panel
# ─────────────────────────────────────────────────────────────────────────────
class BEVPanel:
    M_PER_PIX=40.0/BEV_H
    def __init__(self,ipm,fw,fh): self._ipm=ipm

    @staticmethod
    def _draw_grid(canvas):
        ppm=max(1,int(10.0/BEVPanel.M_PER_PIX)); y=BEV_H-ppm; dm=10
        while y>0:
            cv2.line(canvas,(0,y),(BEV_W,y),C_GRID,1,cv2.LINE_AA)
            cv2.putText(canvas,f"{dm}m",(4,y-2),
                        cv2.FONT_HERSHEY_SIMPLEX,0.28,(70,70,70),1,cv2.LINE_AA)
            y-=ppm; dm+=10
        cx=BEV_W//2
        cv2.line(canvas,(cx,0),(cx,BEV_H),(40,40,40),1,cv2.LINE_AA)

    @staticmethod
    def _draw_ego(canvas):
        cx,cy,cw2,cl=BEV_W//2,BEV_H-18,12,22
        body=np.array([[cx-cw2,cy],[cx+cw2,cy],
                       [cx+cw2,cy-cl],[cx-cw2,cy-cl]],np.int32)
        cv2.fillPoly(canvas,[body],(60,130,60))
        cv2.polylines(canvas,[body],True,(0,220,80),2,cv2.LINE_AA)
        cv2.putText(canvas,"EGO",(cx-10,cy+12),
                    cv2.FONT_HERSHEY_SIMPLEX,0.28,(0,200,80),1,cv2.LINE_AA)

    def render(self,da_mask,ll_mask,lane_an,dets,lead_box,canvas_out,ox,oy):
        bev=np.zeros((BEV_H,BEV_W,3),dtype=np.uint8)
        self._draw_grid(bev)
        if da_mask is not None:
            try:
                db=self._ipm.warp_mask(da_mask)
                dc=np.zeros_like(bev); dc[db==1]=(0,80,0)
                cv2.addWeighted(bev,1.0,dc,0.85,0,bev)
                ctrs,_=cv2.findContours(db,cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(bev,ctrs,-1,(0,160,50),1)
            except: pass
        if ll_mask is not None:
            try:
                lb=self._ipm.warp_mask(ll_mask); bev[lb==1]=(200,200,200)
            except: pass
        pts=lane_an.render_path()
        if len(pts)>=2:
            try:
                bp=self._ipm.cam_to_bev(np.array(pts,np.float32))
                v=((bp[:,0]>=0)&(bp[:,0]<BEV_W)&
                   (bp[:,1]>=0)&(bp[:,1]<BEV_H))
                bp=bp[v].astype(np.int32)
                if len(bp)>=2:
                    cv2.polylines(bev,[bp],False,C_PATH,2,cv2.LINE_AA)
            except: pass
        lpts,rpts=lane_an.lane_edge_points(n=16)
        for side in [lpts,rpts]:
            if len(side)>=2:
                try:
                    arr=self._ipm.cam_to_bev(
                        np.array(side,np.float32)).astype(np.int32)
                    v=((arr[:,0]>=0)&(arr[:,0]<BEV_W)&
                       (arr[:,1]>=0)&(arr[:,1]<BEV_H))
                    arr=arr[v]
                    if len(arr)>=2:
                        cv2.polylines(bev,[arr],False,(0,200,80),2,cv2.LINE_AA)
                except: pass
        self._draw_ego(bev)
        cv2.rectangle(bev,(0,0),(BEV_W-1,BEV_H-1),(60,60,60),1)
        cv2.putText(bev,"BIRD'S-EYE VIEW",(BEV_W//2-56,14),
                    cv2.FONT_HERSHEY_SIMPLEX,0.38,(100,100,100),1,cv2.LINE_AA)
        ch,cw=canvas_out.shape[:2]
        ey,ex=min(oy+BEV_H,ch),min(ox+BEV_W,cw)
        ph,pw=ey-oy,ex-ox
        if ph>0 and pw>0: canvas_out[oy:ey,ox:ex]=bev[:ph,:pw]


# ─────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────────────────────
def apply_brightness(frame,offset):
    if offset==0.0: return frame.copy()
    return np.clip(frame.astype(np.int16)+int(offset),0,255).astype(np.uint8)

def overlay_da(canvas,mask,alpha=0.28):
    ys,xs=np.where(mask==1)
    if len(ys)==0: return
    roi=canvas[ys,xs].astype(np.float32)
    canvas[ys,xs]=np.clip(
        roi*(1-alpha)+np.array(C_GREEN,np.float32)*alpha,0,255).astype(np.uint8)

def overlay_ll(canvas,mask):
    d=cv2.dilate(mask,_DILATE_K,iterations=2)
    ys,xs=np.where(d==1)
    if len(ys): canvas[ys,xs]=C_WHITE

def draw_boxes(canvas,dets,lead_box=None):
    for x1,y1,x2,y2,sc,cls_id in dets:
        name,col,icon=CLASS_META.get(cls_id,(f"cls{cls_id}",C_BOX,"?"))
        is_lead=(lead_box is not None and lead_box[0]==x1 and lead_box[1]==y1)
        is_danger=cls_id in DANGER_CLASSES
        border=C_LEAD if is_lead else col
        thick=3 if (is_lead or is_danger) else 2
        cv2.rectangle(canvas,(x1,y1),(x2,y2),border,thick,cv2.LINE_AA)
        label=(f"{'⚠ ' if is_danger else ''}"
               f"{'LEAD ' if is_lead else ''}{name} {sc:.0%}")
        (tw,th),_=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.44,1)
        cv2.rectangle(canvas,(x1,y1-th-6),(x1+tw+6,y1),border,-1)
        cv2.putText(canvas,label,(x1+3,y1-3),
                    cv2.FONT_HERSHEY_SIMPLEX,0.44,(255,255,255),1,cv2.LINE_AA)

def draw_path_plan(canvas,lane_an,show):
    if not show: return
    h,w=canvas.shape[:2]; lc=lane_an.lane_count
    if lc==2:
        lpts,rpts=lane_an.lane_edge_points(n=20)
        for side in [lpts,rpts]:
            if len(side)>=2:
                cv2.polylines(canvas,[np.array(side,np.int32)],
                              False,(0,210,100),2,cv2.LINE_AA)
        mode_txt="2-LANE"; mode_col=(0,230,100)
    else:
        mode_txt="DA PATH"; mode_col=(200,130,0)
    cv2.putText(canvas,mode_txt,(w//2-35,h-62),
                cv2.FONT_HERSHEY_SIMPLEX,0.40,mode_col,1,cv2.LINE_AA)
    pts=lane_an.render_path()
    if len(pts)<2: return
    col=C_PATH if lc==2 else (200,140,0)
    for i in range(len(pts)-1):
        cv2.line(canvas,pts[i],pts[i+1],col,3,cv2.LINE_AA)
    n=len(pts)
    for i,pt in enumerate(pts):
        r=max(2,int(6*i/max(n-1,1)))
        cv2.circle(canvas,pt,r,col,-1,cv2.LINE_AA)

def draw_horizon(canvas,da_mask):
    if da_mask is None: return
    h,w=canvas.shape[:2]
    rows=np.where(da_mask.sum(axis=1)>w*0.05)[0]
    if len(rows)==0: return
    hy=int(rows.min())
    for x in range(0,w,20):
        cv2.line(canvas,(x,hy),(x+10,hy),C_HORIZON,1,cv2.LINE_AA)

def curvature_banner(canvas,curve_rad,lat_offset):
    h,w=canvas.shape[:2]
    if curve_rad>2000:   txt,col="STRAIGHT  \u2191",(0,220,80)
    elif curve_rad>800:
        txt=f"GENTLE {'-->' if lat_offset>0 else '<--'}  {curve_rad:.0f}m"
        col=(0,180,255)
    elif curve_rad>300:
        txt=f"MODERATE {'-->' if lat_offset>0 else '<--'}  {curve_rad:.0f}m"
        col=(0,120,255)
    else:
        txt=f"SHARP {'-->' if lat_offset>0 else '<--'}  {curve_rad:.0f}m"
        col=(0,60,255)
    (tw,_),_=cv2.getTextSize(txt,cv2.FONT_HERSHEY_SIMPLEX,0.52,1)
    cv2.putText(canvas,txt,(w//2-tw//2,72),
                cv2.FONT_HERSHEY_SIMPLEX,0.52,col,2,cv2.LINE_AA)

def draw_steering_wheel(canvas,angle_deg,cx,cy,R=72):
    angle_deg=float(np.clip(angle_deg,-STEER_MAX,STEER_MAX))
    ang_rad=math.radians(angle_deg)
    frac=abs(angle_deg)/STEER_MAX
    rim_col=(0,int(255*(1-frac)),int(255*frac))
    cv2.circle(canvas,(cx,cy),R+12,(20,20,20),-1,cv2.LINE_AA)
    cv2.circle(canvas,(cx,cy),R+12,(55,55,55), 1,cv2.LINE_AA)
    for deg in range(-135,136,15):
        t=math.radians(deg-90); is_m=deg%45==0; ri=R-(6 if is_m else 3)
        cv2.line(canvas,
                 (int(cx+ri*math.cos(t)),   int(cy+ri*math.sin(t))),
                 (int(cx+(R+4)*math.cos(t)),int(cy+(R+4)*math.sin(t))),
                 (160,160,160) if is_m else (70,70,70),1,cv2.LINE_AA)
    ir=math.radians(angle_deg-90)
    cv2.circle(canvas,(int(cx+(R+8)*math.cos(ir)),
                       int(cy+(R+8)*math.sin(ir))),5,rim_col,-1,cv2.LINE_AA)
    cv2.circle(canvas,(cx,cy),R,rim_col,7,cv2.LINE_AA)
    cv2.circle(canvas,(cx,cy),R,(40,40,40),2,cv2.LINE_AA)
    hub_r=int(R*0.28)
    for bd in [90,210,330]:
        sr=math.radians(bd)+ang_rad
        s1=(int(cx+hub_r*math.cos(sr)),int(cy+hub_r*math.sin(sr)))
        s2=(int(cx+(R-8)*math.cos(sr)),int(cy+(R-8)*math.sin(sr)))
        cv2.line(canvas,(s1[0]+1,s1[1]+1),(s2[0]+1,s2[1]+1),(10,10,10),5,cv2.LINE_AA)
        cv2.line(canvas,s1,s2,(90,90,90),4,cv2.LINE_AA)
        cv2.line(canvas,s1,s2,(160,160,160),1,cv2.LINE_AA)
        cv2.circle(canvas,s2,5,(55,55,55),-1,cv2.LINE_AA)
        cv2.circle(canvas,s2,5,(80,80,80), 1,cv2.LINE_AA)
    cv2.circle(canvas,(cx,cy),hub_r,  (35,35,35),-1,cv2.LINE_AA)
    cv2.circle(canvas,(cx,cy),hub_r,  rim_col,    2,cv2.LINE_AA)
    cv2.circle(canvas,(cx,cy),hub_r-4,(50,50,50),-1,cv2.LINE_AA)
    cv2.putText(canvas,"OP",(cx-9,cy+5),
                cv2.FONT_HERSHEY_SIMPLEX,0.38,rim_col,1,cv2.LINE_AA)
    ang_txt=f"{angle_deg:+.1f}\u00b0"
    (tw,_),_=cv2.getTextSize(ang_txt,cv2.FONT_HERSHEY_SIMPLEX,0.46,1)
    cv2.putText(canvas,ang_txt,(cx-tw//2,cy+R+22),
                cv2.FONT_HERSHEY_SIMPLEX,0.46,(200,200,200),1,cv2.LINE_AA)
    cv2.putText(canvas,"STEER",(cx-18,cy-R-14),
                cv2.FONT_HERSHEY_SIMPLEX,0.34,(110,110,110),1,cv2.LINE_AA)

def draw_warnings(canvas,lane_an,lead,engaged):
    h,w=canvas.shape[:2]; warns=[]
    if lane_an.departure:  warns.append(("LANE DEPARTURE",C_WARN))
    if lead.ttc_s<3.0:     warns.append(("COLLISION RISK",(0,0,255)))
    elif lead.ttc_s<6.0:   warns.append(("FOLLOW CLOSE",C_WARN))
    if not engaged:        warns.append(("SYSTEM STANDBY",C_STBY))
    for i,(txt,col) in enumerate(warns):
        y=h//2-40+i*34; ov=canvas.copy()
        cv2.rectangle(ov,(w//2-140,y-18),(w//2+140,y+8),col,-1)
        cv2.addWeighted(ov,0.45,canvas,0.55,0,canvas)
        cv2.putText(canvas,txt,(w//2-120,y),
                    cv2.FONT_HERSHEY_SIMPLEX,0.60,(255,255,255),2,cv2.LINE_AA)

def draw_hud(canvas,fps,engaged,stub,brt,conf,
             show_da,show_ll,show_box,show_path,show_bev,recording,night):
    h,w=canvas.shape[:2]
    col=C_ENGAGE if engaged else C_STBY
    cv2.putText(canvas,"ENGAGED" if engaged else "STANDBY",
                (w//2-60,34),cv2.FONT_HERSHEY_SIMPLEX,0.80,col,2,cv2.LINE_AA)
    fc=((0,210,70) if fps>=15 else (0,160,255) if fps>=8 else (0,80,255))
    cv2.putText(canvas,f"{fps:.0f} FPS",(w-115,28),
                cv2.FONT_HERSHEY_SIMPLEX,0.58,fc,1,cv2.LINE_AA)
    cv2.putText(canvas,"YOLOP v4.7",(w-120,48),
                cv2.FONT_HERSHEY_SIMPLEX,0.36,(100,100,100),1,cv2.LINE_AA)
    if stub:
        cv2.putText(canvas,"STUB — place yolop-320-320.onnx here",
                    (w//2-160,58),
                    cv2.FONT_HERSHEY_SIMPLEX,0.40,(0,100,255),1,cv2.LINE_AA)
    if recording:
        cv2.circle(canvas,(w-14,14),6,(0,0,220),-1,cv2.LINE_AA)
        cv2.putText(canvas,"REC",(w-38,18),
                    cv2.FONT_HERSHEY_SIMPLEX,0.36,(0,0,220),1,cv2.LINE_AA)
    if night:
        cv2.putText(canvas,"NIGHT",(w-72,64),
                    cv2.FONT_HERSHEY_SIMPLEX,0.36,(180,180,50),1,cv2.LINE_AA)
    tags=" ".join(t for t,on in [("DA",show_da),("LL",show_ll),
        ("DET",show_box),("PATH",show_path),("BEV",show_bev)] if on)
    cv2.putText(canvas,tags or "--",(w-240,h-10),
                cv2.FONT_HERSHEY_SIMPLEX,0.36,(110,110,110),1,cv2.LINE_AA)
    cv2.putText(canvas,
                f"BRT {'+' if brt>=0 else ''}{int(brt)}  CONF {conf:.0%}",
                (8,h-10),cv2.FONT_HERSHEY_SIMPLEX,0.34,(110,110,110),1,cv2.LINE_AA)
    cv2.putText(canvas,
        "[E]engage [D]da [L]ll [B]box [P]path [M]bev "
        "[N]night [R]rec [C]cal [+/-]brt [[]conf [S]save [Q]quit",
        (8,h-26),cv2.FONT_HERSHEY_SIMPLEX,0.28,(70,70,70),1,cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# Window helpers
# ─────────────────────────────────────────────────────────────────────────────
def _init_window(name,w,h):
    try:
        cv2.namedWindow(name,cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name,w,h)
        cv2.imshow(name,np.zeros((h,w,3),dtype=np.uint8))
        cv2.waitKey(1); return True
    except Exception as e:
        print(f"[!] Window init failed: {e}"); return False

def _safe_set_mouse_callback(win,fn):
    try: cv2.setMouseCallback(win,fn); return True
    except cv2.error as e:
        print(f"[!] Mouse callback failed: {e}"); return False


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",default="0")
    ap.add_argument("--model", default=MODEL_FILE)
    ap.add_argument("--skip",  type=int,   default=INF_EVERY)
    ap.add_argument("--conf",  type=float, default=0.40)
    args=ap.parse_args()

    sess,out_names,in_name,is_stub=load_session(args.model)
    src    =int(args.source) if args.source.isdigit() else args.source
    is_file=isinstance(src,str) and os.path.isfile(src)
    cap    =cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[!] Cannot open: {args.source}"); sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  DISPLAY_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_H)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if platform.system()=="Linux" and not is_file:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE,0.25)

    WIN="openpilot · YOLOP v4.7"
    if not _init_window(WIN,DISPLAY_W,DISPLAY_H): sys.exit(1)

    speed_est    = SpeedEstimator()
    ipm          = IPMTransform(DISPLAY_W,DISPLAY_H)
    lane_an      = LaneAnalyser(DISPLAY_W,DISPLAY_H)
    steer_model  = SteeringModel()
    lead_tracker = LeadTracker()
    det_stab     = DetectionStabiliser()
    bev_panel    = BEVPanel(ipm,DISPLAY_W,DISPLAY_H)
    adapt_disp   = AdaptiveDisplay()
    recorder     = DashcamRecorder()
    evt_log      = EventLogger()
    cal_ui       = CalibrationUI(ipm)

    _safe_set_mouse_callback(WIN,cal_ui.mouse)

    engaged=True; show_da=True; show_ll=True; show_box=True
    show_path=True; show_bev=True; show_warn=True
    brt_offset=0.0; conf=args.conf

    fps_t=time.time(); fps=0.0; frame_n=0
    inf_n=0; da_mask=ll_mask=None; dets=[]; fails=0

    print(f"[*] Source: {src}  {DISPLAY_W}x{DISPLAY_H}")
    print("[*] v4.7: poly_x 0-d guard + DA steering active")
    print("[*] Q/ESC to quit")

    while True:
        ret,frame=cap.read()
        if not ret:
            if is_file: cap.set(cv2.CAP_PROP_POS_FRAMES,0); continue
            fails+=1
            if fails>30: print("[!] Camera failed."); break
            time.sleep(0.05); continue
        fails=0

        frame=cv2.resize(frame,(DISPLAY_W,DISPLAY_H),
                         interpolation=cv2.INTER_LINEAR)
        h,w=frame.shape[:2]
        kmh=speed_est.update(frame)

        if inf_n % args.skip == 0:
            blob=preprocess(frame)
            outs=sess.run(out_names,{in_name:blob})
            om=dict(zip(out_names,outs))
            try:
                det_raw=om["det_out"]
                da_raw =om["drive_area_seg"]
                ll_raw =om["lane_line_seg"]
            except KeyError:
                det_raw,da_raw,ll_raw=outs[0],outs[1],outs[2]

            da_mask=seg_mask(da_raw,h,w)
            ll_mask=seg_mask(ll_raw,h,w)
            raw_dets=decode_det(det_raw,h,w,conf=conf)
            dets    =det_stab.update(raw_dets)
            lane_an.update(ll_mask,da_mask)
        inf_n+=1

        # ── Steering: pass full lane_an object ────────────────────────────
        steer_angle=steer_model.update(lane_an)
        lead_tracker.update(dets,w,h,kmh)

        if lane_an.departure:
            evt_log.log("departure",f"off={lane_an.lat_offset:+.2f}")
        if lead_tracker.ttc_s<2.0:
            evt_log.log("near_miss",
                        f"d={lead_tracker.dist_m:.1f}m"
                        f" ttc={lead_tracker.ttc_s:.1f}s")

        canvas=adapt_disp.process(frame)
        canvas=apply_brightness(canvas,brt_offset)

        draw_horizon(canvas,da_mask)
        if show_da and da_mask is not None: overlay_da(canvas,da_mask)
        if show_ll and ll_mask is not None: overlay_ll(canvas,ll_mask)

        draw_path_plan(canvas,lane_an,show_path)
        if show_box: draw_boxes(canvas,dets,lead_tracker.box)

        curvature_banner(canvas,lane_an.curve_rad,lane_an.lat_offset)
        if show_warn: draw_warnings(canvas,lane_an,lead_tracker,engaged)

        draw_left_sidebar(canvas,kmh,lane_an,lead_tracker,engaged)
        draw_steering_wheel(canvas,steer_angle,SW_CX,SW_CY,R=SW_R)

        if show_bev:
            bev_panel.render(da_mask,ll_mask,lane_an,
                             dets,lead_tracker.box,
                             canvas,BEV_OX,BEV_OY)

        cal_ui.draw(canvas)
        draw_hud(canvas,fps,engaged,is_stub,brt_offset,conf,
                 show_da,show_ll,show_box,show_path,show_bev,
                 recorder.active,adapt_disp.night)
        recorder.write(canvas)

        frame_n+=1; now=time.time()
        if now-fps_t>=0.5:
            fps=frame_n/(now-fps_t); frame_n=0; fps_t=now

        cv2.imshow(WIN,canvas)
        key=cv2.waitKey(1)&0xFF

        if   key in(ord("q"),27): break
        elif key==ord("e"): engaged   =not engaged
        elif key==ord("d"): show_da   =not show_da
        elif key==ord("l"): show_ll   =not show_ll
        elif key==ord("b"): show_box  =not show_box
        elif key==ord("p"): show_path =not show_path
        elif key==ord("m"): show_bev  =not show_bev
        elif key==ord("w"): show_warn =not show_warn
        elif key==ord("n"):
            adapt_disp.auto=False; adapt_disp.night=not adapt_disp.night
        elif key==ord("r"): recorder.toggle()
        elif key==ord("c"): cal_ui.toggle()
        elif key==13:       ipm.save()
        elif key in(ord("+"),ord("=")): brt_offset=min(brt_offset+5, 80)
        elif key==ord("-"):             brt_offset=max(brt_offset-5,-80)
        elif key==ord("]"): conf=min(conf+0.05,0.95)
        elif key==ord("["): conf=max(conf-0.05,0.05)
        elif key==ord("s"):
            fname=f"screenshot_{int(time.time())}.jpg"
            cv2.imwrite(fname,canvas); print(f"[+] {fname}")

    recorder.stop(); evt_log.close()
    cap.release(); cv2.destroyAllWindows()
    print("[*] Done.")


if __name__=="__main__":
    main()

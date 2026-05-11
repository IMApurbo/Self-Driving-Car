# 🚗 Autopilot — YOLOP Live Camera UI

An openpilot-style real-time driving assistant powered by **YOLOP** (320×320 ONNX) — runs on a webcam or video file and renders lane detection, drivable area segmentation, object detection, path planning, and a full HUD entirely in OpenCV.

> **Author:** [IMApurbo](https://github.com/IMApurbo)  
> **License:** MIT  
> **Version:** v4.9 (Lane Priority Enhancement)

---

## ✨ Features

### 🛣️ Lane Detection & Path Planning
- Detects **all visible lane blobs** independently on both sides of centre
- **Nearest-lane priority (v4.9):** always selects the single closest lane line per side — ignores outer lanes on 3-lane and 4-lane roads
- **Kalman-filtered** left and right lane polynomials for smooth, jitter-free tracking
- **EMA-stabilised path** with fast/slow blend and history smoothing
- DA-anchored fallback path when lane lines are lost — centre-anchored, drift-limited per row
- Lateral offset and curvature radius computed in real-world units (m)

### 🎛️ HUD & Overlays (all toggleable)
| Overlay | Key | Description |
|---|---|---|
| Drivable Area | `D` | Green segmentation mask over road surface |
| Lane Lines | `L` | Detected lane line overlay |
| Object Boxes | `B` | Bounding boxes with class labels and confidence |
| Path Plan | `P` | Projected driving path drawn on frame |
| Bird's-Eye View | `M` | BEV panel with objects and lanes in top-down view |
| Warnings | `W` | Lane departure and near-miss alerts |
| Night Mode | `N` | Adaptive display / manual night toggle |

### 🧭 Steering & Control
- **Steering model** computes angle from lane geometry with EMA smoothing, deadband, and ±45° clamp
- On-screen **steering wheel** renders current angle in real time
- Engage/disengage autopilot mode with `E`

### 📦 Object Detection
- Detects: **person, bicycle, car, motorcycle, bus, truck**
- IOU-based **detection stabiliser** (voting over N frames) eliminates flickering boxes
- **Lead vehicle tracker** with distance estimate and TTC (time-to-collision) in seconds
- Danger-class highlighting for pedestrians and cyclists

### 📐 Bird's-Eye View (BEV) Panel
- IPM (Inverse Perspective Mapping) transform — **drag-and-drop calibration UI** (press `C`)
- Objects and detections fully rendered in top-down view
- Lane lines and drivable area warped into BEV space
- Calibration saved/loaded from `ipm_calibration.json` (press `Enter` to save)

### 📊 Additional Systems
- **Optical flow speed estimator** — estimates km/h from frame motion (no GPS needed)
- **Dashcam recorder** — toggle recording with `R`; saves timestamped `.avi`
- **Event logger** — logs lane departures and near-miss events to CSV
- **Curvature banner** — shows curve radius and lateral offset on screen
- **Brightness control** (`+`/`-`) and **confidence threshold** (`[`/`]`) adjustable live
- Screenshot with `S` → saves `screenshot_<timestamp>.jpg`

---

## 🛠️ Requirements

- Python 3.8+
- A YOLOP ONNX model file: `yolop-320-320.onnx`
- Webcam or video file

### Install dependencies

```bash
pip install opencv-python numpy onnxruntime
```

> **Optional:** Install `onnx` to allow automatic stub model generation when no model file is found (useful for testing the UI without a real model):
> ```bash
> pip install onnx
> ```

---

## 🚀 Usage

```bash
python main.py
```

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--source` | `0` | Camera index or path to video file |
| `--model` | `yolop-320-320.onnx` | Path to YOLOP ONNX model |
| `--skip` | `2` | Run inference every N frames (higher = faster) |
| `--conf` | `0.40` | Object detection confidence threshold |

### Examples

```bash
# Use default webcam
python main.py

# Use a video file
python main.py --source road_footage.mp4

# Custom model and confidence
python main.py --model yolop-320-320.onnx --conf 0.35

# Reduce CPU load — infer every 3 frames
python main.py --skip 3
```

---

## ⌨️ Keyboard Controls

| Key | Action |
|---|---|
| `E` | Toggle engage / standby |
| `D` | Toggle drivable area overlay |
| `L` | Toggle lane line overlay |
| `B` | Toggle bounding boxes |
| `P` | Toggle path plan |
| `M` | Toggle bird's-eye view panel |
| `W` | Toggle warning overlays |
| `N` | Toggle night / adaptive mode |
| `R` | Start / stop dashcam recording |
| `C` | Open IPM calibration UI |
| `Enter` | Save IPM calibration to file |
| `+` / `-` | Increase / decrease brightness |
| `]` / `[` | Increase / decrease detection confidence |
| `S` | Save screenshot |
| `Q` / `ESC` | Quit |

---

## 📂 Output Files

| File | Description |
|---|---|
| `ipm_calibration.json` | Saved IPM perspective calibration |
| `screenshot_<timestamp>.jpg` | Manual screenshots |
| `dashcam_<timestamp>.avi` | Recorded dashcam footage |
| `events_<timestamp>.csv` | Logged lane departure / near-miss events |

---

## 🏗️ Architecture

```
main.py
├── preprocess()           # Frame → normalised ONNX input blob
├── seg_mask()             # Segmentation tensor → binary mask
├── decode_det()           # Raw detections → NMS filtered boxes
│
├── DetectionStabiliser    # IOU-vote buffer to eliminate flickering
├── LaneKalman             # 6-state Kalman filter per lane line
├── IPMTransform           # Perspective warp (camera ↔ BEV)
├── CalibrationUI          # Drag-and-drop IPM corner calibration
├── SpeedEstimator         # Optical flow → km/h estimate
├── LaneAnalyser           # Full lane geometry, path, curvature, offset
├── SteeringModel          # Lane → steer angle with EMA + deadband
├── LeadTracker            # Lead vehicle distance + TTC
├── BEVPanel               # Bird's-eye view rendering
├── AdaptiveDisplay        # Brightness / night mode adaptation
├── DashcamRecorder        # Video writer wrapper
└── EventLogger            # CSV event log for departures / near-misses
```

---

## ⚠️ Notes

- If `yolop-320-320.onnx` is not found, a **zero-output stub model** is automatically built (requires `onnx` package) so the UI can be tested without a real model.
- Inference runs on **CPU** via ONNX Runtime. Using `--skip 2` or higher is recommended for smooth framerates on most machines.
- On Linux, **auto-exposure is disabled** on webcam sources for more consistent frames.
- The speed estimator uses optical flow — it gives a relative motion estimate, not GPS-accurate speed.

---

## 📄 License

```
MIT License

Copyright (c) 2025 IMApurbo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

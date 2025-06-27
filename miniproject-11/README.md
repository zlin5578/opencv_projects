## 🏗️ Project Structure

```
mini‑project11/
├── WebCamSave.py           # starter (kept for reference)
├── data.yaml               # dataset config for custom YOLO (optional)
├── yolov5s.pt              # best trained YOLOv5 model (80 COCO classes)
└── README.md               # <‑‑ this file
```

---

## 🔧 Setup

```bash
# 1. clone repo and enter mini‑project folder
git clone <your‑repo>
cd mini‑project11

# 2. create virtual env (optional)
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. install deps
pip install -r requirements.txt                  # OpenCV, torch, numpy, ultralytics …
```

---

## 🚀 Usage

```bash
# Run with default webcam (index 0)
python WebCamSave.py

# Run on a recorded video file
python WebCamSave.py -f sample.mov -o output.avi
```

**Hot‑keys**

| Key   | Action                                              |
| ----- | --------------------------------------------------- |
| **s** | Draw a new ROI to start tracking (multiple allowed) |
| **c** | Cancel / remove the last ROI                        |
| **r** | Reset – clear all tracks                            |
| **q** | Quit the application                                |

---

## 🧠 Algorithms & Parameters

### 1 · YOLOv5 Object Detection

* model: `yolov5s.pt` (80 COCO classes) **or** custom weight `runs/train/live_detector/weights/best.pt`
* refresh rate: every **10 frames** (`--detect-every`)
* confidence ≥ 0.6  NMS IoU ≤ 0.7

### 2 · Optical‑Flow Tracking

* algorithm: **Pyramidal Lucas‑Kanade** (`cv2.calcOpticalFlowPyrLK`)
* corner init: `cv2.goodFeaturesToTrack` with

  * `qualityLevel = 0.03` — strong corners only
  * `minDistance  = 10` px — avoid clutter
* trail length: 100 centre points (`--max-trails`)

---

## ✨ Results

| Metric                          | Value                                             |
| ------------------------------- |---------------------------------------------------|
| Avg FPS (CPU, 640×480)          | 18–22                                             |
| Detection mAP (if custom model) | see `runs/train/.../results.png` in miniproject 9 |

---

# YOLO + Optical Flow Object Tracking Report

## Overview
This project combines YOLOv5 object detection with Lucas-Kanade (LK) optical flow tracking. The system detects objects in a video and allows users to manually select regions (ROIs) for motion tracking.

## Target Object(s)
Any object class supported by the YOLOv5 model can be detected. Common targets include persons, vehicles, traffic signs, etc. ROIs for tracking are user-selected via mouse input.

## YOLO Model
- **Type:** YOLOv5 (`yolov5s.pt` by default)
- **Loaded via:** `torch.hub.load("ultralytics/yolov5", "custom", path=...)`
- **Thresholds:** Confidence = 0.6, IoU = 0.7
- **Inference Size:** 640x640

## Optical Flow
- **Algorithm:** Lucas-Kanade (cv2.calcOpticalFlowPyrLK)
- **Features:** Tracked using `cv2.goodFeaturesToTrack` within user-defined ROIs
- **Parameters:**  
  - Window Size: 21×21  
  - Max Pyramid Levels: 3  
  - Termination Criteria: 30 iterations or ε = 0.01
---


## 📚 References

1. Ultralytics YOLOv5: [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
2. OpenCV documentation: [https://docs.opencv.org/master/](https://docs.opencv.org/master/)

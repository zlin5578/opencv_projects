#!/usr/bin/env python
"""
yolo_flow_tracker.py
────────────────────
• YOLOv5 object detection (refresh every N frames, thick coloured boxes)
• LK optical-flow tracking on user-selected ROIs

Keys
----
s : draw a new ROI
c : cancel the last ROI
r : reset all ROIs
q : quit
"""

import argparse, time
from collections import deque
from itertools import cycle

import cv2, numpy as np, torch

# ───────────── CLI ─────────────
ap = argparse.ArgumentParser(description="YOLOv5 + LK optical-flow tracker")
ap.add_argument("-f", "--file",              type=str, help="video file (omit for webcam)")
ap.add_argument("--model",                  default="yolov5s.pt")
ap.add_argument("--conf",        type=float, default=0.6)
ap.add_argument("--iou",         type=float, default=0.7)
ap.add_argument("--detect-every",type=int,  default=10)
ap.add_argument("--max-trails",  type=int,  default=100)
ap.add_argument("-o", "--out", type=str, help="Output video file name")
args = ap.parse_args()

# ───────────── YOLOv5 ─────────────
print("[INFO] loading YOLOv5 …")
model = torch.hub.load("ultralytics/yolov5", "custom", path=args.model)
model.conf, model.iou = args.conf, args.iou
names = model.names
BOX_COLORS = [tuple(np.random.randint(64, 256, 3).tolist()) for _ in range(len(names))]
BOX_THICK  = 4  # thicker detection frame

# ───────────── Video source ─────────────
if args.file:
    cap = cv2.VideoCapture(args.file)
else:
    cap = cv2.VideoCapture(0)  # 0 is the default camera
time.sleep(2.0)
if not cap.isOpened():
    raise SystemExit("[ERROR] Cannot open the video source.")

# ───────────── VideoWriter (if output specified) ─────────────
out = None
if args.out:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 'XVID' or 'MJPG' for AVI, 'mp4v' for MP4
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30  # fallback
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(args.out, fourcc, fps, (width, height))

# ───────────── LK params ─────────────
lk = dict(winSize=(21, 21), maxLevel=3,
          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
prev_gray = None

# ───────────── trackers ─────────────
tracks, next_id = [], 0
ROI_COLORS = cycle([(255,0,0),(0,255,0),(0,255,255),(255,0,255),(255,255,0),(0,128,255)])

print("[INFO] s:new ROI | c:cancel | r:reset | q:quit")
frame_idx, t0 = 0, time.time()
yolo_boxes = np.empty((0, 6))

while True:
    ok, frame = cap.read()
    if not ok:
        break
    disp = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ───── YOLO ─────
    if frame_idx % args.detect_every == 0:
        res = model(frame, size=640)
        yolo_boxes = res.xyxy[0].cpu().numpy()

    for x1, y1, x2, y2, conf, cls in yolo_boxes:
        cls = int(cls)
        col = BOX_COLORS[cls % len(BOX_COLORS)]
        cv2.rectangle(disp, (int(x1), int(y1)), (int(x2), int(y2)), col, BOX_THICK)
        cv2.putText(disp, f"{names[cls]} {conf:.2f}",
                    (int(x1), int(y1) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 4)

    # ───── flow update ─────
    updated = []
    for tr in tracks:
        p_prev = tr["pts"]
        p_next, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p_prev, None, **lk)
        if p_next is None:
            continue
        good_new = p_next[st == 1]
        good_old = p_prev[st == 1]
        if len(good_new) < 5:
            continue

        # update bbox & trail
        x, y, w, h = cv2.boundingRect(good_new.reshape(-1, 1, 2))
        cx, cy = x + w // 2, y + h // 2
        tr.update(bbox=np.array([x, y, x + w, y + h]),
                  pts=good_new.reshape(-1, 1, 2))
        tr["trail"].append((cx, cy))
        updated.append(tr)

        # draw per-point flow
        for (n, o) in zip(good_new, good_old):
            a, b = map(int, n.ravel())
            c, d = map(int, o.ravel())
            cv2.line(disp, (c, d), (a, b), (0, 255, 0), 1)
            cv2.circle(disp, (a, b), 3, (0, 0, 255), -1)

        # draw centroid trail
        for i in range(1, len(tr["trail"])):
            cv2.line(disp, tr["trail"][i-1], tr["trail"][i], tr["color"], 3)

    tracks[:] = updated
    prev_gray = gray.copy()
    frame_idx += 1

    # ───── HUD: FPS + key legend ─────
    fps = frame_idx / (time.time() - t0)
    cv2.putText(disp, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # legend text (black)
    legend = ["Key   Action",
              "s : draw ROI",
              "c : cancel ROI",
              "r : reset all",
              "q : quit"]
    x0, y0, line_h = 10, 55, 22
    for i, txt in enumerate(legend):
        cv2.putText(disp, txt, (x0, y0 + i * line_h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

    cv2.imshow("YOLOv5 + LK Optical-Flow", disp)
    key = cv2.waitKey(1) & 0xFF
    if out:
        out.write(disp)

    # ───── keyboard ─────
    if key == ord('q'):
        break
    elif key == ord('r'):
        tracks.clear();  print("[INFO] all ROIs cleared")
    elif key == ord('c') and tracks:
        removed = tracks.pop()
        print(f"[INFO] ROI ID {removed['id']} cancelled")
    elif key == ord('s'):
        roi = cv2.selectROI("YOLOv5 + LK Optical-Flow", frame,
                            fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("ROI selector")
        x, y, w, h = roi
        if w and h:
            mask = np.zeros_like(gray); mask[y:y+h, x:x+w] = 255
            p0 = cv2.goodFeaturesToTrack(
                gray, mask=mask, maxCorners=200,
                qualityLevel=0.03, minDistance=10, blockSize=7)
            if p0 is not None:
                tracks.append(dict(
                    id=next_id,
                    bbox=np.array([x, y, x + w, y + h]),
                    pts=p0,
                    color=next(ROI_COLORS),
                    trail=deque([(x + w // 2, y + h // 2)],
                                maxlen=args.max_trails)
                ))
                print(f"[INFO] ROI #{next_id} added ({len(p0)} points)")
                next_id += 1

cap.release()
cv2.destroyAllWindows()
if out:
    out.release()

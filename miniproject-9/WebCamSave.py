import cv2
import torch
import numpy as np
import argparse
import time

# Load model 
# model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/live_detector/weights/best.pt')
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt')
model.conf = 0.6  # confidence threshold 
model.iou = 0.7    # NMS IoU threshold 

# Set up argument parser
parser = argparse.ArgumentParser(description="YOLOv5 object detection on webcam or video file")
parser.add_argument("-f", "--file", type=str, help="Path to input video file")
args = parser.parse_args()

# Video source: webcam or file
if args.file:
    vs = cv2.VideoCapture(args.file)
    if not vs.isOpened():
        print(f"[ERROR] Cannot open video file: {args.file}")
        exit(1)
else:
    vs = cv2.VideoCapture(0)
    if not vs.isOpened():
        print("[ERROR] Cannot access webcam.")
        exit(1)

# Wait for camera to warm up
time.sleep(2.0)

print("[INFO] Press 'q' to quit.")

# Initialize FPS timer
prev_time = time.time()

# Detection loop
while True:
    ret, frame = vs.read()
    if not ret:
        break

    # Inference
    results = model(frame)

    # Render results and make a writable copy
    annotated_frame = results.render()[0].copy()

    # FPS calculation
    current_time = time.time()
    fps = 1.0 / (current_time - prev_time)
    prev_time = current_time

    # Overlay FPS text on frame
    cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("YOLOv5 Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
vs.release()
cv2.destroyAllWindows()
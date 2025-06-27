### Group 1  
### CS 5330  
### Assignment 9-3

## YOLO Object Detection with Webcam

This project uses **YOLOv5** to detect selected objects in real-time from the webcam feed. When one of the specified target classes is detected, the system automatically saves a **5-second video** segment of the webcam feed.

### Target Classes

- Mouse  
- Keyboard  
- Cell Phone  

These classes are selected from the standard `coco.names` file.

### How It Works

- The webcam is continuously monitored using OpenCV.
- YOLOv5 detects objects in real-time.
- If a **Mouse**, **Keyboard**, or **Cell Phone** is detected, 5-second video is saved using OpenCV’s `VideoWriter`.


### Dependencies

- Python 3.x  
- OpenCV  
- PyTorch  
- YOLOv5  
- playsound (optional for fun activity)

### ▶Run Instructions

```bash
python3 WebCamSave_Yolo.py

To train the YOLOv5 model:
```bash
python train_live_detector.py
```

To run real-time detection from webcam or video file:
```bash
# From webcam
python WebCamSave.py

# From video file
python WebCamSave.py -f videoplayback.webm -o output.avi
```

---

## Objective  
This project aims to build a real-time object detection system using YOLOv5. The goal is to detect **Stop Signs** and **Traffic Signals** in live webcam feed or pre-recorded videos. The dataset was labeled using Roboflow and trained using transfer learning with pretrained YOLOv5 weights.

---

## Process

### Training Phase (`train_live_detector.py`)  
- YOLOv5s model is trained for 50 epochs with:
  - Image size: 640×640
  - Batch size: 16
  - Optimizer: SGD or Adam (default by YOLOv5)
  - Weights: pretrained `yolov5s.pt`
- Model output is saved to `runs/train/live_detector/weights/best.pt`

- The `data.yaml` defines:
  - Paths to training and validation images
  - `nc`: number of classes (2)
  - `names`: `['Stop Sign', 'Traffic Signal']`

---

## Evaluation Summary  
- Trained with 639 images and validated on 40 images  
- YOLOv5 provides efficient mAP and high-speed detection  

---

## Image Dataset  
Labeled using Roboflow and LabelMe
Structured into:
```
dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
```


## Dataset 
[Dataset](https://drive.google.com/drive/folders/1kn633YzWyRsOBoxdStryYixEJ1VWSuc6?usp=sharing)

---

## Directory Structure

```
.
├── dataset/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   ├── labels/
│   │   ├── train/
│   │   └── val/
├── data.yaml
├── train_live_detector.py
├── WebCamSave.py
├── requirements.txt
└── README.md
```

---

## Dependencies
```bash
pip install torch>=1.7 opencv-python matplotlib PyYAML numpy seaborn
pip install yolov5
```

---

## Controls
---------------------------------------
| Key |             Action            |
|-----|-------------------------------|
| `q` | Quit webcam or video playback |
---------------------------------------


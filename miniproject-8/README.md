### Live Object Classification


## Usage  
To run the live classifier:  
```bash
python WebCamSave.py
```

To train and evaluate the model:  
```bash
python cnn_torch.py
```

---

## Objective  
This project aims to develop a real-time object classification system that detects and classifies objects in video frames using a Convolutional Neural Network (CNN) implemented in PyTorch. 

To enhance reliability, the system integrates ODIN for out-of-distribution (OOD) detection.

---

## Process  

### Training Phase (`cnn_torch.py`)  
Images were collected into class-specific folders under `imagenet_dataset/`. Each image was resized to 224x224, normalized, and converted from BGR to RGB.

A `ResNet-50` model was used with pretrained weights, modified for 4-class classification. Label binarization was performed using `LabelBinarizer` from `scikit-learn`. The dataset was split into training and test sets using `train_test_split`.

The model was trained using the Adam optimizer and cross-entropy loss on GPU (if available). A confusion matrix and classification report were generated for evaluation.

The following files were saved to `saved_model/`:
- `resnet50_model.pth`
- `label_binarizer.pkl`
- `normalization.pkl`

---

### Live Inference Phase (`WebCamSave.py`)  
Inference is performed using a PyTorch ResNet-50 model enhanced with the **ODIN** method to detect out-of-distribution objects.

Each frame from webcam or video:
- Is resized to 224×224 and normalized with dataset mean/std
- Undergoes **ODIN preprocessing**:
  - Applies temperature scaling 
  - Perturbs inputs with gradient-based noise
- The model outputs are passed through a softmax layer

**If the max softmax probability is below a threshold, the object is marked as "Unknown".**

Overlay:
- Top prediction with confidence
- OOD predictions labeled as Unknown
- FPS displayed in real-time

---

## Evaluation Summary  
- **Overall accuracy:** 89%  
- **Best performance:** `TV` (Precision: 0.94, Recall: 0.98, F1-Score: 0.96)  
- `Keyboard` also showed strong performance with 0.90 F1-Score  
- `Remote Control` had lower precision (0.74) but decent recall (0.89)  
- Some misclassification occurred between `Cell Phone` and `Remote Control`  
- **Macro Avg F1-Score:** 0.88


---

## Image Dataset
[Image Dataset Download Link](https://drive.google.com/drive/folders/1Bg2O4aDBp8TymS0B7PkoNTu5sZYB5Bdl?usp=sharing)


## Directory Structure

```
.
├── cnn_torch.py
├── image_scrapper
│   ├── batch_download_imagenet.py
│   ├── classes_in_imagenet.csv
│   └── downloader.py
├── imagenet_class_info.json
├── imagenet_dataset
│   ├── Cell Phone
│   │   ├── (Image Dataset of Cellphone)
│   ├── Keyboard
│   │   ├── (Image Dataset of Keyboard)
│   ├── Remote Control
│   │   ├── (Image Dataset of Remote Control)
│   └── TV
│       ├── (Image Dataset of TV)
├── output
│   ├── accuracy_plot.png
│   ├── confusion_matrix.png
│   └── PyTorch.PNG
├── README.md
├── saved_model
│   ├── label_binarizer.pkl
│   ├── normalization.pkl
│   ├── object_classifier.keras
│   ├── object_classifier.tflite
│   ├── resnet18_model.pth
│   ├── resnet50_model.pth
│   └── torch_cnn.pth
└── WebCamSave.py
```

---

## Dependencies
```bash
pip install torch torchvision scikit-learn opencv-python matplotlib seaborn
```

---

## Controls
| Key | Action |
|-----|--------|
| `q` | Quit program |


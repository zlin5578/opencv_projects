import numpy as np
import cv2
import argparse
import pickle
import time
import os
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

odin_temp=5.0
odin_epsilon= 0.001
ood_threshold=0.5

# Argument parser
parser = argparse.ArgumentParser(description="Video classification with ODIN")
parser.add_argument("-f", "--file", type=str, help="Path to input video file")
parser.add_argument("-o", "--out", type=str, help="Output video filename")
args = parser.parse_args()

# Load label binarizer and normalization
with open("saved_model/label_binarizer.pkl", "rb") as f:
    lb = pickle.load(f)
with open("saved_model/normalization.pkl", "rb") as f:
    norm = pickle.load(f)
mean, std = norm["mean"], norm["std"]

# Load PyTorch model
print("[INFO] Using PyTorch backend with ODIN integration")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=False)
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, len(lb.classes_))
)
model.load_state_dict(torch.load("saved_model/resnet50_model.pth", map_location=device))
model.eval()
model.to(device)

def odin_preprocessing(inputs, model, temperature, epsilon):
    """
    ODIN preprocessing: applies temperature scaling and input preprocessing
    """
    inputs.requires_grad_(True)
    
    # Forward pass with temperature scaling
    outputs = model(inputs)
    outputs = outputs / temperature
    
    # Calculate the criterion (negative log-likelihood of predicted class)
    labels = outputs.argmax(dim=1)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, labels)
    
    # Backward pass to get gradients
    loss.backward()
    
    # Apply input preprocessing
    gradient = inputs.grad.data
    gradient = gradient.sign()
    inputs_processed = inputs - epsilon * gradient
    
    # Clear gradients
    inputs.grad.data.zero_()
    
    return inputs_processed

def predict_with_odin(frame):
    """
    Prediction with ODIN out-of-distribution detection
    """
    # Preprocess frame
    image = cv2.resize(frame, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
    image = (image - mean) / std
    image = np.transpose(image, (2, 0, 1))
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Apply ODIN preprocessing
    with torch.enable_grad():
        image_processed = odin_preprocessing(image, model, odin_temp, odin_epsilon)
    
    # Make prediction with processed input
    with torch.no_grad():
        output = model(image_processed)
        
        # Apply temperature scaling for final prediction
        output_scaled = output / odin_temp
        preds = F.softmax(output_scaled, dim=1).cpu().numpy()[0]
        
        # Calculate ODIN score (maximum softmax probability)
        odin_score = np.max(preds)
        
        # Determine if sample is OOD
        is_ood = odin_score < ood_threshold
    
    return preds, odin_score, is_ood

def predict_baseline(frame):
    """
    Baseline prediction without ODIN (for comparison)
    """
    image = cv2.resize(frame, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
    image = (image - mean) / std
    image = np.transpose(image, (2, 0, 1))
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image)
        preds = F.softmax(output, dim=1).cpu().numpy()[0]
        baseline_score = np.max(preds)
    
    return preds, baseline_score

# Initialize video source
if args.file:
    vs = cv2.VideoCapture(args.file)
else:
    vs = cv2.VideoCapture(0)

# Output settings
time.sleep(2.0)
width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
if args.out:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(args.out, fourcc, 20.0, (width, height), True)

# Counters and timers
frame_count = 0
start_time = time.time()
top1_counter = Counter()
ood_counter = 0
top_k = 4

print(f"[INFO] ODIN Parameters:")
print(f"  Temperature: {odin_temp}")
print(f"  Epsilon: {odin_epsilon}")
print(f"  OOD Threshold: {ood_threshold}")

# Main loop
while True:
    ret, frame = vs.read()
    if not ret:
        break

    frame_count += 1
    current_time = time.time()
    
    # Get predictions with ODIN
    preds, odin_score, is_ood = predict_with_odin(frame)
    
    # Get baseline prediction for comparison
    baseline_preds, baseline_score = predict_baseline(frame)
    
    # Get top predictions
    top_indices = preds.argsort()[-top_k:][::-1]
    if is_ood:
        top1_label = "None"
    else:
        top1_label = lb.classes_[preds.argmax()]

    
    if not is_ood:
        top1_counter[top1_label] += 1
    else:
        ood_counter += 1

    # Create overlay with increased height for ODIN info
    overlay = frame.copy()
    overlay_height = 10 + top_k * 25 + 120  # Increased height for ODIN info
    cv2.rectangle(overlay, (5, 5), (400, overlay_height), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

    # Display top predictions
    for i, idx in enumerate(top_indices):
        label_text = f"{lb.classes_[idx]}: {preds[idx] * 100:.1f}%"
        cv2.putText(frame, label_text, (10, 25 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display main prediction with OOD status
    y_offset = 25 + top_k * 25 + 10
    if is_ood:
        pred_text = f"OOD DETECTED: {top1_label}"
        pred_color = (0, 0, 255)  # Red for OOD
    else:
        if is_ood:
            pred_text = "Prediction: None (OOD Detected)"
            pred_color = (0, 0, 255)
        else:
            pred_text = f"Prediction: {top1_label}"
            pred_color = (255, 255, 0)

    
    cv2.putText(frame, pred_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, pred_color, 2)

    # Display ODIN score
    cv2.putText(frame, f"ODIN Score: {odin_score:.3f}", (10, y_offset + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Display baseline score for comparison
    cv2.putText(frame, f"Baseline Score: {baseline_score:.3f}", (10, y_offset + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    # Display OOD statistics
    ood_rate = (ood_counter / frame_count) * 100
    cv2.putText(frame, f"OOD Rate: {ood_rate:.1f}%", (10, y_offset + 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Display FPS
    fps = frame_count / (current_time - start_time + 1e-5)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if args.out:
        out.write(frame)
    cv2.imshow("Real-time Classification with ODIN", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Print final statistics
print(f"\n[INFO] Final Statistics:")
print(f"Total frames processed: {frame_count}")
print(f"OOD detections: {ood_counter}")
print(f"OOD rate: {(ood_counter / frame_count) * 100:.2f}%")
print(f"Top predictions distribution:")
for label, count in top1_counter.most_common(10):
    print(f"  {label}: {count} ({(count / frame_count) * 100:.1f}%)")

vs.release()
if args.out:
    out.release()
cv2.destroyAllWindows()
# ========================== #
#      IMPORT LIBRARIES     #
# ========================== #
# Core libraries for data handling and image processing
import os
import glob
import pickle
from collections import Counter
import numpy as np
import cv2
from PIL import Image

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# ========================== #
#     CONFIG & CONSTANTS     #
# ========================== #
# Use GPU if available for faster training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# Define important paths for dataset, model saving, label binarizer, and output
dataset_path = "imagenet_dataset"
MODEL_PATH = 'saved_model/resnet50_model.pth'      
LB_PATH = 'saved_model/label_binarizer.pkl'        
OUTPUT_IMG_DIR = 'output'                          

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

# ========================== #
#      LOAD IMAGE DATASET    #
# ========================== #
print("[INFO] Loading images...")
imagePaths = glob.glob(os.path.join(dataset_path, "**", "*.jpg"), recursive=True)
data, labels = [], []

# Loop over all image file paths
for imagePath in imagePaths:
    image = cv2.imread(imagePath)  # Load image using OpenCV
    if image is not None:
        image = cv2.resize(image, (224, 224))                # Resize image to 224x224 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)       # Convert from BGR to RGB
        image = image.astype("float32") / 255.0              # Normalize pixel values to [0, 1]
        data.append(image)
        label = imagePath.split(os.path.sep)[-2]             
        labels.append(label)

# ========================== #
#     ENCODE LABELS          #
# ========================== #
# One-hot encode labels using sklearn's LabelBinarizer
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# Save label binarizer to disk so it can be used later for inference
with open(LB_PATH, 'wb') as f:
    pickle.dump(lb, f)

# ========================== #
#  TRAIN/VAL/TEST SPLIT      #
# ========================== #
# Split into training, validation, and test sets
trainX, testX, trainY, testY = train_test_split(np.array(data), np.array(labels), test_size=0.2, random_state=42)
trainX, valX, trainY, valY = train_test_split(trainX, trainY, test_size=0.2, random_state=42)  # 20% of train = val

# ========================== #
#       CLASS WEIGHTS        #
# ========================== #
# Compute class weights to handle class imbalance
label_counts = Counter([y.argmax() for y in trainY])
weights = [len(trainY) / (len(lb.classes_) * label_counts[i]) for i in range(len(lb.classes_))]
weights = torch.tensor(weights, dtype=torch.float32).to(device)

# ========================== #
#       CUSTOM DATASET       #
# ========================== #
class CustomDataset(Dataset):
    """
    Custom PyTorch Dataset to serve image data for training, validation, or testing.
    Supports optional transforms like augmentation and normalization.
    """
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Convert float image back to uint8 for PIL transforms
        image = (self.images[idx] * 255).astype("uint8")
        image = Image.fromarray(image)

        # Apply any transforms if defined (augmentation + normalization)
        if self.transform:
            image = self.transform(image)

        # Convert label to float tensor (one-hot)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label

# ========================== #
#       DATA TRANSFORMS      #
# ========================== #
# Data augmentation for training: crop, flip, jitter, normalize
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
])

# Minimal preprocessing for validation/testing
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ========================== #
#         DATALOADERS        #
# ========================== #
# Create batched data loaders
train_loader = DataLoader(CustomDataset(trainX, trainY, transform=train_transform), batch_size=32, shuffle=True)
val_loader = DataLoader(CustomDataset(valX, valY, transform=test_transform), batch_size=32, shuffle=False)
test_loader = DataLoader(CustomDataset(testX, testY, transform=test_transform), batch_size=32, shuffle=False)

# ========================== #
#        MODEL SETUP         #
# ========================== #
# Load pre-trained ResNet50 model from torchvision
model = models.resnet50(pretrained=True)

# Freeze the early layers so only later layers + classifier head are trainable
for param in list(model.parameters())[:-30]:
    param.requires_grad = False

# Replace final classification head
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, len(lb.classes_))  # Output layer matches number of classes
)
model = model.to(device)

# ========================== #
#      LOSS & OPTIMIZER      #
# ========================== #
criterion = nn.CrossEntropyLoss(weight=weights)  # Handle class imbalance
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)  # AdamW = Adam + L2 regularization
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)  # Smooth learning rate decay

# ========================== #
#       TRAINING LOOP        #
# ========================== #
print("[INFO] Training network...")
best_val_acc = 0
patience = 10  # Stop training early if val acc doesn’t improve for 10 epochs
patience_counter = 0

for epoch in range(100):
    model.train()
    train_loss, train_correct = 0, 0

    # Training loop
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels.argmax(dim=1))  # Use class indices
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_correct += (outputs.argmax(1) == labels.argmax(1)).sum().item()

    # Validation loop
    model.eval()
    val_loss, val_correct = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels.argmax(1))
            val_loss += loss.item()
            val_correct += (outputs.argmax(1) == labels.argmax(1)).sum().item()

    train_acc = 100 * train_correct / len(trainX)
    val_acc = 100 * val_correct / len(valX)
    scheduler.step()  # Adjust learning rate

    print(f"[EPOCH {epoch+1}/100] Train Loss: {train_loss/len(train_loader):.4f}, "
          f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, Patience Counter: {patience_counter}")

    # Save best model or early stop if no improvement
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), MODEL_PATH)
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"[INFO] Early stopping at epoch {epoch+1}")
            break

print(f"[INFO] Model saved to {MODEL_PATH}")

# ========================== #
#         EVALUATION         #
# ========================== #
# Load the best saved model
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()
all_preds, all_labels = [], []

# Make predictions on test data
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.argmax(dim=1).cpu().numpy())

# Print precision, recall, F1-score for each class
print(classification_report(all_labels, all_preds, target_names=lb.classes_))

# Plot confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=lb.classes_, yticklabels=lb.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_IMG_DIR, "confusion_matrix.png"))
plt.close()

print(f"[INFO] Best validation accuracy: {best_val_acc:.2f}%")

# ========================== #
#   DISPLAY SAMPLE OUTPUTS   #
# ========================== #
# Visualize a few random test predictions
sample_indices = np.random.choice(len(testX), 3, replace=False)

for i, idx in enumerate(sample_indices):
    image_np = (testX[idx] * 255).astype("uint8")
    image_pil = Image.fromarray(image_np)

    # Reapply test transforms before inference
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        pred_label = output.argmax(dim=1).item()

    # Plot the image with predicted and true label
    plt.subplot(1, 3, i + 1)
    plt.imshow(image_np)
    true_label = lb.classes_[testY[idx].argmax()]
    plt.title(f"Pred: {lb.classes_[pred_label]}\nTrue: {true_label}")
    plt.axis('off')

# Save and display all three images
plt.tight_layout()
plt.savefig("output/accuracy_plot.png")
plt.show()

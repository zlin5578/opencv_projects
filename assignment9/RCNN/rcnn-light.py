import os
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.applications import MobileNet
from keras.layers import Dense, GlobalAveragePooling2D
from keras import Model
from keras.callbacks import EarlyStopping

# Dataset paths
image_path = "./data/sm_images"
label_dir = "./data/sm_labels"

# Data holders
train_images, train_labels = [], []
svm_images, svm_labels = [], []

# Class map
class_map = {
    "Remote Control": [1, 0],
    "Keyboard": [0, 1]
}
label_names = ["Remote Control", "Keyboard"]

# IoU calculation
def get_iou(bb1, bb2):
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1 = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    area2 = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])
    return intersection / float(area1 + area2 - intersection)

# Enable OpenCV optimization
cv2.setUseOptimized(True)
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

# Process CSV-labeled data
for e, filename in enumerate(os.listdir(label_dir)):
    if not filename.endswith(".csv"):
        continue
    df = pd.read_csv(os.path.join(label_dir, filename))
    if df.empty:
        continue
    image_file = df.iloc[0]['filename']
    image = cv2.imread(os.path.join(image_path, image_file))
    if image is None:
        print(f"Skipping: {image_file} (not found)")
        continue

    print(f"[{e}] Processing {image_file}")
    gtvalues = []
    for _, row in df.iterrows():
        label = row['class']
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        gtvalues.append({"x1": x1, "x2": x2, "y1": y1, "y2": y2})
        cropped = image[y1:y2, x1:x2]
        resized = cv2.resize(cropped, (224, 224))
        svm_images.append(resized)
        svm_labels.append(class_map[label])

    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    results = ss.process()
    counter, falsecounter = 0, 0
    for i, (x, y, w, h) in enumerate(results):
        if i >= 300:  # Reduced from 2000 to 300
            break
        for gt in gtvalues:
            iou = get_iou(gt, {"x1": x, "x2": x+w, "y1": y, "y2": y+h})
            cropped = image[y:y+h, x:x+w]
            if cropped.shape[0] == 0 or cropped.shape[1] == 0:
                continue
            resized = cv2.resize(cropped, (224, 224))
            if counter < 50 and iou > 0.6:
                train_images.append(resized)
                train_labels.append(1)
                counter += 1
            elif falsecounter < 50 and iou < 0.2:
                train_images.append(resized)
                train_labels.append(0)
                falsecounter += 1
            if falsecounter < 5 and iou < 0.2:
                svm_images.append(resized)
                svm_labels.append([0, 0])

# Convert to arrays
X_new = np.array(train_images)
Y_new = np.array(train_labels)

# MobileNet model for feature extraction + binary classifier
base_model = MobileNet(include_top=False, input_shape=(224, 224, 3), weights='imagenet')

# Freeze base layers for performance
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=x)
model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['acc'])
model.summary()

# Train feature extractor
model.fit(X_new, Y_new, batch_size=32, epochs=5, validation_split=0.1, shuffle=True)

# Add softmax classifier head for multi-class
x = model.layers[-2].output
Y = Dense(2, activation='softmax')(x)
final_model = Model(inputs=model.input, outputs=Y)
final_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
final_model.summary()

# Add early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

# Train final model
hist_final = final_model.fit(
    np.array(svm_images),
    np.array(svm_labels),
    batch_size=32,
    epochs=5,
    validation_split=0.1,
    shuffle=True,
    callbacks=[early_stop]
)

# Save final model weights
final_model.save('my_model_weights.h5')

# Plot loss
plt.plot(hist_final.history['loss'])
plt.plot(hist_final.history['val_loss'])
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["Loss", "Validation Loss"])
plt.savefig("chart_loss.png")
plt.show()
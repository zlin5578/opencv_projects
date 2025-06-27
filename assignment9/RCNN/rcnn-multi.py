import os
import cv2
import keras
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras import Model

# Paths
image_path = "./data/sm_images"
label_dir = "./data/sm_labels"

# Lists to store data
train_images = []
train_labels = []
svm_images = []
svm_labels = []

# Label encoding and name mapping for multi-class
class_map = {
    "Remote Control": [1, 0],
    "Keyboard": [0, 1]
}
label_names = ["Remote Control", "Keyboard"]

# IoU calculation
def get_iou(bb1, bb2):
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])
    return intersection_area / float(bb1_area + bb2_area - intersection_area)

# OpenCV optimization
cv2.setUseOptimized(True)
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

# Iterate over all CSVs in label_dir
for e, csv_file in enumerate(os.listdir(label_dir)):
    if not csv_file.endswith(".csv"):
        continue

    try:
        df = pd.read_csv(os.path.join(label_dir, csv_file))
        if df.empty:
            continue

        filename = df.iloc[0]['filename']
        image = cv2.imread(os.path.join(image_path, filename))
        if image is None:
            print(f"Image not found: {filename}")
            continue

        print(e, filename)
        gtvalues = []

        for _, row in df.iterrows():
            label_name = row["class"]
            x1, y1, x2, y2 = int(row["xmin"]), int(row["ymin"]), int(row["xmax"]), int(row["ymax"])
            gtvalues.append({"x1": x1, "x2": x2, "y1": y1, "y2": y2})
            label_vec = class_map.get(label_name, [0, 0])
            timage = image[y1:y2, x1:x2]
            resized = cv2.resize(timage, (224, 224), interpolation=cv2.INTER_AREA)
            svm_images.append(resized)
            svm_labels.append(label_vec)

        ss.setBaseImage(image)
        ss.switchToSelectiveSearchFast()
        ssresults = ss.process()
        imout = image.copy()
        counter = 0
        falsecounter = 0
        flag = False

        for e, result in enumerate(ssresults):
            if e < 2000 and not flag:
                x, y, w, h = result
                for gtval in gtvalues:
                    iou = get_iou(gtval, {"x1": x, "x2": x + w, "y1": y, "y2": y + h})
                    timage = imout[y:y + h, x:x + w]

                    if counter < 30 and iou > 0.7:
                        resized = cv2.resize(timage, (224, 224), interpolation=cv2.INTER_AREA)
                        train_images.append(resized)
                        train_labels.append(1)
                        counter += 1
                    elif falsecounter < 30 and iou < 0.3:
                        resized = cv2.resize(timage, (224, 224), interpolation=cv2.INTER_AREA)
                        train_images.append(resized)
                        train_labels.append(0)
                        falsecounter += 1
                    if falsecounter < 5 and iou < 0.3:
                        resized = cv2.resize(timage, (224, 224), interpolation=cv2.INTER_AREA)
                        svm_images.append(resized)
                        svm_labels.append([0, 0])

                if counter >= 30 and falsecounter >= 30:
                    flag = True
    except Exception as ex:
        print(f"Error processing {csv_file}: {ex}")
        continue

# Training binary classifier
X_new = np.array(train_images)
Y_new = np.array(train_labels)

vgg = tf.keras.applications.vgg16.VGG16(include_top=True, weights='imagenet')
for layer in vgg.layers[:-2]:
    layer.trainable = False
x = vgg.get_layer('fc2').output
x = Dense(1, activation='sigmoid')(x)
model = Model(vgg.input, x)
model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['acc'])
model.summary()
model.fit(X_new, Y_new, batch_size=16, epochs=1, verbose=1, validation_split=0.05, shuffle=True)

# Training multi-class classifier
x = model.get_layer('fc2').output
Y = Dense(2)(x)
final_model = Model(model.input, Y)
final_model.compile(loss='hinge', optimizer='adam', metrics=['accuracy'])
final_model.summary()

hist_final = final_model.fit(np.array(svm_images), np.array(svm_labels),
                             batch_size=16, epochs=2, verbose=1,
                             shuffle=True, validation_split=0.05)
final_model.save('my_model_weights.h5')

# NMS
def non_max_suppression(boxes, overlapThresh):
    if len(boxes) == 0:
        return []
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
    return boxes[pick].astype("int")

# Plot loss
plt.plot(hist_final.history['loss'])
plt.plot(hist_final.history['val_loss'])
plt.title("Model Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Loss", "Validation Loss"])
plt.savefig('chart_loss.png')
plt.show()

# Test on one image
test_image_file = '1629269_cf658cc39a_jpg.rf.7afd0c6341c4bc33ed4e31852b5dbfb5.jpg'
image = cv2.imread(os.path.join(image_path, test_image_file))
ss.setBaseImage(image)
ss.switchToSelectiveSearchFast()
ssresults = ss.process()

imOut = image.copy()
boxes = []

for e, result in enumerate(ssresults):
    if e < 50:
        x, y, w, h = result
        timage = image[y:y + h, x:x + w]
        resized = cv2.resize(timage, (224, 224), interpolation=cv2.INTER_AREA)
        resized = np.expand_dims(resized, axis=0)
        out = final_model.predict(resized)

        pred_class = np.argmax(out[0])
        score = np.max(out[0])
        if score > 0.5:
            print("Detected:", label_names[pred_class])
            boxes.append([x, y, x + w, y + h, score])

boxes = np.array(boxes)
nms_boxes = non_max_suppression(boxes, overlapThresh=0.3)

for box in nms_boxes:
    x1, y1, x2, y2 = box[:4]
    cv2.rectangle(imOut, (x1, y1), (x2, y2), (0, 255, 0), 1, cv2.LINE_AA)
    score = box[4]
    label = label_names[np.argmax([score, 1 - score])]
    cv2.putText(imOut, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

plt.imshow(imOut)
plt.axis('off')
plt.show()
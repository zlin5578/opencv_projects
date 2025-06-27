import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random

# Load the trained multi-class model
final_model = tf.keras.models.load_model('my_model_weights.h5')

# Class labels
label_names = ["Remote Control", "Keyboard"]

# Pick a random image from the sm_test folder
test_folder = './data/sm_test'
image_files = [f for f in os.listdir(test_folder) if f.endswith('.jpg')]
if not image_files:
    raise FileNotFoundError("No .jpg images found in sm_test folder.")
image_file = random.choice(image_files)
image_path = os.path.join(test_folder, image_file)
print(f"Testing on random image: {image_file}")

# Non-Max Suppression
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

# Load image
image = cv2.imread(image_path)
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
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
        out = final_model.predict(resized, verbose=0)
        pred_class = np.argmax(out[0])
        score = np.max(out[0])
        if score > 0.5:
            boxes.append([x, y, x + w, y + h, score, pred_class])

boxes = np.array(boxes)

if len(boxes) > 0:
    nms_boxes = non_max_suppression(boxes[:, :5], overlapThresh=0.3)
    for box in nms_boxes:
        x1, y1, x2, y2, score = box[:5]
        pred_class = int(box[5]) if box.shape[0] > 5 else np.argmax(score)
        label = label_names[pred_class]
        cv2.rectangle(imOut, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(imOut, f"{label} ({score:.2f})", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# Display result
plt.imshow(cv2.cvtColor(imOut, cv2.COLOR_BGR2RGB))
plt.title(f"Detected: {image_file}")
plt.axis('off')
plt.show()
import os
import cv2
import numpy as np
import tensorflow as tf
import time

# Load trained R-CNN model
final_model = tf.keras.models.load_model('my_model_weights.h5')

# Class labels
label_names = ["Remote Control", "Keyboard"]

# Non-Maximum Suppression function
def non_max_suppression(boxes, overlapThresh):
    if len(boxes) == 0:
        return []
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    pick = []
    x1, y1, x2, y2, scores = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4]
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

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Could not access webcam.")
    exit(1)

print("[INFO] Press 'q' to quit.")
time.sleep(2.0)

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    imOut = frame.copy()
    boxes = []

    # Selective Search on the current frame
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(frame)
    ss.switchToSelectiveSearchFast()
    ssresults = ss.process()

    for e, result in enumerate(ssresults):
        if e < 50:
            x, y, w, h = result
            roi = frame[y:y + h, x:x + w]
            if roi.shape[0] == 0 or roi.shape[1] == 0:
                continue
            resized = cv2.resize(roi, (224, 224), interpolation=cv2.INTER_AREA)
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

    # FPS
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(imOut, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Live R-CNN Detection", imOut)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
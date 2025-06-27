import cv2
import time

# Open default webcam
# vs = cv2.VideoCapture(0)
# Read and Display Video File
vs = cv2.VideoCapture("load_vedio.mp4")
time.sleep(2.0)

# Initialize toggle flags
crop_on = False
resize_on = False
blur_on = False
box_on = False
text_on = False
thresh_on = False
new_func_on = False
grey_on = False

while True:
    ret, frame = vs.read()
    if not ret:
        break

    output = frame.copy()

    # Crop (center 300x300)
    if crop_on:
        h, w = output.shape[:2]
        startY = h // 2 - 150
        startX = w // 2 - 150
        output = output[startY:startY + 300, startX:startX + 300]

    # Resize (to 300x300)
    if resize_on:
        output = cv2.resize(output, (300, 300))

    # Blur
    if blur_on:
        output = cv2.GaussianBlur(output, (15, 15), 0)

    # Draw yellow box
    if box_on:
        h, w = output.shape[:2]
        cv2.rectangle(output, (50, 50), (w - 100, h - 100), (0, 200, 200), 10)

    # Add text
    if text_on:
        cv2.putText(output, "Zhipeng Ling", (60, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 5)

    # Threshold
    if thresh_on:
        gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        _, output = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)

    # grey
    if grey_on:
        output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)

    # New Function: draw a yellow circle in center
    if new_func_on:
        h, w = output.shape[:2]
        cv2.circle(output, (w // 2, h // 2), 50, (0, 255, 255), 3)

    # Show frame
    cv2.imshow("Webcam Output", output)
    key = cv2.waitKey(1) & 0xFF

    # Toggle keys
    if key in (ord("c"), ord("C")): crop_on = not crop_on
    if key in (ord("r"), ord("R")): resize_on = not resize_on
    if key in (ord("b"), ord("B")): blur_on = not blur_on
    if key in (ord("a"), ord("A")): box_on = not box_on
    if key in (ord("t"), ord("T")): text_on = not text_on
    if key in (ord("g"), ord("G")): thresh_on = not thresh_on
    if key in (ord("n"), ord("N")): new_func_on = not new_func_on
    if key in (ord("m"), ord("M")): grey_on = not grey_on

    # Quit
    if key == ord("q") or key == ord("Q"):
        break

# Cleanup
vs.release()
cv2.destroyAllWindows()

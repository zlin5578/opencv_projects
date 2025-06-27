import cv2
import sys
import imutils

img = cv2.imread("flower.png")

if img is None:
    sys.exit("Could not read the image.")

(h, w, d) = img.shape
print("width=", w, "height=", h, "depth=", d)


# ========================
# 1. Show Region of Interest (ROI)
# ========================
startY, endY = 20, 320
startX, endX = 40, 340
roi = img[startY:endY, startX:endX]
cv2.imshow("1. Region of Interest", roi)

# ========================
# 2. Resize Image
# ========================
resized = cv2.resize(img, (150, 150))
cv2.imshow("2. Resized Image", resized)


# ========================
# 3. Rotate Image (45 degrees counter-clockwise)
# ========================
(h, w) = img.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, 45, 1.0)
rotated = cv2.warpAffine(img, M, (w, h))
cv2.imshow("3. Rotated Image", rotated)

# ========================
# 4. Smooth Image (Gaussian Blur)
# ========================
smoothed = cv2.GaussianBlur(img, (11, 11), 0)
cv2.imshow("4. Smoothed Image", smoothed)

# ========================
# 5. Drawing (Rectangle, Circle, Line) - Yellow (0,255,255)
# ========================
drawn = img.copy()
cv2.rectangle(drawn, (50, 50), (200, 200), (0, 255, 255), 2)
cv2.circle(drawn, (300, 300), 40, (0, 255, 255), 2)
cv2.line(drawn, (100, 100), (400, 400), (0, 255, 255), 2)
cv2.imshow("5. Drawing Shapes", drawn)

# ========================
# 6. Add Text
# ========================
texted = img.copy()
cv2.putText(texted, "Zhipeng Ling", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
cv2.imshow("6. Text Added", texted)

# ========================
# 7. Convert to Grayscale
# ========================
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("7. Grayscale Image", gray)

# ========================
# 8. Edge Detection (Canny)
# ========================
edges = cv2.Canny(gray, 100, 200)
cv2.imshow("8. Canny Edges", edges)

# ========================
# 9. Thresholding
# ========================
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
cv2.imshow("9. Thresholded Image", thresh)

# ========================
# 10. Detect and Draw Contours
# ========================
contours_img = img.copy()
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(contours_img, contours, -1, (0, 255, 255), 2)
cv2.imshow("10. Contours", contours_img)


k = cv2.waitKey(0)
if k == ord("s"):
    cv2.imwrite("flower-copy.png", img)

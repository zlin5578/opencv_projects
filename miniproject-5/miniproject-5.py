import cv2
import numpy as np
import time

sift = cv2.SIFT_create()

def detect_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(gray, None)
    return kp, des

def match_features(des1, des2):
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(des1, des2, k=2)
    return [m for m, n in matches if m.distance < 0.75 * n.distance]

def stitch(frame1, frame2, reproj_thresh=5.0):
    kp1, des1 = detect_features(frame1)
    kp2, des2 = detect_features(frame2)

    if des1 is None or des2 is None:
        print("Insufficient descriptors")
        return frame1

    matches = match_features(des1, des2)
    if len(matches) < 4:
        print("Insufficient matches")
        return frame1

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC, reproj_thresh)
    if H is None:
        print("Homography computation failure")
        return frame1

    h1, w1 = frame1.shape[:2]
    h2, w2 = frame2.shape[:2]

    corners_frame2 = np.array([[0,0], [w2,0], [w2,h2], [0,h2]], dtype=np.float32).reshape(-1,1,2)
    warped_corners = cv2.perspectiveTransform(corners_frame2, H)

    all_corners = np.concatenate((np.array([[0,0],[w1,0],[w1,h1],[0,h1]], dtype=np.float32).reshape(-1,1,2),warped_corners), axis=0)

    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    translation = [-xmin, -ymin]
    panorama_size = (xmax - xmin, ymax - ymin)
    h_translation = np.array([[1,0,translation[0]],[0,1,translation[1]],[0,0,1]]) @ H
    warped_frame2 = cv2.warpPerspective(frame2, h_translation, panorama_size)

    panorama = np.zeros((panorama_size[1], panorama_size[0], 3), dtype=np.uint8)
    panorama[translation[1]:translation[1]+h1, translation[0]:translation[0]+w1] = frame1

    mask1 = np.any(panorama != 0, axis=2).astype(np.uint8)
    mask2 = np.any(warped_frame2 != 0, axis=2).astype(np.uint8)

    overlap_mask = cv2.bitwise_and(mask1, mask2)

    only_panorama = cv2.bitwise_and(mask1, cv2.bitwise_not(overlap_mask))
    only_warped = cv2.bitwise_and(mask2, cv2.bitwise_not(overlap_mask))

    blend = np.zeros_like(panorama)
    alpha = 0.3

    for c in range(3):
        blend[:,:,c] = panorama[:,:,c] * (1 - alpha) + warped_frame2[:,:,c] * alpha

    panorama[only_warped == 1] = warped_frame2[only_warped == 1]
    panorama[only_panorama == 1] = panorama[only_panorama == 1]
    panorama[overlap_mask == 1] = blend[overlap_mask == 1]

    gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x,y,w,h = cv2.boundingRect(contours[0])
        panorama = panorama[y:y+h, x:x+w]

    return panorama

start_time = time.time()
frame_count = 0

cap = cv2.VideoCapture(0)
capturing = False
captured_frames = []
last_capture_time = 0
sample_interval = 0.3

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    current_time = time.time()
    fps_diff = current_time - start_time
    
    if fps_diff >= 1.0:
        fps = frame_count / fps_diff
        start_time = current_time
        frame_count = 0
    else:
        fps = frame_count / max(fps_diff, 0.001)

    display_frame = frame.copy()
    
    if capturing:
        cv2.putText(display_frame, f"FPS: {int(fps)} (Capturing. Press 'a' to stop)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
    if not capturing:
        cv2.putText(display_frame, f"FPS: {int(fps)} (Press 's' to start capturing or 'q' to quit.)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
    cv2.imshow("Webcam", display_frame)
    key = cv2.waitKey(1) & 0xFF

    current_time = time.time()

    if key == ord('q'):
        capturing = False
        cap.release()
        cv2.destroyAllWindows()
        exit()

    elif key == ord('s'):
        if not capturing:
            captured_frames.clear()
        capturing = True

    elif key == ord('a'):
        if capturing:
            capturing = False
            break

    if capturing and (current_time - last_capture_time) > sample_interval:
        captured_frames.append(frame.copy())
        last_capture_time = current_time

cap.release()
cv2.destroyAllWindows()

panorama = captured_frames[0]
for i in range(1, len(captured_frames)):
    panorama = stitch(panorama, captured_frames[i])

cv2.imshow("Panorama", panorama)
key = cv2.waitKey(0) & 0xFF

if key == ord('s'):
    filename = "Panorama.jpg"
    cv2.imwrite(filename, panorama)

cv2.destroyAllWindows()

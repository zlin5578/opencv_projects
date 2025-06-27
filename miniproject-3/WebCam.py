import cv2
import time
import numpy as np
import math

cap = cv2.VideoCapture(0)
prev_time = time.time()
time.sleep(1.0)

frame_dur = 1.0 / cap.get(cv2.CAP_PROP_FPS)
warps = [0,0,0,0,0] #X, Y, degree, scale, perspective
translating = 0
condition = ''

while True:
    start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    key = cv2.waitKey(1) & 0xFF
    key_char = chr(key).lower()

    if key_char == 't':
        translating = 0 if translating == 1 else 1
    elif translating == 1:
        if key_char == 'h':  #Left
            warps[0] -= 1
        elif key_char == 'k':  #Down
            warps[1] -= 1
        elif key_char == 'j':  #Up
            warps[1] += 1
        elif key_char == 'l':  #Right
            warps[0] += 1
    elif key_char == 'r': #CWRotation
        warps[2] += 1
    elif key_char == 'e': #CCWRotation
        warps[2] -= 1
    elif key_char == 's': #Upscale
        warps[3] = 1
    elif key_char == 'a': #Downscale
        warps[3] = -1
    elif key_char == 'p': #Perspective
        warps[4] = 1
    elif key_char == 'b': #Original
        warps = [0, 0, 0, 0, 0]
        translating = 0
        
    if key_char == 'q':
        break

    height, width = frame.shape[:2]

    mod_frame = frame.copy()
    if warps[0] != 0 or warps[1] != 0: ##Translation``
        dx = 5 * warps[0]
        dy = 5 * warps[1]
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        mod_frame = cv2.warpAffine(mod_frame, M, (width, height))
    if warps[2] != 0: #Rotation
        angle = -5 * warps[2]
        M = cv2.getRotationMatrix2D((width // 2, height // 2), angle, 1.0)
        mod_frame = cv2.warpAffine(mod_frame, M, (width, height))
    if warps[3] != 0:  # Scaling
        scale_factor = 1 + warps[3] * 0.05
        scale_factor = max(scale_factor, 0.05)

        # Scale image
        scaled = cv2.resize(mod_frame, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LANCZOS4)

        # Resize
        sh, sw = scaled.shape[:2]
        if sh > height or sw > width:
            # Crop to fit
            start_y = (sh - height) // 2
            start_x = (sw - width) // 2
            mod_frame = scaled[start_y:start_y + height, start_x:start_x + width]
        else:
            # Center the image
            top = (height - sh) // 2
            bottom = height - sh - top
            left = (width - sw) // 2
            right = width - sw - left
            mod_frame = cv2.copyMakeBorder(scaled, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    
    if warps[4] == 1: # Perspective
        src = np.float32([
        [0.25 * width, 0.25 * height],
        [0.75 * width, 0.25 * height],
        [0.75 * width, 0.75 * height],
        [0.25 * width, 0.75 * height],
        ])
        dst = np.float32([
        [0.1 * width, 0.4 * height],
        [0.9 * width, 0.1 * height],
        [0.8 * width, 0.9 * height],
        [0.2 * width, 0.9 * height],
        ])
        M = cv2.getPerspectiveTransform(src, dst)
        mod_frame = cv2.warpPerspective(mod_frame, M, (width, height))
    
    elapsed = time.time() - start_time
    sleep_time = max(0, frame_dur - elapsed)
    time.sleep(sleep_time)

    curr_time = time.time()
    fps = 1.0 / (curr_time - start_time)

    text = f"FPS: {math.ceil(fps)}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    cv2.putText(frame, text, (10, 10 + th), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    combined = cv2.hconcat([frame, mod_frame])
    cv2.imshow("Mini-Project 3", combined)

cap.release()
cv2.destroyAllWindows()
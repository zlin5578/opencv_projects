# 🎥 Webcam Video Processor

## 📚 Assignment Overview

This project is a part of **Assignment 2-2: Playing with Camera**, for the Computer Vision course. The application allows users to apply real-time image processing effects to video input from a file using OpenCV in Python. Users can interact with the video by toggling various effects via keyboard input.


---

## ✅ Features Implemented
| Feature         | Key Press        | Description                                                                                                                                      |
|----------------|------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|
| Read Video      | Auto             | Loads and plays a local video file instead of webcam (uses `cv2.VideoCapture("filename")`), change `cv2.VideoCapture(0)` to switch to webcam.   |
| Crop Video     | `c` or `C`       | Toggles cropping a 300×300 region from the center of the frame                                                                                   |
| Resize Video   | `r` or `R`       | Toggles resizing the frame to 300×300 pixels                                                                                                     |
| Blur Video     | `b` or `B`       | Toggles Gaussian blur with a kernel size of 15×15                                                                                                |
| Add a Box      | `a` or `A`       | Toggles drawing a yellow box around the main area of the frame                                                                                   |
| Add Text       | `t` or `T`       | Toggles displaying the name "Zhipeng Ling" in yellow text                                                                                        |
| Thresholding   | `g` or `G`       | Toggles converting the frame to grayscale and applying binary threshold                                                                          |
| Grayscale      | `m` or `M`       | Toggles converting the frame to grayscale (non-binary), showing it in classic black-and-white style                                              |
| New Function   | `n` or `N`       | Toggles drawing a centered yellow circle with radius 50 pixels                                                                                   |
| Quit Program   | `q` or `Q`       | Exits the application                                                                                                                            |


All keys are toggle switches — pressing the same key again disables the effect.
---n
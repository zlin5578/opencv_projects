
## Panoramic Photo (Real-Time Panorama Creation Using Live Camera)

## Process

Frames captured from the webcam are converted to grayscale. Next, SIFT is used to detect keypoints & descriptors. The brute force method is used to compare all descriptors between two sampled frames. KNN is used to find the best matching descriptors between the frames and Lowe's Ratio Test is used to filter the results for better quality matches. The keypoints are used to compute a homography matrix to match the perspectives of the frames. RANSAC is used to filter out outliers. A blurring mask is applied to help eliminate seams between frames and empty blackspace is removed using thresholding.

## Dependencies
- pip install opencv-python numpy
- pip install opencv-contrib-python


## Directory
- miniproject-5.py // main script
- README.md // readme file


## Controls
| Key | Action                      |
|-----|-----------------------------|
| `s` | Initiate frame capture      |
| `a` | Terminate frame capture     |
| `q` | Exit the program            |

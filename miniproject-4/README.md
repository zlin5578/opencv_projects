
# Real-Time Feature Matching with ORB and SIFT

This project uses two webcams to perform real-time feature detection and matching using ORB and SIFT. Users can switch between the two algorithms and visualize live matches, match scores, and FPS on screen.

# Key Features
- Real-time SIFT and ORB keypoint detection
- Match visualization with lines and distance labels
- Match score based on descriptor distance
- FPS and mode display


# Dependencies Required:
- pip install opencv-python numpy
- pip install opencv-contrib-python

# File Directory 
- miniproject4.py // main script for video capture, Sift or Orb matching, and visualizing result with FPS
- feature_matching.py // functions for SIFT and ORB
- feature_matching_score.py // function for match scoring and anotation
- WebCamDual.py // Dual Cam display setup  

# Controls
-------------------------------------
| Key | Action                      |
|-----|-----------------------------|
| `s` | Switch to SIFT              |
| `o` | Switch to ORB （default）    |
| `q` | Quit the program            |
-------------------------------------

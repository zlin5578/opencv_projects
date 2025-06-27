## Lane Detector

## Usage

To run:
python WebCamSave.py -f video_file_name
e.g. python WebCamSave.py -f data\lane_test1.mp4



## Process
This project aims to develop a lane detection system for identifying lane lines on a road. 

Video frames are pre-preprocessed, using a threshold filter for white lines, grayscale conversion, blurring filter, and canny edge detection. The edges are fed into the Hough Line Transform. 

The left and right lanes are separated based on their slope. The lines are averaged using their slopes & intercepts and the lines are extrapolated from the bottom of the frame to the middle of the frame. A rolling average and outlier detection is used to prevent jumpy lane movement. If no lanes are detected, the ROI will eventually be increased to help improve the chance of detection. 

The final lines will be drawn on the frame. Lane detection & FPS display can be toggled uses keyboard controls.


## Dependencies
- pip install opencv-python numpy
- pip install opencv-contrib-python



## Directory
- WebCamSave.py (Main script)
- roi_utils.py (Region of interest function)
- lineAverage.py (Post-processing (Lane detection, variable ROI, line averaging) functions)
- image_processing.py (Pre-processing function)
- README.md (Readme)



## Controls
| Key | Action                      |
|-----|-----------------------------|
| `s` | Toggle lane detection       |
| `f` | Toggle the FPS display      |
| `q` | Exit the program            |

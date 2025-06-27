# USAGE: python WebCamSave.py -f video_file_name

# Import the necessary packages
import cv2
import numpy as np
import time
import argparse
from image_processing import preprocess_image
from roi_utils import get_roi
from lineAverage import draw_lane_lines_from_hough

# Set up argument parser
parser = argparse.ArgumentParser(description="Video file path or camera input")
parser.add_argument("-f", "--file", type=str, help="Path to the video file")
# parser.add_argument("-o", "--out", type=str, help="Output video file name")

args = parser.parse_args()

# Check if the file argument is provided, otherwise use the camera
if args.file:
    vs = cv2.VideoCapture(args.file)
else:
    vs = cv2.VideoCapture(0)  # 0 is the default camera

# time.sleep(2.0) # Webcam capture initialization delay

# Get the default resolutions
width  = int(vs.get(3))
height = int(vs.get(4))

# Define the codec and create a VideoWriter object
# out_filename = args.out
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter(out_filename, fourcc, 20.0, (width, height), True)

masked = True
show_fps = False
start_time = time.time()
frame_count = 0
last_capture_time = 0

left_problem = False
right_problem = False

# Loop over the frames from the video stream
while True:
    # Grab the frame from video stream
    ret, frame = vs.read()
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

    #-----------------------------------------
    #TODO: Parameters for the Hough Transform
    threshold = 50
    minLineLength = 150
    maxLineGap = 20
    #-----------------------------------------

    # Get the trapezoidal ROI
    if right_problem or left_problem:
        masked_image, roi = get_roi(frame, 0.9, 0.2)
    else:
        masked_image, roi = get_roi(frame, 0.2, 0.4)
        
    # Preprocessing
    processed_image, edges, contours = preprocess_image(masked_image)

    # For HoughLinesP:
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold, minLineLength, maxLineGap)
    
    if show_fps:
        cv2.putText(frame, f"FPS: {int(fps)}. Press 's' to toggle lane detection & 'f' to toggle FPS display.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:
        cv2.putText(frame, "Press 's' to toggle lane detection & 'f' to toggle FPS display.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if masked:
        output, left_problem, right_problem = draw_lane_lines_from_hough(frame, lines)
    else:
        output = frame

    # Display Original Frame
    # cv2.imshow("Original Frame", frame)

    # Display processed image with contours
    # cv2.imshow("Processed Image", processed_image)

    # #Display masked image
    # cv2.imshow("Masked Image", masked_image)

    # Display the output with lane lines
    cv2.imshow("Lanes", output)

    # Write the frame to the output video file
    # if args.out:
    #     out.write(frame)

    # show the output frame
    # cv2.imshow("Frame", processed_image)
    key = cv2.waitKey(1) & 0xFF

    # Toggle lane detetcion
    if key == ord("s"):
        masked = not masked

    # Toggle lane detetcion
    if key == ord("f"):
        show_fps = not show_fps

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# Release the video capture object
vs.release()
# out.release()
cv2.destroyAllWindows()

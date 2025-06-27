
### This project utilizes python and Open CV libraries to develop a webcam application in performing various image processing transformation such as translation, rotation, scaling, and perspective

# Packages require
pip3 install opencv-python
pip3 install dumpy

### The following are transformation implemented with original image display on left and transformed image display on right. Frame rate of video processing is displayed on top left of image to ensure smooth operation in real time

# Control
+-----+----------------------------+
| Key | Feature                    |
+-----+----------------------------+
|     | Translation                |
| t   | (Press t for translation,  |
|     | press t again to stop)     |
| h   | move left by 5 pixel       |
| j   | move left by 5 pixel       |
| k   | move left by 5 pixel       |
| l   | move left by 5 pixel       |
|     | Rotation                   |
| r   | rotation by +5 degree      |
| e   | rotation by -5 degree      |
|     | Scaling                    |
| s   | Scale up by +5%            |
| a   | Scale by -5%.              |
|     | Perspective                |
| p   | Apply perspective          |
|     | Back                       | 
| b   | Undo all transformation    | 
|     | to original image.         | 
+-----+----------------------------+

# Usage of the functions
• Translation – Shift the image horizontally and/or vertically.
• Rotation – Rotate the image around a specified point by 5 degrees
• Scaling – Resize the image by scaling up or down by 5%
• Perspective – Apply a perspective warp to simulate a change in viewpoint.





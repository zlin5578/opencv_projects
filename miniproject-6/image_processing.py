import cv2
import numpy as np

def preprocess_image(image):
    """
    Filters for white lines, applies grayscale, Gaussian blur,
    Canny edge detection, and draws contours.

    Args:
        image (ndarray): The image/frame to process.

    Returns:
        output (ndarray): Image with drawn contours.
        edges (ndarray): Canny edge-detected binary image.
        contours (list): List of detected contours.
    """
    if image is None:
        raise ValueError("Input image is None")

    # Convert to HSV and create a mask for white regions
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 50, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    # Apply white mask to filter only white regions
    white_filtered = cv2.bitwise_and(image, image, mask=white_mask)

    # Grayscale conversion and Gaussian blur
    gray = cv2.cvtColor(white_filtered, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours on the edge image
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on a copy of the original image
    output = image.copy()
    cv2.drawContours(output, contours, -1, (0, 255, 0), 2)

    return output, edges, contours
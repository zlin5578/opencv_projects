import numpy as np
import cv2

def get_roi(image, top_width_ratio, top_height_ratio):
    """
    Creates and applies a trapezoidal mask to the input image.

    Args:
        image (ndarray): Input image.
        top_width_ratio (float): Ratio of image width for trapezoid top.
        top_height_ratio (float): Ratio of image height for trapezoid top height.

    Returns:
        masked_image (ndarray): Image with only the trapezoid ROI visible.
        roi (ndarray): The trapezoid polygon points (4, 2).
    """
    height, width = image.shape[:2]

    top_width = int(width * top_width_ratio)
    bottom_width = width  # Full width at the bottom
    top_y = int(height * top_height_ratio)

    # Define trapezoid points
    roi = np.array([
        [(width - bottom_width) // 2, height],         # Bottom-left
        [(width - top_width) // 2, top_y],             # Top-left
        [(width + top_width) // 2, top_y],             # Top-right
        [(width + bottom_width) // 2, height]          # Bottom-right
    ], dtype=np.int32)

    # Create mask
    mask = np.zeros_like(image)

    if len(image.shape) == 2:  # Grayscale image
        cv2.fillPoly(mask, [roi], 255)
    else:  # Color image
        cv2.fillPoly(mask, [roi], (255, 255, 255))

    # Apply mask
    masked_image = cv2.bitwise_and(image, mask)

    return masked_image, roi
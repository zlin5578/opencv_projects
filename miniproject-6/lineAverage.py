import cv2
import numpy as np
from collections import deque

# Buffer for rolling average & miss counter to switch ROI
left_q = deque(maxlen=25)
right_q = deque(maxlen=25)
MISS_LIMIT = 10.0
left_miss_count = 0
right_miss_count = 0

def filter_outliers(lines, threshold=1.5):
    """
    Filters lines based on slope using IQR.
    Args:
        lines: list of (slope, intercept) tuples
        threshold: how strict the IQR cut is (higher = looser)

    Returns:
        filtered list
    """
    if not lines:
        return []

    slopes = np.array([s for s, _ in lines])
    q1, q3 = np.percentile(slopes, [25, 75])
    iqr = q3 - q1
    lower = q1 - threshold * iqr
    upper = q3 + threshold * iqr

    return [line for line in lines if lower <= line[0] <= upper]

def draw_lane_lines_from_hough(frame, lines, slope_threshold=0.35, line_color=(0, 0, 255), thickness=11):
    """
    Processes Hough lines to identify left and right lanes, averages them, and draws them on the frame.

    Args:
        frame (ndarray): Original image (BGR).
        lines (list): Output from cv2.HoughLinesP().
        slope_threshold (float): Minimum absolute slope to classify as left/right lane.
        line_color (tuple): BGR color for the lane lines.
        thickness (int): Thickness of drawn lines.

    Returns:
        output_image (ndarray): Frame with lane lines drawn.
    """
    global left_q, right_q, left_miss_count, right_miss_count

    height, width = frame.shape[:2]
    left_lines, right_lines = [], []

    if lines is None:
        return frame, left_miss_count > MISS_LIMIT, right_miss_count > MISS_LIMIT  # Nothing to draw

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0:
            continue  # avoid division by zero
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1

        if slope < -slope_threshold:
            left_lines.append((slope, intercept))
        elif slope > slope_threshold:
            right_lines.append((slope, intercept))

    if left_lines:
        filtered_left = filter_outliers(left_lines)
        if filtered_left:
            left_avg = np.mean(filtered_left, axis=0)
            left_q.append(left_avg)
            left_miss_count -= 0.05
        else:
            left_miss_count += 1.0
    else:
        left_miss_count += 1.0

    if right_lines:
        filtered_right = filter_outliers(right_lines)
        if filtered_right:
            right_avg = np.mean(filtered_right, axis=0)
            right_q.append(right_avg)
            right_miss_count -= 0.05  # Decrease when detected
        else:
            right_miss_count += 1.0
    else:
        right_miss_count += 1.0

    def make_line(points, height, y_ratio=0.5):
        """
        Averages slope/intercept and extrapolates the line from the bottom to y_ratio height.
        """
        if len(points) == 0:
            return None

        avg_slope, avg_intercept = np.mean(points, axis=0)
        y1 = height
        y2 = int(height * y_ratio)

        try:
            x1 = int((y1 - avg_intercept) / avg_slope)
            x2 = int((y2 - avg_intercept) / avg_slope)
            return (x1, y1, x2, y2)
        except ZeroDivisionError:
            return None

    left_line = make_line(left_q, height)
    right_line = make_line(right_q, height)

    output_image = frame.copy()
    if left_line:
        cv2.line(output_image, (left_line[0], left_line[1]), (left_line[2], left_line[3]), line_color, thickness)
    if right_line:
        cv2.line(output_image, (right_line[0], right_line[1]), (right_line[2], right_line[3]), line_color, thickness)

    return output_image, left_miss_count > MISS_LIMIT, right_miss_count > MISS_LIMIT
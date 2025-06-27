import numpy as np
import matplotlib.pyplot as plt
import cv2


# Add scores
def draw_matches_with_scores(img_match, img1, kp1, img2, kp2, matches, max_matches):
    # Convert image from BGR to RGB for matplotlib display
    annotated_img = img_match.copy()

    # Image widths for keypoint position offset
    h1, w1 = img1.shape[:2]

    for i in range(max_matches):
        match = matches[i]
        pt1 = tuple(np.round(kp1[match.queryIdx].pt).astype(int))
        pt2 = tuple(np.round(kp2[match.trainIdx].pt).astype(int))
        pt2_shifted = (int(pt2[0] + w1), int(pt2[1]))  # shift second point by width of first image

        # Get score (distance)
        score = match.distance
        text = f"{score:.1f}"

        # Position to place the score: midpoint between matched keypoints
        mid_point = (int((pt1[0] + pt2_shifted[0]) / 2), int((pt1[1] + pt2_shifted[1]) / 2))

        # Draw text on the image
        cv2.putText(annotated_img, text, mid_point, cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5, color=(255, 0, 0), thickness=1, lineType=cv2.LINE_AA)

    return annotated_img

def calculate_score(matches, max_matches=5):
    """
    Calculates a quality score for a list of matches.
    The lower the average distance of top matches, the higher the score.
    """
    if not matches:
        return 0.0

    top_matches = matches[:max_matches]
    distances = [m.distance for m in top_matches if m.distance > 0]

    if not distances:
        return 0.0

    avg_distance = sum(distances) / len(distances)
    score = 1000.0 / avg_distance  # Inverse scoring: smaller distance → higher score

    return score

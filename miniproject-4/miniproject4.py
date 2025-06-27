import cv2
import time
from feature_matching import compute_sift_matches, compute_orb_matches
from feature_matching_score import calculate_score

def draw_matches_with_scores(img_match, img1, kp1, img2, kp2, matches, max_matches):
    """
    Annotates top matches with their distance score on the output image.
    """
    for i, match in enumerate(matches[:max_matches]):
        img1_idx = match.queryIdx
        img2_idx = match.trainIdx

        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt

        text = f"{match.distance:.0f}"
        cv2.putText(img_match, text,
                    ((int(x1) + int(x2) + img1.shape[1]) // 2,
                     (int(y1) + int(y2)) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 0, 0), 1, cv2.LINE_AA)
    return img_match

def main():
    # Open two video capture devices
    cap1 = cv2.VideoCapture(0)
    cap2 = cv2.VideoCapture(1)

    # Request 30 FPS from both cameras
    cap1.set(cv2.CAP_PROP_FPS, 30)
    cap2.set(cv2.CAP_PROP_FPS, 30)

    if not cap1.isOpened() or not cap2.isOpened():
        print("Error: Could not open one or both video sources.")
        return

    use_sift = False  # Default to ORB
    print("Press 's' to switch to SIFT, 'o' to switch to ORB, 'q' to quit.")

    while True:
        start_time = time.time()

        # Capture frames
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            print("Failed to capture frames.")
            break

        # Resize for consistency
        frame1 = cv2.resize(frame1, (640, 480))
        frame2 = cv2.resize(frame2, (640, 480))

        # Compute keypoints and matches
        if use_sift:
            kp1, kp2, matches = compute_sift_matches(frame1, frame2)
        else:
            kp1, kp2, matches = compute_orb_matches(frame1, frame2)

        # Draw top matches using OpenCV's drawMatches
        max_matches = 5
        matching_result = cv2.drawMatches(frame1, kp1,
                                          frame2, kp2,
                                          matches[:max_matches], None,
                                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # Annotate scores on the match lines
        matching_result = draw_matches_with_scores(matching_result,
                                                   frame1, kp1,
                                                   frame2, kp2,
                                                   matches, max_matches)

        # Compute match quality score
        score = calculate_score(matches, max_matches)

        # Calculate FPS
        fps = 1.0 / max((time.time() - start_time), 1e-5)

        # Overlay information on the result image
        cv2.putText(matching_result, f"Score: {score:.2f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(matching_result, f"FPS: {fps:.1f}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(matching_result, f"Mode: {'SIFT' if use_sift else 'ORB'}", (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Show the final image
        cv2.imshow("Feature Matching", matching_result)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            use_sift = True
        elif key == ord('o'):
            use_sift = False

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

import argparse
import math
import os

import cv2


def capture_frames(path_input, path_output, frame_rate):
    count = 0
    videoFile = path_input
    cap = cv2.VideoCapture(videoFile)  # capturing the video from the given path
    frameRate = cap.get(frame_rate)  # frame rate
    while cap.isOpened():
        print("Reading the frames...\n")
        frameId = cap.get(1)  # current frame number
        ret, frame = cap.read()
        if not ret:
            break
        if frameId % math.floor(frameRate) == 0:
            filename = "test%d.jpg" % count
            count += 1
        cv2.imwrite(os.path.join(path_output, filename), frame)
    cap.release()


# Driver Code
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse the path to the video file and frame rate"
    )
    parser.add_argument(
        "--path_input", type=str, help="Path to the video file",
    )
    parser.add_argument(
        "--frame_rate", type=int, default=5, help="Frame rate",
    )
    parser.add_argument(
        "--path_output", type=str, help="Path to the output frames",
    )
    args = parser.parse_args()

    path_input = args.path_input
    path_output = args.path_output
    frame_rate = args.frame_rate
    # Calling the function
    capture_frames(path_input, path_output, frame_rate)

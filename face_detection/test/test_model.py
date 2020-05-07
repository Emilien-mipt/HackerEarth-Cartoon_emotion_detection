import argparse
import os

import sys

sys.path.append("..")

from utils.warp import load_detector, warp

detection_config_path = "../../face_detection/models/detection_config.json"
OUTPUT_DIR = "test-detected"


def main():
    path_to_model = args.model
    path_to_input = args.test_image
    path_to_output = os.path.join(OUTPUT_DIR, "detected_" + path_to_input)
    detection_probability = args.probability
    detector = load_detector(path_to_model, detection_config_path)

    detections, extracted_objects_array = warp(
        detector, detection_probability, path_to_input, path_to_output
    )

    print(detections)

    for detection, object_path in zip(detections, extracted_objects_array):
        print(object_path)
        print(
            detection["name"],
            " : ",
            detection["percentage_probability"],
            " : ",
            detection["box_points"],
        )
        print("---------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Choose the set (train or validation) to warp"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="../../../face_detection/models/detection_model-ex-098--loss-0001.688.h5",
        help="Choose the detection model",
    )
    parser.add_argument(
        "--probability", type=int, default=50, help="Probability threshold",
    )
    parser.add_argument(
        "--test_image",
        type=str,
        help="Choose the path to the image that you want to test",
    )
    args = parser.parse_args()
    main()

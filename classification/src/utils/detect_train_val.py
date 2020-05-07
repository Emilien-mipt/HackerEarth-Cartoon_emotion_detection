import argparse
import os
import sys

sys.path.insert(
    1,
    "/home/etagiev/Documents/Competitions/HackerEarth/Detect_emotions_cartoons/face_detection/utils/",
)

from warp import warp, load_detector

TRAIN_PATH = "../../data/train_frames/"
VAL_PATH = "../../data/val_frames/"

TRAIN_OUTPUT_PATH = "../../data/train_warped_frames/"
VAL_OUTPUT_PATH = "../../data/val_warped_frames/"

detection_config_path = "../../../face_detection/models/detection_config.json"


def main():
    path_to_model = args.model
    detector = load_detector(path_to_model, detection_config_path)

    mode = args.set
    probability = args.probability

    print("Warp frames from the {} set: \n".format(mode))

    if mode == "train":
        input = TRAIN_PATH
        output = TRAIN_OUTPUT_PATH

    elif mode == "val":
        input = VAL_PATH
        output = VAL_OUTPUT_PATH

    for frame in os.listdir(input):
        print("Start warping frame {} ...".format(frame))
        input_path = os.path.join(input, frame)
        output_path = os.path.join(output, "detected_" + frame)
        detections, extracted_objects_array = warp(
            detector, probability, input_path, output_path
        )

        print("Detections: ", detections)
        print("Extracted object arrays: ", extracted_objects_array)

        for detection, object_path in zip(detections, extracted_objects_array):
            print(object_path)
            print(
                detection["name"],
                " : ",
                detection["percentage_probability"],
                " : ",
                detection["box_points"],
            )
            print("---------------------------------------------------------\n")


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
        "--set", type=str, default="train", help="The set (train or val) to warp",
    )
    parser.add_argument(
        "--probability", type=int, default=50, help="Probability threshold",
    )
    args = parser.parse_args()
    main()

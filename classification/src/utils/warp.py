import argparse
import os

from imageai.Detection.Custom import CustomObjectDetection

TRAIN_PATH = "../../data/train_frames/"
VAL_PATH = "../../data/val_frames/"

TRAIN_OUTPUT_PATH = "../data/train_warped_frames/"
VAL_OUTPUT_PATH = "../data/val_warped_frames/"


def main():
    detector = CustomObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(
        "../../face_detection/models/detection_model-ex-079--loss-0002.743.h5"
    )
    detector.setJsonPath("../../face_detection/models/detection_config.json")
    detector.loadModel()

    path = args.set
    probability = args.probability

    print("Warp frames from the {} set: \n".format(path))

    if path == "train":
        path = TRAIN_PATH
        output_path = TRAIN_OUTPUT_PATH

    else:
        path = VAL_PATH
        output_path = VAL_OUTPUT_PATH

    for frame in os.listdir(path):
        detections, extracted_objects_array = detector.detectObjectsFromImage(
            input_image=os.path.join(path, frame),
            output_image_path=os.path.join(output_path, "detected_" + frame),
            minimum_percentage_probability=probability,
            extract_detected_objects=True,
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
            print("---------------------------------------------------------\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Choose the set (train or validation) to warp"
    )
    parser.add_argument(
        "--set", type=str, default="train", help="The set (train or val) to warp",
    )
    parser.add_argument(
        "--probability", type=int, default=50, help="Probability threshold",
    )
    args = parser.parse_args()
    main()

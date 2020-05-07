import argparse
import os
import re
import sys

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(
    1,
    "/home/etagiev/Documents/Competitions/HackerEarth/Detect_emotions_cartoons/face_detection/utils/",
)

from warp import warp, load_detector

DATA_PATH = "../data/"
TEST_DIR = DATA_PATH + "test/"
OUTPUT_TEST_DIR = DATA_PATH + "test_warped_frames/"

MODEL_DETECTION_DIR = "../../face_detection/models/"
detection_config_path = "../../face_detection/models/detection_config.json"

MODEL_CLASSIFICATION_DIR = "../checkpoints/models/"

class_names = ["Angry", "Happy", "Sad", "Surprised", "Unknown"]


def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval


def natural_keys(text):
    """
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    float regex comes from https://stackoverflow.com/a/12643073/190597
    """
    return [atof(c) for c in re.split(r"[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)", text)]


def _load_classifier(path_to_model: str):
    """Load the trained model."""
    print("Loading the classifier {}...")
    classifier = torch.load(path_to_model)
    classifier.eval()
    print("The classifier has been loaded!")
    return classifier


def _image_loader(loader, image_name):
    image = Image.open(image_name)
    image = loader(image).float()
    image = image.clone().detach().requires_grad_(True)
    image = image.unsqueeze(0)
    return image


def _transform_input():
    data_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return data_transforms


def make_prediction(detector, detection_probability, classifier, data_transform):
    frame_name = []
    emotion_predictions = []
    device = torch.device("cpu")
    list_frames = os.listdir(TEST_DIR)
    list_frames.sort(key=natural_keys)
    for frame in list_frames:
        print("Start warping frame {} ...".format(frame))
        frame_name.append(frame)
        input_path = os.path.join(TEST_DIR, frame)
        output_path = os.path.join(OUTPUT_TEST_DIR, "detected_" + frame)

        detections, extracted_objects_array = warp(
            detector, detection_probability, input_path, output_path
        )

        if not detections:
            print("No one has been found on frame {}".format(frame))
            emotion_predictions.append("Unknown")
            continue
        else:
            path_to_input = ""
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
                path_to_input = object_path
                if detection["name"] == "Tom":
                    break
            # Send a warped image to the input of a classifier
            print("Sending a warped image to the input of a classifier...")
            with torch.set_grad_enabled(False):
                input_image = _image_loader(data_transform, path_to_input).to(device)
                classifier.to(device)
                preds = classifier(input_image)
                preds_class = preds.argmax(dim=1)
            emotion_predictions.append(class_names[preds_class])
    return (frame_name, emotion_predictions)


def make_submit(image_list, prediction_list):
    df = pd.DataFrame({"Frame_ID": image_list, "Emotion": prediction_list})
    df.to_csv("./submit.csv", index=False)


def main():
    # Load detector model
    detector = args.detector
    path_to_detector = os.path.join(MODEL_DETECTION_DIR, detector)
    face_detector = load_detector(path_to_detector, detection_config_path)

    # Load classidier model
    classifier = args.classifier
    path_to_classifier = os.path.join(MODEL_CLASSIFICATION_DIR, classifier)
    emotion_classifier = _load_classifier(path_to_classifier)
    # Init transforms from image to Pytorch tensor for classifier input
    image_transform = _transform_input()

    # Make predictions
    detector_probability = args.probability
    test_name_list, predictions = make_prediction(
        face_detector, detector_probability, emotion_classifier, image_transform
    )

    # Create a submission file
    make_submit(test_name_list, predictions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse arguments for the inference of the trained model and metric evaluation on the test set"
    )
    parser.add_argument(
        "--detector", type=str, help="Choose the model for face detection"
    )
    parser.add_argument(
        "--probability", type=int, default=50, help="Probability threshold",
    )
    parser.add_argument("--classifier", type=str, help="Name of the classifier model")
    args = parser.parse_args()
    print(args.__dict__)
    main()

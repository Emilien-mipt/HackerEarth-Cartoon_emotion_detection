from imageai.Detection.Custom import CustomObjectDetection


def load_detector(path_to_model: str, path_to_config: str):
    print("Loading the face detector...")
    detector = CustomObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(path_to_model)
    detector.setJsonPath(path_to_config)
    detector.loadModel()
    print("The detector has been loaded!")
    return detector


def warp(detector, detection_probability, input_path, output_path):
    detections, extracted_objects_array = detector.detectObjectsFromImage(
        input_image=input_path,
        output_image_path=output_path,
        minimum_percentage_probability=detection_probability,
        extract_detected_objects=True,
    )
    return (detections, extracted_objects_array)

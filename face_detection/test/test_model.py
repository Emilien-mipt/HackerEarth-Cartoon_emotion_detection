from imageai.Detection.Custom import CustomObjectDetection

detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("../detection_model-ex-045--loss-0003.172.h5")
detector.setJsonPath("../detection_config.json")
detector.loadModel()
detections, extracted_objects_array = detector.detectObjectsFromImage(
    input_image="test_image.jpg",
    output_image_path="test_image-detected.jpg",
    minimum_percentage_probability=80,
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
    print("---------------")

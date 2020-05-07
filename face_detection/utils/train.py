from imageai.Detection.Custom import DetectionModelTrainer

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory="../data/dataset/")
trainer.setGpuUsage("0")
trainer.setTrainConfig(
    object_names_array=["Tom", "Jerry"],
    batch_size=2,
    num_experiments=100,
    train_from_pretrained_model="../models/pretrained-yolov3.h5",
)
trainer.trainModel()

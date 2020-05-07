import os
import random
import shutil

images_path = "../data/images/"
labels_path = "../data/annotations/"
train_path = "../data/dataset/train/"
validation_path = "../data/dataset/validation/"


for image_file in os.listdir(images_path):
    labels_file = image_file.replace(".jpg", ".xml")
    if random.uniform(0, 1) > 0.2:
        shutil.copy(images_path + image_file, train_path + "images/" + image_file)
        shutil.copy(
            labels_path + labels_file, train_path + "annotations/" + labels_file
        )
    else:
        shutil.copy(images_path + image_file, validation_path + "images/" + image_file)
        shutil.copy(
            labels_path + labels_file, validation_path + "annotations/" + labels_file
        )

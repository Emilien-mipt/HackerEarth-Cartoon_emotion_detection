import os
import random
import shutil

train_path = "./data/train_frames/"
validation_path = "./data/val_frames/"

for frame in os.listdir(train_path):
    if random.uniform(0, 1) < 0.2:
        shutil.move(train_path + frame, validation_path + frame)

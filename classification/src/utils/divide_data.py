import os
import random
import shutil

TRAIN_PATH = "../../data/train_frames/"
VAL_PATH = "../../data/val_frames/"

for frame in os.listdir(TRAIN_PATH):
    if random.uniform(0, 1) < 0.2:
        shutil.move(os.path.join(TRAIN_PATH, frame), os.path.join(VAL_PATH, frame))

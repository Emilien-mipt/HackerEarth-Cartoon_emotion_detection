# Detect emotions of your favorite toons
## Problem statement

The International Day of Happiness is celebrated globally on March 20 every year with an objective to promote happiness and well-being as a fundamental human right for all. On this International Day of Happiness, we are bringing back all the joy and happiness for you. This fun challenge requires you to build a model that detects emotions of your favorite characters of the iconic toon show—Tom and Jerry.
Task

You are required to extract frames from a video clip that is provided in the dataset and classify the primary character’s emotion into one of the following five classes:
* Angry
* Happy
* Sad
* Surprised
* Unknown

### Note
The character prioritization is Tom > Jerry > Others. In case a frame does not contain Tom or Jerry’s faces for emotion detection, it is classified as ‘Unknown’,

## Data

### Data files

The dataset contains two video files and two .csv files. These files perform the following tasks:
Train Tom and jerry.mp4: Video file used for training
Train.csv: Contains human-generated labels for 298 frames [FrameRate = 5] from the training video
Test Tom and jerry.mp4: Video file to detect emotions on the face
Test.csv: Contains 186 frames [FrameRate = 5] from the test video in .csv format.

## Solution
The solution is based on the idea, that training and emotion classification should be done on warped faces of Tom or Jerry. Therefore it consists of 2 parts: face detection of Tom & Jerry and emotion classification.

### Face detection
Since there are not any available face detectors for cartoon characters, custom face detector was trained. The data was labeled by LabelImg package and the detector was trained using ImageAI library. In order to train the detector just go to `face_detection/utils` and run `python train.py`.

### Emotion classification
In order to classify the emotions, another network was trained. The approach is based on transfer learning: the pretrained net (in our case ResNet18, but it can be changed) is trained on prepared data. In order to train the classifier network, go to `classification/src` and run `python run_trainer` with corresponding parameters. The weights and models are saved in `classification/checkpoints/` and plots with losses are saved in `classification/logs/`.

As for the prediction and creating the submit file the pipeline goes as follows:
1) The loop goes over images in the test set. Every image goes to the input of the face detector model;
2) If no faces are detected, the class is being set as "Unknown". In opposite scenario the image with warped face is sent to the emotion classifier
3) The output of emotion classifier is written to the corresponding table in the submit file. Once the loop is over, the file is ready.

In order to make prediction and create submit file run `python predict.py` with parameters from `classification/src` folder.

# Detect emotions of your favorite toons
## Problem statement

![image_detect-emotions-tom-jerry-cartoon](https://user-images.githubusercontent.com/44554040/135051244-4eff8dab-a0c5-46ae-980b-c6b10f22a77d.png)

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

The dataset consists of two video files and two .csv files. These files perform the following tasks:
* Train Tom and jerry.mp4: Video file to train the models
* Train.csv: Contains human-generated labels for 298 frames [FrameRate = 5] from the training video
* Test Tom and jerry.mp4: Video file to detect emotions on the face
* Test.csv: Contains 186 frames [FrameRate = 5] from the test video in .csv format.

### Adding additional data to the train set
Since the data provided by the organizers is obviously not enough for training the face detector described below, additional data was gathered by downloading the cartoons from YouTube and running the command `extract_frames.py` to extract the frames.

## Solution
The solution is based on the idea, that training and emotion classification should be done on cropped faces of Tom or Jerry. Therefore it consists of 2 parts: face detection of Tom & Jerry and emotion classification.

### Face detection
Since there are not any available face detectors for cartoon characters, custom face detector was trained. The face detector is based on the YOLOv3 object detector. The data was labeled by LabelImg package and the detector was trained using ImageAI library. In order to train the detector just go to `face_detection/utils` and run `python train.py`. 

![Jerry-00000](https://user-images.githubusercontent.com/44554040/135054466-54d3dada-1864-4aea-90e3-689ec80f57c7.jpg)

*Example of the trained face detector output: Jerry's cropped face with label "Happy"*

![Tom-00000](https://user-images.githubusercontent.com/44554040/135055197-abb15ff7-fa60-486b-b83a-4c9ca0a9f7bf.jpg)

*Tom's cropped face with label "Sad"*

### Emotion classification
In order to classify the emotions, another network was trained using Pytorch framework. The approach is based on transfer learning: the pretrained net (in our case ResNet18, but it can be changed) is trained on prepared data. In order to train the classifier network, go to `classification/src` and run `python run_trainer` with corresponding parameters. The weights and models are saved in `classification/checkpoints/` and plots with losses are saved in `classification/logs/`.

As for the prediction and creating the submit file the pipeline goes as follows:
1) The loop goes over images in the test set. Every image goes to the input of the face detector model;
2) If no faces are detected, the class is being set as "Unknown". In opposite scenario the image with warped face is sent to the emotion classifier
3) The output of emotion classifier is written to the corresponding table in the submit file. Once the loop is over, the file is ready.

In order to make prediction and create submit file run `python predict.py` with parameters from `classification/src` folder.

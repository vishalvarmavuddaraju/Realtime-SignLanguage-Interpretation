# Real-Time Sign Language Interpretation

This project focuses on real-time sign language interpretation using a Convolutional Neural Network (CNN) trained on the Sign Language ASL dataset. The goal is to build a deep learning model capable of recognizing hand gestures corresponding to alphabetic signs and providing real-time predictions. It is a sign language interpreter using live video feed from the camera. 

## Table of Contents

* [Overview](#Overview)
* [Features](#Features)
* [Project Structure](#Project-Structure)
* [Technologies and Tools](#Technologies-and-Tools)
* [Setup](#Setup)
* [Process](#Process)
* [Status](#Status)
* [Reference](#Reference)
* [Group](#Group)

## Overview

The Real-Time Sign Language Interpretation using CNN project aims to enable real-time recognition of sign language gestures using a deep learning model trained on the Sign Language ASL dataset. The system is designed to recognize hand gestures representing alphabets A-Y , Numbers 0-9 and convert them into textual output.

The model is optimized for accurate classification real-time processing, and scalability for assistive communication applications.

## Features

Our model was able to predict the all the numbers in the ASL with a prediction accuracy >95%.

Features that can be added:
* Increasing the vocabulary of our model
* Adding feedback mechanism to make the model more robust
* Adding more sign languages

## Project Structure

```
├── handhist_set.py    # For setting up hand histogram.
├── gesture_creation.py       # For creating and generating gesture data.
├── all_gestures_display.py      # For visualizing all the generated sign gestures.
├── images_augmentation.py         # For image augmentation by performing rotations.
├── preprocess_images.py           # For loading and preprocessing the images in the dataset
├── CNN_training.py       # logic that required for model training.
├── main.py                 # Main script used for real-time recognition
├── Install_Packages.txt     # File that contains all required packages for the application
├── README.md                # Project documentation
```

## Technologies and Tools

- Python 3 or Anaconda
- TensorFlow
- Libraries:
    - `h5py`
    - `numpy`
    - `scikit-learn`
    - `keras`
    - `opencv-python`
    - `pyttsx3`

## Setup

Use comand promt to setup environment by using install_packages.txt and install_packages_gpu.txt files. 

`python -m pip r install_packages.txt`

This will help you in installing all the libraries required for the project.

## Process

* Run `handhist_set.py` to set the hand histogram for creating gestures. 
* Once you get a good histogram, save it in the code folder, or you can use the histogram created by us that can be found [here](https://github.com/vishalvarmavuddaraju/Realtime-SignLanguage-Interpretation/tree/main/code).
* Added gestures and label them using OpenCV which uses webcam feed. by running ` gesture_creation.py` and stores them in a database. Alternately, you can use the gestures created by us [here](https://github.com/vishalvarmavuddaraju/Realtime-SignLanguage-Interpretation/tree/main/code).
* Add different variations to the captured gestures by flipping all the images by using `images_augmentation.py`.
* Run `preprocess_images.py` to split all the captured gestures into training, validation and test set. 
* To view all the gestures, run `all_gestures_display.py` .
* Train the model using Keras by running `CNN_training.py`.
* Run `main.py`. This will open up the gesture recognition window which will use your webcam to interpret the trained American Sign Language gestures.  
* This model currently only detects the numbers from 0-9 and letters A-C
* Press `s` to save hand histogram
* Press `c` to start capturing gestures
* Press `q` to exit the application

## Status

* Completed the Model training with more than 95% test accuracy
* Working to expand the model by add more gestures
* working on model so that it can interpret under any light conditions

## Reference

https://youtu.be/NBzqY9tJd7M?feature=shared
We have taken this model as reference and used chatgpt and deepseek for development of model

## Group
Vishal Varma Vuddaraju UID: U39828798
Rasmitha Chinthalapally UID: U57992748
Srinija Reddy Maddula UID: U20959745

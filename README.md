# Rock-Paper-Scissors game

The project is in the context of my postgraduate program(Web and Data science) based on the famous game rock-paper-scissors. The aim of this project is to create a game app that two agents will play against each other. Our agent is trying to detect an image with noise which have been picked by the another agent and then tries to pick the right move in order to win the round. The model is based on Convolutional Neural Networks(CNN) which is a type of Deep Learning neural network architecture commonly used in Computer Vision.

## Table of Contents
- [Overview](#overview)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
- [Training the Neural Network](#training-the-neural-network)
- [Model Evaluation](#model-evaluation)
- [Results](#results)
- [Project Structure](#project-structure)

## Overview

The dataset that our model is trained contains 2188 images of hand gestures of each class (rock-paper-scissors). The dataset is imported as zip file and then you can extract it into different folders depending on the project requirements. The images in the dataset are 300x200 pixel size in PNG format. You can download it directly from the [Kaggle](https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors).

## Getting Started

The app is written on google colab notebook or you can choose any other notebook. The libraries that need to imported are:

- NumPy
- Matplotlib
- Scipy
- Splitfolders
- PIL
- Sklearn
- Random
- Cv2
- Zipfile, Os

And for deep learning libraries:
- Tensorflow
- Keras

## Training the Neural Network

To train the model it has been used the ImageDataGenerator class from the Keras library to apply various image augmentation techniques on the training dataset, such as rotation, horizontal and vertical flips, and shear transformations. The pixel values are also rescaled to be between 0 and 1. The generators are then set up to flow batches of preprocessed images from the specified directories during model training and validation. The images are resized to 100x100 pixels, and the color mode is set to grayscale. The labels are encoded in categorical format, indicating a multi-class classification task. The dataset is split into training and validation subsets, with 50% of the data allocated to each. Finally, the data generators are set to shuffle the data during training for better model generalization.

## Model Evaluation

[Explain how users can evaluate the performance of the trained model. Include information about metrics used, testing datasets, and any other relevant details.]

## Results

[Share the results of your project. This may include performance metrics, visualizations, or any other relevant output.]

## Project Structure

[Provide an overview of the project structure. Explain the purpose of key directories and files.]


## Acknowledgments

[Give credit to individuals or organizations that contributed to your project or inspired your work. Include links to relevant resources.]



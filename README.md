# Rock-Paper-Scissors game

The project is in the context of my postgraduate program(Web and Data science) based on the famous game rock-paper-scissors. The aim of this project is to create a game app that two agents will play against each other. Our agent is trying to detect an image with noise which have been picked by a random agent and then tries to pick the right move in order to win the round. The model is based on Convolutional Neural Networks(CNN) which is a type of Deep Learning neural network architecture commonly used in Computer Vision. The reason of choosing CNN model is because they can process large amounts of dataand produce highly accurate predictions.

## Table of Contents
- [Overview](#overview)
- [Getting Started](#getting-started)
- [CNN architecture](#cnn-architecture)
- [Training the CNN model](#training-the-cnn-model)
- [Model Evaluation](#model-evaluation)
- [Results](#results)

## Overview

The dataset that our model is trained contains 2188 images of hand gestures of each class (rock-paper-scissors). The dataset is imported as zip file and then you can extract it into different folders depending on the project requirements. The images in the dataset are 300x200 pixel size in PNG format. You can download it directly from the [Kaggle](https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors).

## Getting Started

The app is written on google colab notebook or you can choose any other notebook. The libraries that need to be imported are:

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

## CNN architecture 

The model is structured as a sequential stack of layers designed to extract hierarchical features from input images and make predictions. The architecture is the following:

**Input Layer**: The model begins with a Conv2D Layer with 16 filters, a kernel size of 3x3, and the ReLU activation function. The input shape is set to (100, 100, 1), indicating grayscale images with dimensions 100x100 pixels.

**MaxPooling Layer(Downsampling)**: A MaxPooling2D layer follows each Conv2D layer, reducing the spatial dimensions of the feature maps by a factor of 2 (2x2 pooling window).

**Dropout Layers**: After each MaxPooling layer, a Dropout layer is introduced with a dropout rate of 0.25. Dropout helps prevent overfitting by randomly "droping out" a fraction of input units during training.

**Stacked Convolutional Layers**: The model further stacks Conv2D, MaxPooling, and Dropout layers, gradually increasing the number of filters to capture more complex features. This hierarchy allows the model to learn patterns of varying abstraction.

**Flatten Layer**: Following the convolutional layers, a Flatten layer is added to convert the 3D feature maps into a 1D vector, preparing the data for the densely connected layers.

**Densely Connected Layers**: Two Dense (fully connected) layers are included with 512 units and ReLU activation. These layers act as classifiers, capturing global patterns and relationships among the learned features.

**Output Layer**: The final Dense layer has 3 units (equal to the number of classes) and uses the softmax activation function. This layer outputs probabilities for each class, representing the likelihood of the input image belonging to 'Rock,' 'Paper,' or 'Scissors.'

Model summary: 
- The model consists of a total of 12 layers.
- It has approximately 1.5 million parameters, including weights and biases.
- The summary provides insights into the size and structure of each layer, aiding in model inspection and debugging.

## Training the CNN model

To train the model it has been used the ImageDataGenerator class from the Keras library to apply various image augmentation techniques on the training dataset, such as rotation, horizontal and vertical flips, and shear transformations. The pixel values are also rescaled to be between 0 and 1. The generators are then set up to flow batches of preprocessed images from the specified directories during model training and validation. The images are resized to 100x100 pixels, and the color mode is set to grayscale. The labels are encoded in categorical format, indicating a multi-class classification task. The dataset is split into training and validation subsets, with 50% of the data allocated to each. Finally, the data generators are set to shuffle the data during training for better model generalization.

## Model Evaluation

To evaluate the performance of the cnn model takes place with two plots an accuracy plot and a loss plot. The first subplot, the accuracy one displays the training and validation accuract over epochs. The red line represent the training accuracy, while the blue line represents its validation accuracy. Trends in accuracy curves indicating how well the model is learning and generalizing. The second subplot exhibits the training and validation loss over epochs. The red line corresponds to the training loss, and the blue line to the validation loss. The loss curves insights into how effectively the model is minimizing errors during training and validation.

Except plots, the notebook represents some other evaluation metrics which are confusion matrix and classification report. The confusion matrix is a table that represents the model's performance by comparing predicted class labels against true class labels from the validation dataset. So, the confusion matrix provides a detailed berakdown of correct and incorrect predictions for each class (rock-paper-scissors). The classification report on the other hand generates and prints a classification report, offering a comprehensive summary of the model's precision, F1-score, and support for each class.

## Results

The plot after running the application for 10 rounds shows us that the agent predicted correctly 5 times the right action to win the round over the random agent. Also at the end of the notebook, the model is trying to recognize an image that is not part of the initial dataset and the model predicted it correctly. Surely there is a plenty room for improvements and even better predictions in the future. 



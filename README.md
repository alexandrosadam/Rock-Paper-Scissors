# Rock-Paper-Scissors
This project is for academic purpose and is based on the famous game rock-paper-scissors. Our model which is a cnn model is trying to detect an image in order to recognize it and choose the right move to win the round.

## The dataset contains 2188 images of hand gestures of each class (rock-paper-scissors). The dataset is imported as zip file and then you can extract it into different folders depending on the project requirements. The images in the dataset are 300x200 pixel size in PNG format. You can download it directly from the [Kaggle](https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors). So the next after importing successfully the dataset is to plot some random images from the dataset to see the different gestures and understand in more detail the data.

## The model that is trained to detect the images is a Convolutional Neural Network(CNN) model. The CNN is a category of machine learning model, a deep learning algorithm that is a proper choice for processing visual data such as images. The model is trained on the above dataset with the addition of some extra noise with the the help of ImageDataGenarator function from tensorflow library. The purpose of this is the model to be trained on more images that have been changed and be well trained to face and detect whatever it will come up.





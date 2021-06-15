<!-- [//]: # (Image References)

[image1]: ./images/sample_dog_output.png "Sample Output"
[image2]: ./images/vgg16_model.png "VGG-16 Model Layers"
[image3]: ./images/vgg16_model_draw.png "VGG16 Model Figure" -->

# DOG BREED CLASSIFIER

This project is the final project for the Machine Learning Nanodegree. In this project, the idea is to construct a dog breed classifier that can with certain accuracy tell the breed of a dog given a picture. 

![Sample Output][/images/sample_dog_output.png]

First, we have to develop a classifier (using OpenCV) to identify faces in the pictures. Similar, a pre-trained network (VGG 16) is used to determine if a dog is present in the picture. 

Finally, the last part of the notebook focuses in the creation of a Convolutional Neural Network from scratch an one via Transfer Learning (to use a pretrained model but adapt it to our desired goal)

You might find interesting [this](https://cs231n.github.io/convolutional-networks/#conv) course from Stanford to have a familiarity with computer vision and Convolutional Neural Networks.

## SET UP

To be able to run this notebook, you will need the following packages from python:

- os
- pandas
- numpy 
- PIL
- matplotlib
- seaborn
- pytorch
- glob
- openCV

all of this can be easily install using conda and 'conda install <name>'

## DATA
Dogs Data can be found in the [link](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip)

Humans Data can be found in the [link](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip)

## MODELS

Model used for human detection: [Haar feature-based cascade classifiers.](https://docs.opencv.org/master/db/d28/tutorial_cascade_classifier.html)
Model used for dog detection: [VGG16.](https://neurohive.io/en/popular-networks/vgg16/)
Model used for transfer learning: [Resnet50](https://www.mathworks.com/help/deeplearning/ref/resnet50.html#:~:text=ResNet%2D50%20is%20a%20convolutional,%2C%20pencil%2C%20and%20many%20animals.)

## ENJOY

I tried to make it as clear as possible, but if it happens, please, do not hesitate to contact me.


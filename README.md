# Doodle Recognition
## Introduction
In this, the model is fed hand-drawn doodles, scanned or created in front of the camera by the user. The model uses various machine learning and deep learning concepts to classify the doodle and provides results on the screen and in audible form. The main objective of this model is to create and recognize doodle images as well as use the given results for various applications. This model helps search for names of various objects based on the doodles. It can also be used by the hearing and speech impaired people to convey their immediate requirements to those around them so that they can be understood and assisted. It can be used for password protection i.e., doodle-based password system. It can serve as the fundamental model for sketch-based modelling and sketch-based image retrieval, among other things. 
## Problem Statement
The aim of the proposed work is to design and develop a framework to create and classify the doodle images into correct category.
## Motivation
* Quick, Draw! was where we first learned about doodle recognition.
* Quick, Draw! is an online game built with machine learning developed by Google Creative Lab and Data Arts Team. 
* It prompts the player to doodle an image in a certain category, and while the player is drawing, the model guesses what the image depicts in a human-to-computer game of Pictionary.
* Later, we considered creating our own hand drawn doodles. After extensive research, we discovered there are two methods to accomplish it: scan a drawn doodle or make a doodle in front of the camera in air.
* Then we tried to cover as many applications as possible by adding extensions like audio results, password protection, etc.
## Objectives
The main objectives of this model are:
* Create doodle images.
* Classify or recognize doodle images.
## Architecture
1. Application will provide us with two input options. 
   1. First, doodles can be created on paper and scanned so that the application can recognize them. 
   2. Second, it can be obtained by drawing in the air in front of the camera.
2. Following the input mode selection, the application will create the doodle image and classify the doodle.
3. The result will be displayed.
## Model Comparision
We trained the models on 10 classes: Alarm clock, Apple, Birthday cake, Butterfly, Candle, Ceiling fan, Donut, Door, Eyeglasses and T-shirt with 3000 images in each class. The results were as follows:
|Sl. No.| Model| Train Accuracy| Test Accuracy|
| --- | --- | --- | --- |
|1. | KNN | 85.15| 82.17|
|2. |SVM | 87.36| 84.58|
|3. | SVM with Kmeans| 88.11| 86.09|
|4. | CNN | 99.37| 97.20|
|5. | Resnet | 100| 96.78|
## Conclusion
We trained the models on 10 classes with 3000 images each and CNN showed the best results on test data within minimal time. Hence, CNN was opted. We implemented the scan hand drawn doodle images and the create doodle in air input modes on OpenCV. We used Graphical User Interface to implement the application on an user interface. Further, it can be used to implement sketch-based modelling and sketch-based image retrieval. 

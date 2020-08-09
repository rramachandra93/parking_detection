
**Parking Detection**
=================

**Problem Scenerio**
----------------

From  the video provided, it is required to identify vacant parking spaces. The car parked on the road should not be considered as parking space. After finding out all parking space in video frame , mark occupied space as red and free space as green.

**Feature Extraction**
------------------

For detecting weather a slot is empty or not, we capture each slot from videoframe seperately using opencv library then convert them into grayscale, apply gaussian blur , calculate histogram and extract features using canny edge detection. And store features of each image in an array.

**Training and prediction**
-----------------------

For predicting purpose we trained an svm model with images of empty slots and parked slots with label 0 and 1, and saved the model after getting good accuracy. Then for predicting we give the extracted features from each image as input to the model and the model detect weather it is occupied or not. According to the prediction, with the help of an if else statement we draw rectangles with color green to empty slots and red to occupied slots using opencv library

**SVM**
---

The support vector machine is a model used for both classification and regression problems though it is mostly used to solve classification problems. The algorithm creates a hyperplane or line(decision boundary) which separates data into classes. It uses the kernel trick to find the best line separator (decision boundary that has same distance from the boundary point of both classes). It is a clear and more powerful way of learning complex non linear functions.

**Results**
-------

1. The detection phase - perfect
2. Detection phase – Accuracy 60 % 
3. Marking slots with different colors – perfect

Accuracy is calculated using equation,  Accuracy  =  (True Positive + True Negative) / (True Positive + True Negative + False Positive + False Negative)

We can improve the performance of model by using more diverse dataset with sufficient amount of images
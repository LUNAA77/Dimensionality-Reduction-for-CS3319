# Assignment 1: "Dimensionality Reduction" for CS3319: Principles of Data Science
1. Download Animals with Attributes (AwA2) dataset from https://cvml.ist.ac.at/AwA2/. This 
dataset consists of 37322 images of 50 animal classes with pre-extracted deep learning 
features for each image. Split the images in each category into 60% for training and 40% for 
testing. You can use K-fold cross-validation within the training set to determine hyper-parameters, such as C in SVM.

2. Use linear SVM for image classification based on the deep learning features.
   
3. Reduce the dimensionality of deep learning features using three methods (one feature 
selection method, one feature projection method, one feature learning method) and perform
image classification again based on the obtained low-dimensional features. Record the 
performance variance w.r.t. different feature dimensionality.

4. Explore the optimal dimensionality reduction method and the optimal dimensionality.
   
5. Summarize your experimental results and write a project report in English. The project report 
should contain experimental setting (i.e., dataset, feature, training/testing split), the 
dimensionality reduction methods you tried, the experimental results you obtained, and the
experimental observations based on your experimental results.
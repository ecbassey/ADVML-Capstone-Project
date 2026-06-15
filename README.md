# ADVML-Capstone-Project
Tea Leaf Disease Detection Using Deep Learning

## Project Overview
The project involves building a **multiclass image classification model** using Convolutional Neural Networks (CNNs). The model will learn visual patterns from tea leaf images and classify them into one of five disease categories. The final system should be capable of accepting a tea leaf image and predicting its disease category with high accuracy.

## Problem Statement
Tea farming is an important agricultural activity, but plant diseases can significantly reduce crop quality and yield. Traditionally, disease identification is performed manually by agricultural experts, which is time-consuming, costly, and prone to human error.

This project aims to develop a machine learning-based image classification system capable of automatically identifying the health status of tea leaves from images. The model will classify a tea leaf image into one of five categories:

- Healthy
- Algal Disease 
- Brown Blight Disease
- Red Leaf Disease
- White Spot Disease 

By automating disease detection, farmers can receive early warnings and take corrective action to minimize crop losses.

### Goal

- Build a CNN-based multiclass classification model.
- Compare different deep learning architectures.
- Evaluate model performance using standard metrics.
- Identify the best-performing model.
- Develop a prediction pipeline for classifying new tea leaf images.


### Dataset Structure
Tea_Leaf_Dataset/

├── Healthy/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...

├── Disease_1/
│   ├── img1.jpg
│   └── ...

├── Disease_2/
│   ├── img1.jpg
│   └── ...

├── Disease_3/
│   ├── img1.jpg
│   └── ...

└── Disease_4/
    ├── img1.jpg
    └── ...

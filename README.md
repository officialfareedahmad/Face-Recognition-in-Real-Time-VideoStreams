# Face Recognition in Real-Time Video Streams

## Overview

This project demonstrates a **real-time face recognition application** utilizing Python, Keras, OpenCV, and TensorFlow. The application accurately identifies pre-trained faces in live video streams and assigns labels to them based on a deep learning classification model. Designed for high performance, the system leverages cutting-edge libraries and tools to provide accurate and efficient real-time recognition.

## Features

- **Face Detection**: Implements OpenCV's Haar Cascade Classifier for efficient and accurate face localization in video frames.
- **Deep Learning Integration**: Utilizes a pre-trained TensorFlow/Keras model (`keras_model.h5`) to classify and identify faces.
- **Real-Time Performance**: Processes live video input from a webcam and delivers instant results with probability values for classification.
- **Dynamic Labeling**: Assigns names to recognized faces, demonstrating multi-class classification.

## Technologies Used

- **Python**: The core programming language for implementation.
- **OpenCV**: For video stream handling and face detection.
- **Keras and TensorFlow**: Frameworks for building and running the deep learning model.
- **Numpy**: Utilized for numerical operations and image array preprocessing.

## Project Architecture

1. **Face Detection**:  
   The application uses the Haar Cascade Classifier (`haarcascade_frontalface_default.xml`) to detect facial regions in the input frames.

2. **Preprocessing**:  
   Detected faces are cropped, resized to 224x224 pixels, and normalized before being passed into the model.

3. **Model Prediction**:  
   The TensorFlow/Keras model classifies the cropped image into pre-trained categories and outputs the class index and probability score.

4. **Real-Time Display**:  
   Each recognized face is labeled on the video stream, along with the recognition confidence percentage.

## Applications

- **Authentication Systems**: Can be adapted for secure, contactless login systems.
- **Security Surveillance**: Useful for identifying individuals in real-time in surveillance footage.
- **Smart Homes**: Can integrate into home automation systems for personalized services.


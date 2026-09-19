# Sign Language Detector

An AI-based desktop application that detects and recognizes predefined sign language gestures using a webcam.

## Overview

The Sign Language Detector uses computer vision and machine learning to recognize hand gestures in real time.

The system captures video from a webcam, detects the user's hand, preprocesses the detected region, and passes it to a trained machine learning model for classification.

## Supported Gestures

- Hello
- I Love You
- Thank You
- Yes

## Technologies Used

- Python
- OpenCV
- CVZone
- MediaPipe
- NumPy
- TensorFlow / Keras

## System Workflow

Webcam Input  
↓  
Hand Detection  
↓  
Hand Cropping  
↓  
Image Preprocessing  
↓  
Machine Learning Classification  
↓  
Gesture Prediction  
↓  
Text Output

## Project Structure

```text
Model/
    keras_model.h5
    labels.txt

Data/
    Hello/
    I Love You/
    Thank You/
    Yes/

datacollection.py
test.py
requirements.txt

# ASL Hand Sign Detector (A–E)

A Python project that builds a real-time American Sign Language (ASL) alphabet classifier for letters A through E using a convolutional neural network (CNN). The model is trained on a subset of the ASL alphabet dataset and leverages OpenCV to capture live webcam video and predict hand signs in real time.

## Features

- Custom CNN built with PyTorch for image classification
- Preprocessing and data augmentation with torchvision transforms
- Live webcam capture and real-time prediction with OpenCV
- Dataset subset focused on ASL letters A to E for fast training and inference
- Easy-to-run training and detection scripts

## Technologies

- Python 3.x
- PyTorch
- OpenCV
- torchvision

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/Suhas2Anumolu/asl-hand-sign-detector.git
cd asl-hand-sign-detector

Download the ASL alphabet dataset from Kaggle: ASL Alphabet
Extract only the folders A, B, C, D, and E into the dataset/ directory inside the project folder.

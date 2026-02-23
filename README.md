# deepfake-and-real-video-differentiation
# Deepfake Video Detection Project

## Overview
This project aims to build a **Deepfake Detection** system using **computer vision** techniques. The system analyzes video files and detects whether the video is **real** or **fake**. The model was trained using a dataset of real and fake videos, with frame extraction and processing used to feed the data into the neural network.

## Architecture
The architecture used in this project is based on a **ResNet-18 model**, modified for **video classification**. The model uses **3D CNNs (Convolutional Neural Networks)** to analyze temporal and spatial patterns in videos, effectively distinguishing between real and fake videos.

- **Model**: 3D ResNet-18 with pretrained weights from the `ResNet` model for feature extraction.
-**Dataset**: Uses a combination of Celeb DF and FaceForensis++ model for dataset for training.
- **Input**: The model takes a sequence of frames (video clips) as input.
- **Output**: It outputs a classification label (Real or Fake) with a confidence score.

## Dataset and Model Weights
The dataset and model weights are too large to be hosted on GitHub. You can download them from Google Drive:

- **https://drive.google.com/drive/folders/1nhyOuWnRcuB_H00eREH-4CDdNje_CLy2?usp=drive_link**

Make sure to download both the **dataset** (for training) and the **model weights** (`best_model.pt`).


---
title: Emotion Recognition Application
emoji: 😃
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 5.29.1
app_file: app.py
pinned: false
license: mit
safe_serialization: true
---

# Emotion Recognition Application

This application identifies emotions (Angry, Happy, Neutral, Sad, Surprise) in uploaded facial images.

## Features

- Image upload and emotion recognition
- 5 different emotion classes: Angry, Happy, Neutral, Sad, Surprise
- Model trained on ResNet34 architecture and saved in SafeTensors format

## Installation

To install the required libraries:

```bash
pip install -r requirements.txt
```

## Usage

To start the application:

```bash
python app.py
```

In the browser interface:

1. Upload a face image in the "Upload Image" section
2. Click the "Predict" button
3. The result will show the predicted emotion and confidence levels

## About the Model

The application uses a emotion classification model trained with transfer learning on ResNet34 architecture using PyTorch. The model processes 48x48 pixel grayscale facial images to predict emotions. Model weights are stored in the efficient and secure safetensors format.

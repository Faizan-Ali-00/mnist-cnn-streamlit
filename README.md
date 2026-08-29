# MNIST CNN Handwritten Digit Recognizer

A handwritten digit recognition web application built with a Convolutional Neural Network (CNN) and Streamlit.

## Features

- Recognizes handwritten digits from 0 to 9
- Upload PNG, JPG, and JPEG images
- Automatically preprocesses the image
- Displays the predicted digit
- Displays prediction confidence
- Shows prediction probabilities

## Technologies

- Python
- TensorFlow
- Keras
- NumPy
- Pillow
- Streamlit
- MNIST Dataset

## CNN Architecture

The model uses:

- Conv2D
- MaxPooling2D
- Conv2D
- MaxPooling2D
- Flatten
- Dense
- Softmax

## How It Works

Upload Image → Preprocessing → 28×28 Image → CNN → Prediction → Confidence

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py

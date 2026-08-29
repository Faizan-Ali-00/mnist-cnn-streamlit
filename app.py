import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np


# Page configuration
st.set_page_config(
    page_title="MNIST Digit Recognizer",
    page_icon="🔢",
    layout="centered"
)


# Title
st.title("🔢 MNIST Handwritten Digit Recognizer")

st.write(
    "Upload an image of a handwritten digit from 0 to 9."
)


# Show TensorFlow version
st.write(
    "TensorFlow version:",
    tf.__version__
)


# Load trained CNN model
model = tf.keras.models.load_model(
    "mnist_model.keras",
    compile=False
)


# Upload image
uploaded_file = st.file_uploader(
    "Upload a handwritten digit",
    type=["png", "jpg", "jpeg"]
)


if uploaded_file is not None:

    # Open image
    image = Image.open(
        uploaded_file
    ).convert("L")


    # Display original image
    st.subheader("Original Image")

    st.image(
        image,
        width=300
    )


    # Convert image to NumPy array
    img = np.array(image)


    # Invert colors
    img = 255 - img


    # Convert image to black and white
    img[img < 100] = 0
    img[img >= 100] = 255


    # Find the area containing the digit
    rows = np.any(
        img > 0,
        axis=1
    )

    cols = np.any(
        img > 0,
        axis=0
    )


    # Crop the digit
    if np.any(rows) and np.any(cols):

        img = img[
            rows
        ][:, cols]


    # Convert to PIL image
    digit = Image.fromarray(
        img.astype(np.uint8)
    )


    # Resize while maintaining proportions
    digit.thumbnail(
        (20, 20)
    )


    # Create 28x28 black canvas
    canvas = Image.new(
        "L",
        (28, 28),
        0
    )


    # Center the digit
    x = (
        28 - digit.width
    ) // 2

    y = (
        28 - digit.height
    ) // 2


    canvas.paste(
        digit,
        (x, y)
    )


    # Display processed image
    st.subheader("Processed Image")

    st.image(
        canvas,
        width=200
    )


    # Convert processed image to NumPy
    final_image = np.array(
        canvas
    )


    # Normalize pixel values
    final_image = (
        final_image / 255.0
    )


    # Reshape for CNN
    final_image = final_image.reshape(
        1,
        28,
        28,
        1
    )


    # Make prediction
    prediction = model.predict(
        final_image,
        verbose=0
    )


    # Get predicted digit
    predicted_digit = np.argmax(
        prediction[0]
    )


    # Get confidence
    confidence = (
        np.max(prediction[0]) * 100
    )


    # Display prediction
    st.subheader("Prediction")

    st.success(
        f"Predicted Digit: {predicted_digit}"
    )


    st.info(
        f"Confidence: {confidence:.2f}%"
    )


    # Display probabilities
    st.subheader(
        "Prediction Probabilities"
    )


    probabilities = (
        prediction[0] * 100
    )


    chart_data = {
        "Digit": list(range(10)),
        "Probability": probabilities
    }


    st.bar_chart(
        chart_data,
        x="Digit",
        y="Probability"
    )

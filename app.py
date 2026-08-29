import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

model = tf.keras.models.load_model("mnist_cnn_model.keras")

st.title("MNIST Handwritten Digit Recognizer")
st.write("Upload a handwritten digit from 0 to 9.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("L")

    st.subheader("Original Image")
    st.image(image, width=300)

    img = np.array(image)

    # Invert colors
    img = 255 - img

    # Remove background
    img[img < 100] = 0
    img[img >= 100] = 255

    # Find digit
    rows = np.any(img > 0, axis=1)
    cols = np.any(img > 0, axis=0)

    if np.any(rows) and np.any(cols):
        img = img[rows][:, cols]

    digit = Image.fromarray(img.astype(np.uint8))

    # Resize
    digit.thumbnail((20, 20))

    # Create 28x28 canvas
    canvas = Image.new("L", (28, 28), 0)

    # Center digit
    x = (28 - digit.width) // 2
    y = (28 - digit.height) // 2

    canvas.paste(digit, (x, y))

    # Normalize
    final_image = np.array(canvas) / 255.0

    # Reshape
    final_image = final_image.reshape(1, 28, 28, 1)

    # Prediction
    prediction = model.predict(final_image, verbose=0)

    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    st.subheader("Prediction")
    st.success(f"Predicted Digit: {predicted_digit}")
    st.write(f"Confidence: {confidence:.2f}%")

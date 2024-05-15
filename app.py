import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from keras.models import load_model
import io

# Load the trained model
model_path = 'model/v3_red_cott_dis.h5'  # Update this path if necessary
model = load_model(model_path)

# Define the class names based on your dataset
class_names = ['Healthy', 'Disease Type 1', 'Disease Type 2', 'Disease Type 3']

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = image.resize((150, 150))  # Resize the image to match the input shape of the model
    image = np.array(image) / 255.0   # Normalize the image
    image = np.expand_dims(image, axis=0)  # Expand dimensions to match the model input
    return image

# Function to predict the class of the image
def predict(image):
    preprocessed_image = preprocess_image(image)
    predictions = model.predict(preprocessed_image)
    predicted_class = np.argmax(predictions, axis=1)
    confidence = np.max(predictions)
    return class_names[predicted_class[0]], confidence

# Streamlit app
st.title("Cotton Plant Disease Detection")

st.write("Upload an image of a cotton plant leaf to detect if it has any disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(io.BytesIO(uploaded_file.read()))
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    
    label, confidence = predict(image)
    st.write(f"Prediction: {label}")
    st.write(f"Confidence: {confidence:.2f}")

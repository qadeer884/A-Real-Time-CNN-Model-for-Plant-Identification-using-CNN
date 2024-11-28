import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Path to the saved model (replace with your actual model path)
model_path = "C:\\Skill\\self_learning\\plant_cnn\\my_model.h5"

# Load the model from the saved file
model = tf.keras.models.load_model(model_path)

# Define label_mapping (the classes from your dataset)
label_mapping = {
    'aloevera': 0, 'banana': 1, 'bilimbi': 2, 'cantaloupe': 3, 'cassava': 4, 'coconut': 5,
    'corn': 6, 'cucumber': 7, 'curcuma': 8, 'eggplant': 9, 'galangal': 10, 'ginger': 11,
    'guava': 12, 'kale': 13, 'longbeans': 14, 'mango': 15, 'melon': 16, 'orange': 17,
    'paddy': 18, 'papaya': 19, 'peper chili': 20, 'pineapple': 21, 'pomelo': 22, 'shallot': 23,
    'soybeans': 24, 'spinach': 25, 'sweet potatoes': 26, 'tobacco': 27, 'waterapple': 28, 'watermelon': 29
}

# Streamlit app layout
st.title("Plant Image Classifier")
st.write("Upload an image of a plant, and the model will predict the class.")

# Image upload
uploaded_image = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

# Check if an image has been uploaded
if uploaded_image is not None:
    # Load and preprocess the image
    image = Image.open(uploaded_image)
    image = image.convert('RGB')  # Ensure it's RGB
    image = image.resize((128, 128))  # Resize to model input size
    image_array = np.array(image)
    image_array = image_array.astype('float32') / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Make a prediction
    predicted_label = model.predict(image_array)
    predicted_label_idx = np.argmax(predicted_label, axis=1)[0]  # Get the index of the predicted class

    # Get the class name from the label mapping
    predicted_class_name = list(label_mapping.keys())[list(label_mapping.values()).index(predicted_label_idx)]

    # Display the uploaded image and prediction result
    st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)
    st.write(f"Predicted Class: {predicted_class_name}")

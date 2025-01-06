import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load the trained model
model = tf.keras.models.load_model(r"C:\Users\user\OneDrive\Desktop\Pneumonia_Disease\chest_xray\my_model.keras")

# Function to predict pneumonia
def predict_pneumonia(image_path, model):
    # Preprocess the image
    img = load_img(image_path, target_size=(224, 224))  # Resize to 224x224 as per model input
    img_array = img_to_array(img)  # Convert to array
    img_array = img_array / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(img_array)[0][0]

    # Interpret prediction
    if prediction >= 0.5:
        return "The X-ray indicates pneumonia.", prediction
    else:
        return "The X-ray is normal.", prediction

# Streamlit app UI
st.title("Pneumonia Detection from X-ray")
st.write("Upload a chest X-ray image to check if the person has pneumonia.")

# File uploader
uploaded_file = st.file_uploader("Choose an X-ray image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded X-ray", use_column_width=True)

    # Save the uploaded file temporarily
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Run prediction
    st.write("Analyzing the image...")
    result, confidence = predict_pneumonia("temp_image.jpg", model)
    st.write(result)
    st.write(f"Prediction confidence: {confidence:.2f}")


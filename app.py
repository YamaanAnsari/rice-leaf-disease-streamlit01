# ================== CREATE STREAMLIT APP ==================
app_code = '''
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
import os

st.set_page_config(page_title="Rice Leaf Disease Detection", layout="centered")

st.title("🌾 Rice Leaf Disease Detection App")
st.write("Upload a rice leaf image to predict the disease type using an AI model.")

# Load Model
model_path = "rice_leaf_model.h5"
model = tf.keras.models.load_model(model_path)

# Class Labels
class_labels = sorted(os.listdir("''' + dataset_path + '''"))
st.sidebar.title("Class Labels")
for c in class_labels:
    st.sidebar.markdown(f"- {c}")

# File Upload
uploaded_file = st.file_uploader("📤 Upload a rice leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = load_img(uploaded_file, target_size=(224, 224))
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success(f"✅ Predicted Disease: **{predicted_class}**")
    st.info(f"Confidence: {confidence:.2f}%")

st.markdown("---")
st.caption("Developed using Streamlit, TensorFlow & EfficientNetB3")
'''

# Write Streamlit app file
with open("app.py", "w") as f:
    f.write(app_code)

print("✅ Streamlit app (app.py) created successfully!")

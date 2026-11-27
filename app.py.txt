
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Skin Cancer Detection")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("skin_cancer_cnn.keras")

model = load_model()

st.title("Skin Cancer Detection")
st.write("Upload a skin image to get a model prediction.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)

    image = image.resize((128, 128))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    prediction = model.predict(image_array)[0][0]

    if prediction > 0.5:
        st.error("Prediction: Malignant")
        st.write("Confidence:", round(float(prediction) * 100, 2), "%")
    else:
        st.success("Prediction: Benign")
        st.write("Confidence:", round((1 - float(prediction)) * 100, 2), "%")

st.warning("Educational project only. This tool is not for medical diagnosis.")


import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
import os
from PIL import Image

st.set_page_config(page_title="Skin Cancer Detection")

MODEL_PATH = "skin_cancer_cnn.keras"
FILE_ID = "1yJuXTFukSvHc12By-MSx8l5b8ZPta2g0"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        gdown.download(
            id=FILE_ID,
            output=MODEL_PATH,
            quiet=False
        )

    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

st.title("Skin Cancer Detection")
st.write("Upload a skin image to get a prediction.")

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
        confidence = float(prediction) * 100
    else:
        st.success("Prediction: Benign")
        confidence = (1 - float(prediction)) * 100

    st.write("Model confidence:", round(confidence, 2), "%")

st.warning(
    "Educational project only. "
    "This model is not suitable for medical diagnosis."
)

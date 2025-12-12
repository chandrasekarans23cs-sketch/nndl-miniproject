import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from utils import preprocess_image, postprocess_output

# Load pretrained model
model = load_model("colorization_model.h5")

st.title("🎨 Black & White Image Colorization")
st.write("Upload a grayscale photo or choose a sample to see it colorized!")

uploaded_file = st.file_uploader("Upload a grayscale image", type=["jpg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    gray_img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    st.image(gray_img, caption="Original Grayscale", use_column_width=True)

    input_img = preprocess_image(gray_img)
    ab_channels = model.predict(input_img)
    colorized_img = postprocess_output(gray_img, ab_channels)

    st.image(colorized_img, caption="Colorized Output", use_column_width=True)

import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, UpSampling2D, Input

# --- Build CNN model directly ---
def build_colorization_model():
    input_layer = Input(shape=(256, 256, 1))
    x = Conv2D(64, (3,3), activation='relu', padding='same')(input_layer)
    x = Conv2D(128, (3,3), activation='relu', padding='same', strides=2)(x)
    x = Conv2D(256, (3,3), activation='relu', padding='same', strides=2)(x)
    x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = UpSampling2D((2,2))(x)
    x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = UpSampling2D((2,2))(x)
    output_layer = Conv2D(2, (3,3), activation='tanh', padding='same')(x)
    return Model(inputs=input_layer, outputs=output_layer)

model = build_colorization_model()

# --- Streamlit UI ---
st.title("🎨 Black & White Image Colorization")
st.write("Upload a grayscale photo and see the pipeline run!")

uploaded_file = st.file_uploader("Upload a grayscale image", type=["jpg", "png"])

def preprocess_image(img):
    img = cv2.resize(img, (256, 256))
    img = img / 255.0
    img = img.reshape(1, 256, 256, 1)
    return img

def postprocess_output(gray_img, ab_channels):
    gray_img = cv2.resize(gray_img, (256, 256))
    lab_img = np.zeros((256, 256, 3))
    lab_img[:,:,0] = gray_img
    lab_img[:,:,1:] = ab_channels[0] * 128
    colorized = cv2.cvtColor(lab_img.astype("uint8"), cv2.COLOR_LAB2BGR)
    return colorized

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    gray_img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    st.image(gray_img, caption="Original Grayscale", use_column_width=True)

    input_img = preprocess_image(gray_img)
    ab_channels = model.predict(input_img)   # random output if not trained
    colorized_img = postprocess_output(gray_img, ab_channels)

    st.image(colorized_img, caption="Colorized Output (demo)", use_column_width=True)

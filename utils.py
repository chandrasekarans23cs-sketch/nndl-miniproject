import cv2
import numpy as np

def preprocess_image(gray_img):
    gray_img = cv2.resize(gray_img, (256, 256))
    gray_img = gray_img / 255.0
    gray_img = gray_img.reshape(1, 256, 256, 1)
    return gray_img

def postprocess_output(gray_img, ab_channels):
    # Convert grayscale + predicted ab channels back to LAB
    lab_img = np.zeros((256, 256, 3))
    lab_img[:,:,0] = cv2.resize(gray_img, (256, 256))
    lab_img[:,:,1:] = ab_channels[0]

    # Convert LAB → BGR
    colorized = cv2.cvtColor(lab_img.astype("uint8"), cv2.COLOR_LAB2BGR)
    return colorized

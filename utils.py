import cv2
import numpy as np

def preprocess_image(img):
    img = cv2.resize(img, (256, 256))
    img = img / 255.0
    img = img.reshape(1, 256, 256, 1)
    return img

def postprocess_output(gray_img, ab_channels):
    gray_img = cv2.resize(gray_img, (256, 256))
    lab_img = np.zeros((256, 256, 3))
    lab_img[:,:,0] = gray_img
    lab_img[:,:,1:] = ab_channels[0] * 128  # scale back from [-1,1]

    colorized = cv2.cvtColor(lab_img.astype("uint8"), cv2.COLOR_LAB2BGR)
    return colorized

import numpy as np
import cv2
import os
from tensorflow.keras.optimizers import Adam
from model import build_colorization_model

# Load dataset (example: small set of images in 'samples/')
def load_images(path="samples/"):
    images = []
    for file in os.listdir(path):
        img = cv2.imread(os.path.join(path, file))
        if img is not None:
            img = cv2.resize(img, (256, 256))
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            images.append(lab)
    return np.array(images)

images = load_images()

# Split into L and ab channels
X = images[:,:,:,0] / 255.0
X = X.reshape(-1, 256, 256, 1)
Y = images[:,:,:,1:] / 128.0  # normalize to [-1,1]

# Build model
model = build_colorization_model()
model.compile(optimizer=Adam(1e-3), loss='mse')

# Train
model.fit(X, Y, epochs=10, batch_size=8)

# Save model
model.save("colorization_model.h5")

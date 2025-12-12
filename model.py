from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, UpSampling2D, Input

def build_colorization_model():
    input_layer = Input(shape=(256, 256, 1))

    # Encoder
    x = Conv2D(64, (3,3), activation='relu', padding='same')(input_layer)
    x = Conv2D(128, (3,3), activation='relu', padding='same', strides=2)(x)
    x = Conv2D(256, (3,3), activation='relu', padding='same', strides=2)(x)

    # Decoder
    x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = UpSampling2D((2,2))(x)
    x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = UpSampling2D((2,2))(x)
    output_layer = Conv2D(2, (3,3), activation='tanh', padding='same')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model

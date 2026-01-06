import tensorflow as tf
from tensorflow.keras import layers, Model
from src.config import PATCH_SIZE, BASE_FILTERS

def build_lightweight_unet(input_shape=(*PATCH_SIZE, 1)):
    """
    Builds a custom lightweight 3D U-Net (~1.4M params).
    """
    def conv_block(x, filters):
        x = layers.Conv3D(filters, 3, padding='same', activation='relu')(x)
        x = layers.Conv3D(filters, 3, padding='same', activation='relu')(x)
        return x

    def encoder_block(x, filters):
        c = conv_block(x, filters)
        p = layers.MaxPool3D((2, 2, 2))(c)
        return c, p

    def decoder_block(x, skip, filters):
        x = layers.Conv3DTranspose(filters, 2, strides=2, padding='same')(x)
        x = layers.Concatenate()([x, skip])
        x = conv_block(x, filters)
        return x

    # Inputs
    inputs = layers.Input(shape=input_shape)

    # Encoder
    c1, p1 = encoder_block(inputs, BASE_FILTERS)
    c2, p2 = encoder_block(p1, BASE_FILTERS * 2)
    c3, p3 = encoder_block(p2, BASE_FILTERS * 4)

    # Bottleneck
    b = conv_block(p3, BASE_FILTERS * 8)

    # Decoder
    d3 = decoder_block(b, c3, BASE_FILTERS * 4)
    d2 = decoder_block(d3, c2, BASE_FILTERS * 2)
    d1 = decoder_block(d2, c1, BASE_FILTERS)

    # Output
    outputs = layers.Conv3D(1, 1, activation='sigmoid')(d1)

    return Model(inputs, outputs, name="Lightweight_3D_UNet")
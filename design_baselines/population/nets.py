from tensorflow_probability import distributions as tfpd
import tensorflow.keras.layers as tfkl
import tensorflow as tf
import numpy as np

from tensorflow_addons.layers import SpectralNormalization


class ForwardModel(tf.keras.Sequential):
    def __init__(self, input_shape, hidden):
        layers = [
            tfkl.Flatten(input_shape=input_shape),
            tfkl.Dense(hidden, activation=tfkl.LeakyReLU()),
            tfkl.Dense(hidden, activation=tfkl.LeakyReLU()),
            tfkl.Dense(2),
            ]

        super(ForwardModel, self).__init__(layers)

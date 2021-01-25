from tensorflow_probability import distributions as tfpd
import tensorflow.keras.layers as tfkl
import tensorflow as tf
import numpy as np

from tensorflow_addons.layers import SpectralNormalization


class ForwardModel(tf.keras.Sequential):
    def __init__(
        self,
        input_shape,
        activations=("relu", "relu"),
        hidden=2048,
    ):
        super(ForwardModel, self).__init__()
        #self.add(SpectralNormalization(tfkl.Dense(hidden, activation="relu")))
        #self.add(SpectralNormalization(tfkl.Dense(hidden, activation="relu")))
        #self.add(SpectralNormalization(tfkl.Dense(1)))
        self.add(tfkl.Dense(hidden, activation="relu"))
        self.add(tfkl.Dense(hidden, activation="relu"))
        self.add(tfkl.Dense(1))
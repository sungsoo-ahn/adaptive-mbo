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
        activations = [
            tfkl.LeakyReLU
            if act == "leaky_relu"
            else tfkl.Activation(tf.math.cos)
            if act == "cos"
            else act
            for act in activations
        ]

        layers = [tfkl.Flatten(input_shape=input_shape)]
        #for act in activations:
        #    layers.extend(
        #        [SpectralNormalization(tfkl.Dense(hidden)), tfkl.Activation(act) if isinstance(act, str) else act()]
        #    )
        #layers.append(SpectralNormalization(tfkl.Dense(1)))
        for act in activations:
            layers.extend(
                [tfkl.Dense(hidden), tfkl.Activation(act) if isinstance(act, str) else act()]
            )
        layers.append(tfkl.Dense(1))

        super(ForwardModel, self).__init__(layers)

from tensorflow_probability import distributions as tfpd
import tensorflow.keras.layers as tfkl
import tensorflow as tf
import numpy as np
from collections import defaultdict
from tensorflow_addons.layers import SpectralNormalization


class ForwardModel(tf.keras.Sequential):
    distribution = tfpd.Normal
    def __init__(self, input_shape, hidden):
        self.input_shape_ = input_shape
        self.hidden = hidden

        self.max_logstd = tf.Variable(
            tf.fill([1, 1], np.log(0.2).astype(np.float32)), trainable=True
        )
        self.min_logstd = tf.Variable(
            tf.fill([1, 1], np.log(0.1).astype(np.float32)), trainable=True
        )

        layers = [
            tfkl.Flatten(input_shape=input_shape),
            tfkl.Dense(hidden, activation=tfkl.LeakyReLU()),
            tfkl.Dense(hidden, activation=tfkl.LeakyReLU()),
            tfkl.Dense(2),
        ]

        super(ForwardModel, self).__init__(layers)

    def get_params(self, inputs, **kwargs):
        prediction = super(ForwardModel, self).__call__(inputs, **kwargs)
        mean, logstd = tf.split(prediction, 2, axis=-1)
        logstd = self.max_logstd - tf.nn.softplus(self.max_logstd - logstd)
        logstd = self.min_logstd + tf.nn.softplus(logstd - self.min_logstd)
        return {"loc": mean, "scale": tf.math.exp(logstd)}

    def get_distribution(self, inputs, **kwargs):
        return self.distribution(**self.get_params(inputs, **kwargs))
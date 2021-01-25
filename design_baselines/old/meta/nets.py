from tensorflow_probability import distributions as tfpd
import tensorflow.keras.layers as tfkl
import tensorflow as tf
import numpy as np
from tensorflow_addons.layers import SpectralNormalization

class ForwardModel(tf.keras.Sequential):
    """A Fully Connected Network with 2 trainable layers"""

    distribution = tfpd.Normal

    def __init__(
        self,
        input_shape,
        activations=("relu", "relu"),
        hidden=2048,
        initial_max_std=0.2,
        initial_min_std=0.1,
    ):
        self.max_logstd = tf.Variable(
            tf.fill([1, 1], np.log(initial_max_std).astype(np.float32)), trainable=True
        )
        self.min_logstd = tf.Variable(
            tf.fill([1, 1], np.log(initial_min_std).astype(np.float32)), trainable=True
        )

        layers = [tfkl.Flatten(input_shape=input_shape)]
        for act in activations:
            layer = tfkl.Dense(hidden, activation=tfkl.LeakyReLU())#tfkl.LeakyReLU())
            layers.append(layer)

        layer = tfkl.Dense(2)
        layers.append(layer)
        super(ForwardModel, self).__init__(layers)

    def get_params(self, inputs, **kwargs):
        prediction = super(ForwardModel, self).__call__(inputs, **kwargs)
        mean, logstd = tf.split(prediction, 2, axis=-1)
        logstd = self.max_logstd - tf.nn.softplus(self.max_logstd - logstd)
        logstd = self.min_logstd + tf.nn.softplus(logstd - self.min_logstd)
        return {"loc": mean, "scale": tf.math.exp(logstd)}

    def get_distribution(self, inputs, **kwargs):
        return self.distribution(**self.get_params(inputs, **kwargs))

    @property
    def inner_weights(self):
        return [
            self.layers[1].kernel,
            self.layers[1].bias,
            self.layers[2].kernel,
            self.layers[2].bias,
            self.layers[3].kernel,
            self.layers[3].bias,
            self.max_logstd,
            self.min_logstd,
            ]

    @classmethod
    def copy_from(cls, config, weights, grads=None, step_size=0.0, weight_decay_coef=0.0, clip_weights=None):
        new_model = cls(**config)
        if grads is None or step_size == 0.0:
            new_model.layers[1].kernel = weights[0]
            new_model.layers[1].bias = weights[1]
            new_model.layers[2].kernel = weights[2]
            new_model.layers[2].bias = weights[3]
            new_model.layers[3].kernel = weights[4]
            new_model.layers[3].bias = weights[5]
            new_model.max_logstd = weights[6]
            new_model.min_logstd = weights[7]

        else:
            new_grads = []
            for grad in grads:
                if grad is None:
                    new_grads.append(0.0)
                else:
                    new_grads.append(grad)

            grads = new_grads
            new_model.layers[1].kernel = weights[0] - step_size * grads[0]
            new_model.layers[1].bias = weights[1] - step_size * grads[1]
            new_model.layers[2].kernel = weights[2] - step_size * grads[2]
            new_model.layers[2].bias = weights[3] - step_size * grads[3]
            new_model.layers[3].kernel = weights[4] - step_size * grads[4]
            new_model.layers[3].bias = weights[5] - step_size * grads[5]
            new_model.max_logstd = weights[6] - step_size * grads[6]
            new_model.min_logstd = weights[7] - step_size * grads[7]

        if clip_weights is not None:
            new_model.layers[1].kernel = tf.clip_by_value(new_model.layers[1].kernel, clip_weights[0][0], clip_weights[1][0])
            new_model.layers[1].bias = tf.clip_by_value(new_model.layers[1].bias, clip_weights[0][1], clip_weights[1][1])
            new_model.layers[2].kernel = tf.clip_by_value(new_model.layers[2].kernel, clip_weights[0][2], clip_weights[1][2])
            new_model.layers[2].bias = tf.clip_by_value(new_model.layers[2].bias, clip_weights[0][3], clip_weights[1][3])
            new_model.layers[3].kernel = tf.clip_by_value(new_model.layers[3].kernel, clip_weights[0][4], clip_weights[1][4])
            new_model.layers[3].bias = tf.clip_by_value(new_model.layers[3].bias, clip_weights[0][5], clip_weights[1][5])
            new_model.max_logstd = tf.clip_by_value(new_model.max_logstd, clip_weights[0][6], clip_weights[1][6])
            new_model.min_logstd = tf.clip_by_value(new_model.min_logstd, clip_weights[0][7], clip_weights[1][7])

        return new_model


from tensorflow_probability import distributions as tfpd
import tensorflow.keras.layers as tfkl
import tensorflow as tf
import numpy as np
from tensorflow_addons.layers import SpectralNormalization
from collections import defaultdict


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

class UnrolledForwardModel():
    def __init__(self, input_shape, hidden, anchor_coef):
        self._model = ForwardModel(input_shape, hidden)
        self._meta_model = ForwardModel(input_shape, hidden)
        self.anchor_coef = anchor_coef

    def get_distribution(self, inputs, **kwargs):
        return self._model.get_distribution(inputs, **kwargs)

    @property
    def trainable_variables(self):
        return self._model.trainable_variables

    def get_meta_model(self, meta_weights):
        for layer_idx in [1, 2, 3]:
            self._meta_model.layers[layer_idx].kernel = meta_weights[2 * layer_idx - 2]
            self._meta_model.layers[layer_idx].bias = meta_weights[2 * layer_idx - 1]

        self._meta_model.max_logstd = meta_weights[-2]
        self._meta_model.min_logstd = meta_weights[-1]

        return self._meta_model

    def get_model_weights(self):
        meta_weights = []
        for layer_idx in [1, 2, 3]:
            meta_weights.extend([self._model.layers[layer_idx].kernel, self._model.layers[layer_idx].bias])

        meta_weights.extend([self._model.max_logstd, self._model.min_logstd])

        return meta_weights

    def get_meta_model_weights(self):
        meta_weights = []
        for layer_idx in [1, 2, 3]:
            meta_weights.extend([self._meta_model.layers[layer_idx].kernel, self._meta_model.layers[layer_idx].bias])

        meta_weights.extend([self._meta_model.max_logstd, self._meta_model.min_logstd])

        return meta_weights

    def get_unrolled_distributions(self, x, unrolls, unroll_rate):
        unrolled_ds = []
        anchor_weights = self.get_model_weights()
        meta_model = self.get_meta_model(anchor_weights)
        for unroll in range(unrolls):
            meta_weights = self.get_meta_model_weights()
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(meta_weights)
                d = meta_model.get_distribution(x)
                meta_loss = d.mean()

            meta_grads = tape.gradient(meta_loss, meta_weights)
            next_meta_weights = []
            for meta_weight, meta_grad, anchor_weight in zip(meta_weights, meta_grads, anchor_weights):
                meta_grad = 0.0 if meta_grad is None else meta_grad
                next_meta_weight = (
                    meta_weight - unroll_rate * meta_grad + self.anchor_coef * (anchor_weight - meta_weight)
                )
                next_meta_weights.append(next_meta_weight)

            meta_model = self.get_meta_model(next_meta_weights)
            unrolled_ds.append(d)

        d = meta_model.get_distribution(x)
        unrolled_ds.append(d)

        return unrolled_ds


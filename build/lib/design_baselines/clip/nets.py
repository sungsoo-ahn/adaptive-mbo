from tensorflow_probability import distributions as tfpd
import tensorflow.keras.layers as tfkl
import tensorflow as tf
import numpy as np
from tensorflow_addons.layers import SpectralNormalization
from collections import defaultdict


class ForwardModel(tf.keras.Sequential):
    distribution = tfpd.Normal
    def __init__(self, input_shape, hidden, spectral_normalization):
        self.input_shape_ = input_shape
        self.hidden = hidden
        self.spectral_normalization = spectral_normalization

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

        if spectral_normalization:
            for i in [1, 2, 3]:
                layers[i] = SpectralNormalization(layers[i])

        super(ForwardModel, self).__init__(layers)

    def get_params(self, inputs, **kwargs):
        prediction = super(ForwardModel, self).__call__(inputs, **kwargs)
        mean, logstd = tf.split(prediction, 2, axis=-1)
        logstd = self.max_logstd - tf.nn.softplus(self.max_logstd - logstd)
        logstd = self.min_logstd + tf.nn.softplus(logstd - self.min_logstd)
        return {"loc": mean, "scale": tf.math.exp(logstd)}

    def get_distribution(self, inputs, **kwargs):
        return self.distribution(**self.get_params(inputs, **kwargs))

    def get_meta_model(self, meta_weights):
        meta_model = ForwardModel(self.input_shape_, self.hidden, self.spectral_normalization)
        for layer_idx in [1, 2, 3]:
            if self.spectral_normalization:
                meta_model.layers[layer_idx].layer.kernel = meta_weights[2 * layer_idx - 2]
                meta_model.layers[layer_idx].layer.bias = meta_weights[2 * layer_idx - 1]
            else:
                meta_model.layers[layer_idx].kernel = meta_weights[2 * layer_idx - 2]
                meta_model.layers[layer_idx].bias = meta_weights[2 * layer_idx - 1]

        return meta_model

    def get_meta_weights(self):
        meta_weights = []
        for layer_idx in [1, 2, 3]:
            if self.spectral_normalization:
                meta_weights.extend(
                    [self.layers[layer_idx].layer.kernel, self.layers[layer_idx].layer.bias]
                )
            else:
                meta_weights.extend([self.layers[layer_idx].kernel, self.layers[layer_idx].bias])

        return meta_weights

    def get_unrolled_pred(self, x, lambdas, steps, step_size):
        meta_statistics = defaultdict(list)
        anchor_weights = self.get_meta_weights()
        meta_model = self.get_meta_model(anchor_weights)
        for step in range(steps):
            meta_weights = meta_model.get_meta_weights()
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(meta_weights)
                meta_loss = meta_model.get_distribution(x).mean()

            meta_grads = tape.gradient(meta_loss, meta_weights)
            next_meta_weights = []
            for meta_weight, meta_grad, anchor_weight, lambda_ in zip(
                meta_weights, meta_grads, anchor_weights, lambdas
            ):
                next_meta_weight = (
                    meta_weight
                    - step_size * meta_grad
                    + lambda_ * (anchor_weight - meta_weight)
                )
                next_meta_weights.append(next_meta_weight)

            meta_model = self.get_meta_model(next_meta_weights)
            meta_statistics["loss"].append(meta_loss)

        unrolled_pred = meta_loss = meta_model.get_distribution(x).mean()
        meta_statistics["loss"].append(meta_loss)

        return unrolled_pred, meta_model, meta_statistics


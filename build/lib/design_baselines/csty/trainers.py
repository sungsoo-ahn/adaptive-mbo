from design_baselines.utils import spearman
from design_baselines.utils import soft_noise
from design_baselines.utils import cont_noise
from collections import defaultdict
from tensorflow_probability import distributions as tfpd
import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np
import random


class Smoothing(tf.Module):
    def __init__(
        self,
        model,
        model_optim,
        data_size,
        ema_model,
        ema_rate,
        sol_x,
        sol_x_optim,
        consistency_coef,
        is_discrete,
        continuous_noise_std,
        discrete_smoothing,
    ):
        super().__init__()
        self.model = model
        self.model_optim = model_optim
        self.data_size = data_size

        self.ema_model = ema_model
        self.ema_rate = ema_rate

        self.init_sol_x = sol_x.read_value()
        self.sol_x = sol_x
        self.sol_x_optim = sol_x_optim

        self.consistency_coef = consistency_coef

        self.is_discrete = is_discrete
        self.noise_std = continuous_noise_std
        self.keep = discrete_smoothing

    @tf.function(experimental_relax_shapes=True)
    def train_step(self, x0, y0):
        # corrupt the inputs with noise
        if self.is_discrete:
            x0 = soft_noise(x0, self.keep)
            x1 = tf.math.softmax(self.sol_x)
        else:
            x0 = cont_noise(x0, self.noise_std)
            x1 = cont_noise(self.sol_x, self.noise_std)

        total_loss = 0.0
        nll_loss = 0.0
        consistency_loss = 0.0
        rank_correlation = 0.0
        grad_norm = 0.0
        pred_mean0 = 0.0
        pred_std0 = 0.0
        pred_mean1 = 0.0
        pred_std1 = 0.0

        with tf.GradientTape() as tape:
            d0 = self.model.get_distribution(x0, training=True)
            nll_loss = -d0.log_prob(y0)

            rank_correlation = spearman(y0[:, 0], d0.mean()[:, 0])

            # model loss that combines maximum likelihood
            params10 = self.model.get_params(x1, training=True)
            params11 = self.model.get_params(x1, training=True)
            consistency_loss = 0.5 * (
                tf.square(params10["loc"] ** 2 - params11["loc"] ** 2)
                + tf.square(params10["scale"] ** 2 - params11["scale"] ** 2)
            )
            pessimism_loss = 0.5 * (params10["loc"] + params11["loc"])

            total_loss = (
                tf.reduce_mean(nll_loss)
                + self.consistency_coef * tf.reduce_mean(consistency_loss)
                + self.pessimism_coef * tf.reduce_mean(pessimism_loss)
            )

        grads = tape.gradient(total_loss, self.model.trainable_variables)
        grad_norm = tf.reduce_mean([tf.linalg.norm(grad) for grad in grads])
        grads = [tf.clip_by_norm(grad, 1.0) for grad in grads]
        self.model_optim.apply_gradients(zip(grads, self.model.trainable_variables))

        for var, ema_var in zip(self.model.trainable_variables, self.ema_model.trainable_variables):
            ema_var.assign(self.ema_rate * ema_var + (1 - self.ema_rate) * var)

        pred_mean0 = tf.reduce_mean(d0.mean())
        pred_std0 = tf.reduce_mean(d0.stddev())
        pred_mean1 = 0.5 * tf.reduce_mean(params10["loc"] + params11["loc"])
        pred_std1 = 0.5 * tf.reduce_mean(params10["scale"] + params11["scale"])

        statistics = dict()
        statistics["total_loss"] = total_loss
        statistics["nll_loss"] = nll_loss
        statistics["rank_corr"] = rank_correlation
        statistics["consistency_loss"] = consistency_loss
        statistics["grad_norm"] = grad_norm
        statistics["mean0"] = pred_mean0
        statistics["std0"] = pred_std0
        statistics["mean1"] = pred_mean1
        statistics["std1"] = pred_std1

        return statistics

    @tf.function(experimental_relax_shapes=True)
    def validate_step(self, x, y):
        nll_loss = 0.0
        rank_correlation = 0.0
        d = self.ema_model.get_distribution(x, training=False)
        nll_loss += -d.log_prob(y)
        rank_correlation += spearman(y[:, 0], d.mean()[:, 0])

        statistics = dict()
        statistics["nll_loss"] = nll_loss
        statistics["rank_corr"] = rank_correlation
        statistics["mean"] = tf.reduce_mean(d.mean())
        statistics["std"] = tf.reduce_mean(d.stddev())

        return statistics

    @tf.function(experimental_relax_shapes=True)
    def update_solution(self):
        sol_x_grad = 0.0
        sol_y_pred = 0.0

        with tf.GradientTape() as tape:
            tape.watch(self.sol_x)
            inp = self.sol_x
            if self.is_discrete:
                inp = tf.math.softmax(inp)

            d = self.ema_model.get_distribution(inp, training=True)
            sol_x_loss = -d.mean()

        sol_x_grad = tape.gradient(sol_x_loss, self.sol_x)
        normalized_sol_x_grad, grad_norm = tf.linalg.normalize(sol_x_grad)
        sol_x_grad = tf.clip_by_norm(sol_x_grad, 1.0)
        sol_y_pred = d.mean()

        self.sol_x_optim.apply_gradients([[sol_x_grad, self.sol_x]])

        travelled = tf.linalg.norm(self.sol_x - self.init_sol_x) / tf.cast(
            tf.shape(self.sol_x)[0], dtype=tf.float32
        )

        statistics = dict()
        statistics["sol_y_pred"] = tf.reduce_mean(sol_y_pred)
        statistics["grad_norm"] = grad_norm
        statistics["travelled"] = travelled

        return statistics


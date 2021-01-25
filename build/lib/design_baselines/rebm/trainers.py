from design_baselines.utils import spearman
from design_baselines.utils import soft_noise
from design_baselines.utils import cont_noise
from collections import defaultdict
from tensorflow_probability import distributions as tfpd
import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np
import random


class MaximumLikelihood(tf.Module):
    def __init__(
        self,
        forward_model,
        forward_model_opt,
        forward_model_lr,
        sgld_lr,
        sgld_noise_penalty,
        sgld_train_steps,
        sgld_eval_steps,
        init_noise_std,
        is_discrete,
        continuous_noise_std,
        discrete_smoothing,
    ):

        super().__init__()
        self.forward_model = forward_model
        self.forward_model_opt = forward_model_opt(learning_rate=forward_model_lr)

        # parameters for controlling learning rate for negative samples
        self.sgld_lr = sgld_lr
        self.sgld_noise_penalty = sgld_noise_penalty
        self.sgld_train_steps = sgld_train_steps
        self.sgld_eval_steps = sgld_eval_steps
        self.init_noise_std = init_noise_std

        # extra parameters for controlling data noise
        self.is_discrete = is_discrete
        self.continuous_noise_std = continuous_noise_std
        self.discrete_smoothing = discrete_smoothing

    @tf.function(experimental_relax_shapes=True)
    def run_markovchain(self, x, y):
        with tf.GradientTape() as tape:
            tape.watch(y)
            inp = tf.math.softmax(x) if self.is_discrete else x
            energy = self.forward_model(inp, y)

        grad = tape.gradient(energy, y)
        noise = tf.random.normal(tf.shape(y), 0, 1)
        y = (
            y
            + 0.5 * self.sgld_lr * grad
            + self.sgld_noise_penalty * np.sqrt(self.sgld_lr) * noise
        )

        return y, energy


    @tf.function(experimental_relax_shapes=True)
    def train_step(self, x, y):
        statistics = dict()
        x = (
            soft_noise(x, self.discrete_smoothing)
            if self.is_discrete
            else cont_noise(x, self.continuous_noise_std)
        )

        #y_neg = y + tf.random.normal(tf.shape(y), 0, self.init_noise_std)
        y_neg = tf.random.normal(tf.shape(y), 0, 1.0)
        for _ in range(self.sgld_train_steps):
            y_neg, _ = self.run_markovchain(x, y_neg)

        with tf.GradientTape() as tape:
            energy_pos = self.forward_model(x, y)
            energy_neg = self.forward_model(x, y_neg)
            loss = -tf.reduce_mean(energy_pos) + tf.reduce_mean(energy_neg)

        grads = tape.gradient(loss, self.forward_model.trainable_variables)
        grads = [tf.clip_by_norm(grad, 1.0) for grad in grads]
        self.forward_model_opt.apply_gradients(zip(grads, self.forward_model.trainable_variables))

        rank_correlation = spearman(y[:, 0], y_neg[:, 0])

        statistics[f"train/energy_pos"] = energy_pos
        statistics[f"train/energy_neg"] = energy_neg
        statistics[f"train/loss"] = loss
        statistics[f"train/rank_correlation"] = rank_correlation

        return statistics

    def validate_step(self, x, y):
        statistics = dict()
        x = (
            soft_noise(x, self.discrete_smoothing)
            if self.is_discrete
            else cont_noise(x, self.continuous_noise_std)
        )
        energy_pos = self.forward_model(x, y)
        statistics[f"eval/energy_pos"] = energy_pos

        y_neg = tf.random.normal(tf.shape(y), 0, 1.0)
        for _ in range(self.sgld_eval_steps):
            y_neg, energy_neg = self.run_markovchain(x, y_neg)

        loss = -tf.reduce_mean(energy_pos) + tf.reduce_mean(energy_neg)

        rank_correlation = spearman(y[:, 0], y_neg[:, 0])

        statistics[f"eval/energy_neg"] = energy_neg
        statistics[f"eval/loss"] = loss
        statistics[f"eval/rank_correlation"] = rank_correlation

        return statistics




from design_baselines.utils import spearman
from design_baselines.utils import soft_noise
from design_baselines.utils import cont_noise
from collections import defaultdict
from tensorflow_probability import distributions as tfpd
import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np
import random


class ModelTrainer(tf.Module):
    def __init__(
        self,
        model,
        model_optim,
        is_discrete,
        continuous_noise_std,
        discrete_smoothing,
    ):
        super().__init__()
        self.model = model
        self.model_optim = model_optim

        self.is_discrete = is_discrete
        self.noise_std = continuous_noise_std
        self.keep = discrete_smoothing

    @tf.function(experimental_relax_shapes=True)
    def train_step(self, x, y, b):
        x = soft_noise(x, self.keep) if self.is_discrete else cont_noise(x, self.noise_std)

        with tf.GradientTape() as tape:
            d = self.model.get_distribution(x, training=True)
            loss_nll = -tf.math.divide_no_nan(tf.reduce_sum(b * d.log_prob(y)), tf.reduce_sum(b))
            loss_total = loss_nll
            rank_corr = spearman(y[:, 0], d.mean()[:, 0])

        grads = tape.gradient(loss_total, self.model.trainable_variables)
        self.model_optim.apply_gradients(zip(grads, self.model.trainable_variables))

        statistics = dict()
        statistics["loss/total"] = loss_total
        statistics["loss/nll"] = loss_nll
        statistics["rank_corr"] = rank_corr

        return statistics

    @tf.function(experimental_relax_shapes=True)
    def validate_step(self, x, y):
        d = self.model.get_distribution(x, training=True)
        loss_nll = -tf.reduce_mean(d.log_prob(y))
        loss_total = loss_nll
        rank_corr = spearman(y[:, 0], d.mean()[:, 0])

        statistics = dict()
        statistics["loss/total"] = loss_total
        statistics["loss/nll"] = loss_nll
        statistics["rank_corr"] = rank_corr

        return statistics

class SolutionTrainer(tf.Module):
    def __init__(
        self,
        model,
        model_weights_list,
        sol,
        sol_optim,
        is_discrete,
        continuous_noise_std,
        discrete_smoothing,
    ):
        super().__init__()
        self.model = model
        self.model_weights_list = model_weights_list

        self.sol = sol
        self.init_sol = sol.read_value()
        self.sol_optim = sol_optim

        self.is_discrete = is_discrete
        self.noise_std = continuous_noise_std
        self.keep = discrete_smoothing

    @tf.function(experimental_relax_shapes=True)
    def update_solution(self):
        model_weights = random.choice(self.model_weights_list)
        self.model.set_weights(model_weights)

        with tf.GradientTape() as tape:
            tape.watch(self.sol)
            inp = self.sol
            if self.is_discrete:
                inp = tf.math.softmax(inp)

            d = self.model.get_distribution(inp, training=True)
            loss = -d.mean()

        solution_grad = tape.gradient(loss, self.sol)
        self.sol_optim.apply_gradients([[solution_grad, self.sol]])

        travelled = tf.linalg.norm(self.sol - self.init_sol) / tf.cast(
            tf.shape(self.sol)[0], dtype=tf.float32
        )

        statistics = dict()
        statistics["loss"] = tf.reduce_mean(loss)
        statistics["travelled"] = travelled
        return statistics
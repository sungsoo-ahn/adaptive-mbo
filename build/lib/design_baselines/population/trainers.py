from design_baselines.utils import spearman
from design_baselines.utils import soft_noise
from design_baselines.utils import cont_noise
from collections import defaultdict
from tensorflow_probability import distributions as tfpd
import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np
import random

class Buffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.elems = []
        self.offset = 0

    def __len__(self):
        return len(self.elems)

    def full(self):
        return self.__len__() == self.buffer_size

    def empty(self):
        return self.__len__() == 0

    def add(self, x):
        x = x.numpy()
        new_elems = np.split(x, tf.shape(x)[0], axis=0)
        self.elems.extend(new_elems)

    def sample(self, num_samples, remove=True):
        if self.offset == self.buffer_size:
            perm = list(range(self.__len__()))
            random.shuffle(perm)
            self.elems = [self.elems[idx] for idx in perm]
            self.offset = 0

        batch_size = min(num_samples, self.buffer_size - self.offset)
        sampled_elems = self.elems[:batch_size]
        sampled_x = np.concatenate(sampled_elems, axis=0)

        if remove:
            self.offset += batch_size
            self.elems = self.elems[batch_size:]

        return sampled_x


class MaximumLikelihood(tf.Module):
    def __init__(
        self,
        forward_model,
        forward_model_opt,
        forward_model_lr,
        is_discrete,
        continuous_noise_std,
        discrete_smoothing,
    ):

        super().__init__()
        self.forward_model = forward_model
        self.forward_model_opt = forward_model_opt(learning_rate=forward_model_lr)

        # extra parameters for controlling data noise
        self.is_discrete = is_discrete
        self.continuous_noise_std = continuous_noise_std
        self.discrete_smoothing = discrete_smoothing

    @tf.function(experimental_relax_shapes=True)
    def train_step(self, x, y):
        statistics = dict()

        # corrupt the inputs with noise
        x = (
            soft_noise(x, self.discrete_smoothing)
            if self.is_discrete
            else cont_noise(x, self.continuous_noise_std)
        )

        with tf.GradientTape(persistent=True) as tape:
            #
            energy_pos = self.forward_model(x, training=True)
            y_pred = energy_pos
            loss = tf.reduce_mean((y - y_pred) ** 2)
            statistics[f"pretrain/loss_mse"] = loss

            rank_correlation = spearman(y[:, 0], y_pred[:, 0])
            statistics[f"pretrain/rank_corr"] = rank_correlation

        # take gradient steps on the model
        grads = tape.gradient(loss, self.forward_model.trainable_variables)
        self.forward_model_opt.apply_gradients(zip(grads, self.forward_model.trainable_variables))

        return statistics


class EnergyMaximumLikelihood(tf.Module):
    def __init__(
        self,
        forward_model,
        forward_model_opt,
        forward_model_lr,
        x_lr,
        alpha,
        is_discrete,
        continuous_noise_std,
        discrete_smoothing,
    ):

        super().__init__()
        self.forward_model = forward_model
        self.forward_model_opt = forward_model_opt(learning_rate=forward_model_lr)

        # parameters for controlling learning rate for negative samples
        self.x_lr = x_lr

        #
        self.alpha = alpha

        # extra parameters for controlling data noise
        self.is_discrete = is_discrete
        self.continuous_noise_std = continuous_noise_std
        self.discrete_smoothing = discrete_smoothing

    @tf.function(experimental_relax_shapes=True)
    def run_markovchain(self, x):
        with tf.GradientTape() as tape:
            tape.watch(x)
            energy = self.forward_model(tf.math.softmax(x) if self.is_discrete else x)

        grads = tape.gradient(energy, x)
        x = x + self.x_lr * grads

        return x, energy


    @tf.function(experimental_relax_shapes=True)
    def train_step(self, x, y, x_neg):
        statistics = dict()
        x = (
            soft_noise(x, self.discrete_smoothing)
            if self.is_discrete
            else cont_noise(x, self.continuous_noise_std)
        )

        with tf.GradientTape() as tape:
            energy_pos = self.forward_model(x, training=True)
            energy_neg = self.forward_model(tf.math.softmax(x_neg) if self.is_discrete else x_neg)
            y_pred = energy_pos

            loss_mse = tf.reduce_mean((y - y_pred) ** 2)
            loss_pos = -tf.reduce_mean(energy_pos)
            loss_neg = tf.reduce_mean(energy_neg)
            total_loss = loss_mse + self.alpha * (loss_pos + loss_neg)

        # take gradient steps on the model
        grads = tape.gradient(total_loss, self.forward_model.trainable_variables)
        grads = [tf.clip_by_norm(g, 1.0) for g in grads]
        self.forward_model_opt.apply_gradients(zip(grads, self.forward_model.trainable_variables))

        rank_correlation = spearman(y[:, 0], y_pred[:, 0])

        statistics[f"train/loss_mse"] = loss_mse
        statistics[f"train/loss_pos"] = loss_pos
        statistics[f"train/loss_neg"] = loss_neg
        statistics[f"train/loss_total"] = total_loss
        statistics[f"train/rank_corr"] = rank_correlation

        return statistics



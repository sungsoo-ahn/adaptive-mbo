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
    def __init__(self):
        self.elems = []
        self.offset = 0

    def __len__(self):
        return len(self.elems)

    def add(self, x):
        x = x.numpy()
        new_elems = np.split(x, tf.shape(x)[0], axis=0)
        self.elems.extend(new_elems)
        self.shuffle()

    def sample(self, num_samples):
        if self.offset == len(self.elems):
            self.shuffle()

        batch_size = min(num_samples, len(self.elems) - self.offset)
        sampled_elems = self.elems[self.offset : self.offset + batch_size]
        sampled_x = np.concatenate(sampled_elems, axis=0)

        return sampled_x

    def shuffle(self):
        perm = list(range(len(self.elems)))
        random.shuffle(perm)
        self.elems = [self.elems[idx] for idx in perm]
        self.offset = 0


class EntMin(tf.Module):
    def __init__(
        self,
        forward_model,
        forward_model_optim,
        forward_model_lr,
        alpha,
        is_discrete,
        continuous_noise_std,
        discrete_smoothing,
    ):
        super().__init__()
        self.fm = forward_model
        self.optim = forward_model_optim(learning_rate=forward_model_lr)

        self.alpha = alpha

        self.is_discrete = is_discrete
        self.noise_std = continuous_noise_std
        self.keep = discrete_smoothing

    @tf.function(experimental_relax_shapes=True)
    def train_step(self, x, y, x_neg):
        # corrupt the inputs with noise
        x0 = soft_noise(x, self.keep) if self.is_discrete else cont_noise(x, self.noise_std)
        x1 = soft_noise(x_neg, self.keep) if self.is_discrete else cont_noise(x_neg, self.noise_std)

        statistics = dict()
        with tf.GradientTape(persistent=True) as tape:
            # calculate the prediction error and accuracy of the model
            d0 = self.fm.get_distribution(x0, training=True)
            nll = -d0.log_prob(y)
            y_pred = d0.mean()
            y_ent = d0.entropy()

            # evaluate how correct the rank fo the model predictions are
            rank_correlation = spearman(y[:, 0], y_pred[:, 0])

            # model loss that combines maximum likelihood
            d1 = self.fm.get_distribution(x1, training=True)
            y_prior = tfpd.Normal(loc=0.0, scale=1.0)
            d1_kl_div = tfp.distributions.kl_divergence(d1, y_prior)

            model_loss = tf.reduce_mean(nll) + self.alpha * tf.reduce_mean(d1_kl_div)

        model_grads = tape.gradient(model_loss, self.fm.trainable_variables)
        self.optim.apply_gradients(zip(model_grads, self.fm.trainable_variables))

        statistics["nll"] = tf.reduce_mean(nll)
        statistics["stddev"] = tf.reduce_mean(d0.stddev())
        statistics["rank_corr"] = rank_correlation
        statistics["neg_pred"] = tf.reduce_mean(d1.mean())
        statistics["neg_kl_div"] = tf.reduce_mean(d1_kl_div)
        #statistics["alpha"] = self.alpha

        return statistics

    @tf.function(experimental_relax_shapes=True)
    def warmup_train_step(self, x, y):
        # corrupt the inputs with noise
        x0 = soft_noise(x, self.keep) if self.is_discrete else cont_noise(x, self.noise_std)

        statistics = dict()
        with tf.GradientTape(persistent=True) as tape:
            # calculate the prediction error and accuracy of the model
            d0 = self.fm.get_distribution(x0, training=True)
            nll = -d0.log_prob(y)
            y_pred = d0.mean()
            y_ent = d0.entropy()

            model_loss = tf.reduce_mean(nll)

            # evaluate how correct the rank fo the model predictions are
            rank_correlation = spearman(y[:, 0], y_pred[:, 0])

        model_grads = tape.gradient(model_loss, self.fm.trainable_variables)
        self.optim.apply_gradients(zip(model_grads, self.fm.trainable_variables))

        statistics["nll"] = tf.reduce_mean(nll)
        statistics["entropy"] = tf.reduce_mean(y_ent)
        statistics["rank_corr"] = rank_correlation

        return statistics
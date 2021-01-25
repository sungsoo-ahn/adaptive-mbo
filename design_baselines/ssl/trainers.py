from design_baselines.utils import spearman
from design_baselines.utils import soft_noise
from design_baselines.utils import cont_noise
from collections import defaultdict
from tensorflow_probability import distributions as tfpd
import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np


class SemiSupervisedMaximumLikelihood(tf.Module):
    def __init__(
        self,
        forward_models,
        forward_model_optim,
        forward_model_lr,
        augment,
        augment_steps=50,
        augment_lr=0.001,
        augment_std_penalty=10.0,
        is_discrete=False,
        continuous_noise_std=0.0,
        discrete_smoothing=0.6,
    ):
        super().__init__()
        self.forward_models = forward_models
        self.bootstraps = len(forward_models)

        self.forward_model_optims = [
            forward_model_optim(learning_rate=forward_model_lr) for i in range(self.bootstraps)
        ]

        self.augment_ = augment

        self.augment_steps = augment_steps
        self.augment_lr = augment_lr
        self.augment_std_penalty = augment_std_penalty
        self.is_discrete = is_discrete
        self.continuous_noise_std = continuous_noise_std
        self.discrete_smoothing = discrete_smoothing

    def get_distribution(self, x, **kwargs):
        params = defaultdict(list)
        for fm in self.forward_models:
            for key, val in fm.get_params(x, **kwargs).items():
                params[key].append(val)

        for key, val in params.items():
            params[key] = tf.stack(val, axis=-1)

        weights = tf.fill([self.bootstraps], 1 / self.bootstraps)
        return tfpd.MixtureSameFamily(
            tfpd.Categorical(probs=weights), self.forward_models[0].distribution(**params)
        )

    @tf.function(experimental_relax_shapes=True)
    def augment(self, x, steps, **kwargs):
        def gradient_step(xt):
            with tf.GradientTape() as tape:
                tape.watch(xt)
                yd = self.get_distribution(
                    tf.math.softmax(xt) if self.is_discrete else xt, **kwargs
                )
                score = yd.mean() - self.augment_std_penalty * yd.stddev()

            return xt + self.augment_lr * tape.gradient(score, xt)

        return tf.while_loop(lambda xt: True, gradient_step, (x,), maximum_iterations=steps)[0]

    @tf.function(experimental_relax_shapes=True)
    def train_step(self, x, y, b):
        statistics = dict()

        aug_x_init = (
            soft_noise(x, self.discrete_smoothing)
            if self.is_discrete
            else cont_noise(x, self.continuous_noise_std)
        )
        aug_x_init = tf.math.log(aug_x_init) if self.is_discrete else aug_x_init
        aug_x = self.augment(aug_x_init, self.augment_steps, training=False)
        aug_x = tf.math.softmax(aug_x) if self.is_discrete else aug_x
        aug_y = self.get_distribution(aug_x).mean()

        statistics["aug_diff"] = tf.norm(aug_x - aug_x_init)

        if aug_x.shape[0] == self.solution.shape[0]:
            self.solution.assign(aug_x)

        for i in range(self.bootstraps):
            fm = self.forward_models[i]
            fm_optim = self.forward_model_optims[i]

            with tf.GradientTape(persistent=True) as tape:
                d = fm.get_distribution(x, training=True)
                nll = -d.log_prob(y)[:, 0]

                rank_correlation = spearman(y[:, 0], d.mean()[:, 0])

                total_loss = tf.math.divide_no_nan(
                    tf.reduce_sum(b[:, i] * nll), tf.reduce_sum(b[:, i])
                )

                if self.augment_:
                    aug_d = fm.get_distribution(aug_x, training=True)
                    aug_nll = -aug_d.log_prob(aug_y)[:, 0]
                    total_loss += tf.math.divide_no_nan(
                        tf.reduce_sum(b[:, i] * aug_nll), tf.reduce_sum(b[:, i])
                    )

            grads = tape.gradient(total_loss, fm.trainable_variables)
            fm_optim.apply_gradients(zip(grads, fm.trainable_variables))

            statistics[f"oracle_{i}/train/nll"] = nll
            statistics[f"oracle_{i}/train/rank_corr"] = rank_correlation
            if self.augment_:
                statistics[f"oracle_{i}/train/aug_nll"] = aug_nll

        return statistics

    @tf.function(experimental_relax_shapes=True)
    def validate_step(self, x, y):
        statistics = dict()

        for i in range(self.bootstraps):
            fm = self.forward_models[i]

            d = fm.get_distribution(x, training=False)
            nll = -d.log_prob(y)[:, 0]

            rank_correlation = spearman(y[:, 0], d.mean()[:, 0])

            statistics[f"oracle_{i}/validate/nll"] = nll
            statistics[f"oracle_{i}/validate/rank_corr"] = rank_correlation

        return statistics

    def train(self, dataset):
        statistics = defaultdict(list)
        for x, y, b in dataset:
            for name, tensor in self.train_step(x, y, b).items():
                statistics[name].append(tensor)
        for name in statistics.keys():
            statistics[name] = tf.concat(statistics[name], axis=0)
        return statistics

    def validate(self, dataset):
        statistics = defaultdict(list)
        for x, y in dataset:
            for name, tensor in self.validate_step(x, y).items():
                statistics[name].append(tensor)
        for name in statistics.keys():
            statistics[name] = tf.concat(statistics[name], axis=0)
        return statistics

    def launch(self, train_data, validate_data, logger, epochs, start_epoch=0):
        for e in range(start_epoch, start_epoch + epochs):
            for name, loss in self.train(train_data).items():
                logger.record(name, loss, e)
            for name, loss in self.validate(validate_data).items():
                logger.record(name, loss, e)

    def get_saveables(self):
        saveables = dict()
        for i in range(self.bootstraps):
            saveables[f"forward_model_{i}"] = self.forward_models[i]
            saveables[f"forward_model_optim_{i}"] = self.forward_model_optims[i]
        return saveables

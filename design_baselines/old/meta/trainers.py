from design_baselines.utils import spearman
from design_baselines.utils import soft_noise
from design_baselines.utils import cont_noise
from collections import defaultdict
from tensorflow_probability import distributions as tfpd
import tensorflow as tf

class MaximumLikelihood(tf.Module):
    def __init__(
        self,
        forward_model,
        forward_model_optim,
        forward_model_lr,
        bootstrap_id,
        is_discrete,
        continuous_noise_std,
        discrete_smoothing,
    ):
        super().__init__()
        self.fm = forward_model
        self.optim = forward_model_optim(learning_rate=forward_model_lr)
        self.bootstrap_id = bootstrap_id

        # create machinery for sampling adversarial examples
        self.is_discrete = is_discrete
        self.noise_std = continuous_noise_std
        self.keep = discrete_smoothing

    @tf.function(experimental_relax_shapes=True)
    def train_step(self, x, y, b):
        # corrupt the inputs with noise
        x0 = soft_noise(x, self.keep) if self.is_discrete else cont_noise(x, self.noise_std)

        statistics = dict()
        with tf.GradientTape(persistent=True) as tape:

            # calculate the prediction error and accuracy of the model
            d = self.fm.get_distribution(x0, training=True)
            nll = -d.log_prob(y)

            # evaluate how correct the rank fo the model predictions are
            rank_correlation = spearman(y[:, 0], d.mean()[:, 0])

            # model loss that combines maximum likelihood
            model_loss = nll

            # build the total and lagrangian losses
            #denom = tf.reduce_sum(b)
            #total_loss = tf.math.divide_no_nan(tf.reduce_sum(b * model_loss), denom)

            total_loss = tf.math.divide_no_nan(
                tf.reduce_sum(b[:, self.bootstrap_id] * nll), tf.reduce_sum(b[:, self.bootstrap_id])
            )



        grads = tape.gradient(total_loss, self.fm.trainable_variables)
        self.optim.apply_gradients(zip(grads, self.fm.trainable_variables))

        statistics[f"train/nll"] = nll
        statistics[f"train/rank_corr"] = rank_correlation

        return statistics

    @tf.function(experimental_relax_shapes=True)
    def validate_step(self, x, y):
        # corrupt the inputs with noise
        x0 = soft_noise(x, self.keep) if self.is_discrete else cont_noise(x, self.noise_std)

        statistics = dict()

        # calculate the prediction error and accuracy of the model
        d = self.fm.get_distribution(x0, training=False)
        nll = -d.log_prob(y)

        # evaluate how correct the rank fo the model predictions are
        rank_correlation = spearman(y[:, 0], d.mean()[:, 0])

        statistics[f"validate/nll"] = nll
        statistics[f"validate/rank_corr"] = rank_correlation

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

    def launch(self, train_data, validate_data, logger, epochs, start_epoch=0, header=""):
        for e in range(start_epoch, start_epoch + epochs):
            for name, loss in self.train(train_data).items():
                logger.record(header + name, loss, e)
            for name, loss in self.validate(validate_data).items():
                logger.record(header + name, loss, e)
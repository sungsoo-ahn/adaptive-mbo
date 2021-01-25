from design_baselines.utils import spearman
from design_baselines.utils import soft_noise
from design_baselines.utils import cont_noise
from collections import defaultdict
from tensorflow_probability import distributions as tfpd
import tensorflow as tf

class MaximumLikelihood(tf.Module):
    def __init__(
        self,
        model,
        model_optim,
        model_lr,
        unrolls,
        unroll_rate,
        unrolled_coef,
        is_discrete,
        continuous_noise_std,
        discrete_smoothing,
    ):
        super().__init__()
        self.model = model
        self.optim = model_optim(learning_rate=model_lr)

        #
        self.unrolls = unrolls
        self.unroll_rate = unroll_rate
        self.unrolled_coef = unrolled_coef

        #
        self.is_discrete = is_discrete
        self.noise_std = continuous_noise_std
        self.keep = discrete_smoothing

    @tf.function(experimental_relax_shapes=True)
    def warmup_train_step(self, x, y):
        x0 = soft_noise(x, self.keep) if self.is_discrete else cont_noise(x, self.noise_std)

        statistics = dict()
        with tf.GradientTape(persistent=True) as tape:
            d = self.model.get_distribution(x0, training=True)
            nll = - d.log_prob(y)

            total_loss = tf.reduce_mean(nll)

            rank_corr = spearman(y[:, 0], d.mean()[:, 0])

        grads = tape.gradient(total_loss, self.model.trainable_variables)
        grads = [tf.clip_by_norm(grad, 1.0) for grad in grads]
        self.optim.apply_gradients(zip(grads, self.model.trainable_variables))

        statistics[f"train/nll"] = nll
        statistics[f"train/rank_corr"] = rank_corr

        return statistics

    @tf.function(experimental_relax_shapes=True)
    def train_step(self, x, y):
        x0 = soft_noise(x, self.keep) if self.is_discrete else cont_noise(x, self.noise_std)

        statistics = dict()
        with tf.GradientTape(persistent=True) as tape:
            # evaluate how correct the rank fo the model predictions are
            unrolled_ds = self.model.get_unrolled_distributions(
                x0, unrolls=self.unrolls, unroll_rate=self.unroll_rate
                )
            nll = - unrolled_ds[0].log_prob(y)
            unrolled_nll = -unrolled_ds[-1].log_prob(y)
            total_loss = tf.reduce_mean(nll + self.unrolled_coef * unrolled_nll)

            y_pred = unrolled_ds[0].mean()
            unrolled_y_pred = unrolled_ds[-1].mean()

            rank_corr = spearman(y[:, 0], y_pred[:, 0])
            unrolled_rank_corr = spearman(y[:, 0], unrolled_y_pred[:, 0])

        grads = tape.gradient(total_loss, self.model.trainable_variables)
        grads = [tf.clip_by_norm(grad, 1.0) for grad in grads]
        self.optim.apply_gradients(zip(grads, self.model.trainable_variables))

        statistics[f"train/nll"] = nll
        statistics[f"train/unrolled_nll"] = unrolled_nll
        statistics[f"train/total_loss"] = total_loss

        statistics[f"train/y_pred"] = y_pred
        statistics[f"train/unrolled_y_pred"] = unrolled_y_pred
        statistics[f"train/unrolled_delta"] = y_pred - unrolled_y_pred

        statistics[f"train/rank_corr"] = rank_corr
        statistics[f"train/unrolled_rank_corr"] = unrolled_rank_corr

        return statistics

    @tf.function(experimental_relax_shapes=True)
    def warmup_validate_step(self, x, y):
        # corrupt the inputs with noise
        x0 = soft_noise(x, self.keep) if self.is_discrete else cont_noise(x, self.noise_std)

        statistics = dict()

        # calculate the prediction error and accuracy of the model
        d = self.model.get_distribution(x0, training=False)
        nll = -d.log_prob(y)[:, 0]

        rank_corr = spearman(y[:, 0], d.mean()[:, 0])

        statistics[f"validate/nll"] = nll
        statistics[f"validate/rank_corr"] = rank_corr

        return statistics

    @tf.function(experimental_relax_shapes=True)
    def validate_step(self, x, y):
        # corrupt the inputs with noise
        x0 = soft_noise(x, self.keep) if self.is_discrete else cont_noise(x, self.noise_std)

        statistics = dict()

        # calculate the prediction error and accuracy of the model
        unrolled_ds = self.model.get_unrolled_distributions(
            x0, unrolls=self.unrolls, unroll_rate=self.unroll_rate
            )
        nll = - unrolled_ds[0].log_prob(y)
        unrolled_nll = -unrolled_ds[-1].log_prob(y)
        total_loss = tf.reduce_mean(nll + self.unrolled_coef * unrolled_nll)

        y_pred = unrolled_ds[0].mean()
        unrolled_y_pred = unrolled_ds[-1].mean()

        rank_corr = spearman(y[:, 0], y_pred[:, 0])
        unrolled_rank_corr = spearman(y[:, 0], unrolled_y_pred[:, 0])

        statistics[f"validate/nll"] = nll
        statistics[f"validate/unrolled_nll"] = unrolled_nll
        statistics[f"validate/total_loss"] = total_loss

        statistics[f"validate/y_pred"] = y_pred
        statistics[f"validate/unrolled_y_pred"] = unrolled_y_pred
        statistics[f"validate/unrolled_delta"] = y_pred - unrolled_y_pred

        statistics[f"validate/rank_corr"] = rank_corr
        statistics[f"validate/unrolled_rank_corr"] = unrolled_rank_corr

        return statistics

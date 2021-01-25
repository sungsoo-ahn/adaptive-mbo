from design_baselines.utils import spearman
from design_baselines.utils import soft_noise
from design_baselines.utils import cont_noise
from collections import defaultdict
from tensorflow_probability import distributions as tfpd
import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np

class ConservativeMaximumLikelihood(tf.Module):
    def __init__(
        self,
        forward_model,
        forward_model_opt,
        forward_model_lr,
        alpha,
        solver_lr,
        solver_noise_penalty,
        is_discrete,
        continuous_noise_std,
        discrete_smoothing,
    ):
        super().__init__()
        self.forward_model = forward_model
        self.forward_model_opt = forward_model_opt(learning_rate=forward_model_lr)

        # lagrangian dual descent variables
        self.alpha = alpha
        self.solver_lr = solver_lr
        self.solver_noise_penalty = solver_noise_penalty

        # extra parameters for controlling data noise
        self.is_discrete = is_discrete
        self.continuous_noise_std = continuous_noise_std
        self.discrete_smoothing = discrete_smoothing

        self.solution = None

    @tf.function(experimental_relax_shapes=True)
    def lookahead(self, x):
        with tf.GradientTape() as tape:
            tape.watch(x)
            score = self.forward_model.get_distribution(
                tf.math.softmax(x) if self.is_discrete else x
            ).mean()

        grad = tape.gradient(score, x)
        #grad = tf.clip_by_norm(grad, 1.0)
        noise = tf.random.normal(x.shape, 0, 1)
        x = (
            x
            + 0.5 * self.solver_lr * grad
            + self.solver_noise_penalty * np.sqrt(self.solver_lr) * noise
        )

        return x

    @tf.function(experimental_relax_shapes=True)
    def train_step(self, x, y):
        statistics = dict()
        batch_dim = tf.shape(y)[0]

        # corrupt the inputs with noise
        x = (
            soft_noise(x, self.discrete_smoothing)
            if self.is_discrete
            else cont_noise(x, self.continuous_noise_std)
        )

        with tf.GradientTape() as tape:
            d = self.forward_model.get_distribution(x)
            nll = -d.log_prob(y)
            rank_corr = spearman(y[:, 0], d.mean()[:, 0])
            total_loss = tf.reduce_mean(nll)

            if self.alpha > 0.0:
                y_pos_mean = d.mean()[:, 0]

                x_neg = self.lookahead(self.solution)
                x_neg = tf.math.softmax(x_neg) if self.is_discrete else x_neg
                d_neg = self.forward_model.get_distribution(x_neg)
                y_neg_mean = d_neg.mean()[:, 0]

                total_loss = total_loss + self.alpha * (tf.reduce_mean(y_neg_mean) - tf.reduce_mean(y_pos_mean))

            statistics[f"train/nll"] = nll
            statistics[f"train/rank_corr"] = rank_corr
            statistics[f"train/total_loss"] = total_loss

            if self.alpha > 0.0:
                statistics[f"train/y_pos_mean"] = y_pos_mean
                statistics[f"train/y_neg_mean"] = y_neg_mean

        # take gradient steps on the model
        grads = tape.gradient(total_loss, self.forward_model.trainable_variables)
        self.forward_model_opt.apply_gradients(zip(grads, self.forward_model.trainable_variables))

        return statistics

    def update_solution(self):
        statistics = dict()
        update = self.lookahead(self.solution)
        statistics[f"update/travelled"] = tf.linalg.norm(self.solution - update, axis=-1)

        self.solution.assign(update)

        return statistics


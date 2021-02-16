from design_baselines.utils import spearman
from design_baselines.utils import perturb
import tensorflow as tf
import numpy as np
import random

class Trainer(tf.Module):
    def __init__(
        self,
        model,
        model_opt,
        ema_model,
        perturb_fn,
        sol_x,
        sol_x_opt,
        coef_pessimism,
        coef_smoothing,
        ema_rate,
    ):

        super().__init__()
        self.model = model
        self.model_opt = model_opt
        self.ema_model = ema_model
        self.perturb_fn = perturb_fn
        self.init_sol_x = sol_x
        self.sol_x = tf.Variable(sol_x)
        self.sol_x_opt = sol_x_opt
        self.coef_pessimism = coef_pessimism
        self.coef_smoothing = coef_smoothing
        self.ema_rate = ema_rate

    def get_sol_x(self):
        return self.sol_x.read_value()

    @tf.function(experimental_relax_shapes=True)
    def warmup_step(self, x, y):
        x = self.perturb_fn(x)
        with tf.GradientTape(persistent=True) as tape:
            d = self.model.get_distribution(x, training=True)
            loss_nll = -tf.reduce_mean(d.log_prob(y))
            rank_correlation = spearman(y[:, 0], d.mean()[:, 0])
            loss_total = loss_nll

        # take gradient steps on the model
        grads = tape.gradient(loss_total, self.model.trainable_variables)
        self.model_opt.apply_gradients(zip(grads, self.model.trainable_variables))

        statistics = dict()
        statistics["loss/nll"] = loss_nll
        statistics["loss/total"] = loss_total
        statistics["mean"] = tf.reduce_mean(d.mean())
        statistics["stddev"] = tf.reduce_mean(d.stddev())
        statistics["rank_corr"] = rank_correlation

        return statistics

    @tf.function(experimental_relax_shapes=True)
    def train_step(self, x, y):
        pert_x = self.perturb_fn(x)
        pert_sol_x0 = self.perturb_fn(self.sol_x)
        pert_sol_x1 = self.perturb_fn(self.sol_x)
        with tf.GradientTape(persistent=True) as tape:
            d = self.model.get_distribution(pert_x, training=True)
            loss_nll = -d.log_prob(y)
            rank_correlation = spearman(y[:, 0], d.mean()[:, 0])

            pert_sol_d0 = self.model.get_distribution(pert_sol_x0, training=True)
            pert_sol_d1 = self.model.get_distribution(pert_sol_x1, training=True)
            loss_pessimism = 0.5 * (pert_sol_d0.mean() + pert_sol_d1.mean())
            loss_mean_smoothing = tf.square(pert_sol_d0.mean() - pert_sol_d1.mean())
            loss_stddev_smoothing = tf.square(pert_sol_d0.stddev() ** 2 - pert_sol_d1.stddev() ** 2)

            loss_total = (
                tf.reduce_mean(loss_nll)
                + self.coef_pessimism * tf.reduce_mean(loss_pessimism)
                + 0.5 * self.coef_smoothing * tf.reduce_mean(loss_mean_smoothing)
                + 0.5 * self.coef_smoothing * tf.reduce_mean(loss_stddev_smoothing)
            )

        # take gradient steps on the model
        grads = tape.gradient(loss_total, self.model.trainable_variables)
        grads = [tf.clip_by_norm(grad, 1.0) for grad in grads]
        self.model_opt.apply_gradients(zip(grads, self.model.trainable_variables))

        for var, ema_var in zip(self.model.trainable_variables, self.ema_model.trainable_variables):
            ema_var.assign(self.ema_rate * ema_var + (1 - self.ema_rate) * var)

        statistics = dict()
        statistics["loss/nll"] = loss_nll
        statistics["loss/pessimism"] = loss_pessimism
        statistics["loss/mean_smoothing"] = loss_mean_smoothing
        statistics["loss/stddev_smoothing"] = loss_stddev_smoothing
        statistics["loss/total"] = loss_total
        statistics["mean"] = tf.reduce_mean(d.mean())
        statistics["stddev"] = tf.reduce_mean(d.stddev())
        statistics["rank_corr"] = rank_correlation

        return statistics

    @tf.function(experimental_relax_shapes=True)
    def validate_step(self, x, y):
        d = self.model.get_distribution(x, training=True)
        loss_nll = -tf.reduce_mean(d.log_prob(y))
        rank_correlation = spearman(y[:, 0], d.mean()[:, 0])
        loss_total = loss_nll

        statistics = dict()
        statistics["loss/nll"] = loss_nll
        statistics["loss/total"] = loss_total
        statistics["mean"] = tf.reduce_mean(d.mean())
        statistics["stddev"] = tf.reduce_mean(d.stddev())
        statistics["rank_corr"] = rank_correlation

        return statistics

    @tf.function(experimental_relax_shapes=True)
    def update_step(self):
        with tf.GradientTape() as tape:
            tape.watch(self.sol_x)
            d = self.ema_model.get_distribution(self.sol_x, training=False)
            loss = -d.mean()

        sol_x_grad = tape.gradient(loss, self.sol_x)
        sol_x_grad = tf.clip_by_norm(sol_x_grad, 1.0)
        self.sol_x_opt.apply_gradients([[sol_x_grad, self.sol_x]])

        travelled = tf.linalg.norm(self.sol_x - self.init_sol_x) / tf.cast(
            tf.shape(self.sol_x)[0], dtype=tf.float32
        )

        statistics = dict()
        statistics["loss"] = tf.reduce_mean(loss)
        statistics["mean"] = tf.reduce_mean(d.mean())
        statistics["stddev"] = tf.reduce_mean(d.stddev())
        statistics["travelled"] = travelled

        return statistics
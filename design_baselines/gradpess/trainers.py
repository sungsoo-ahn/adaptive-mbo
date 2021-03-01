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
        ema_rate,
        perturb_fn,
        is_discrete,
        sol_x,
        sol_x_opt,
        coef_pessimism,
        coef_stddev,
    ):

        super().__init__()
        self.model = model
        self.model_opt = model_opt
        self.ema_model = ema_model
        self.ema_rate = ema_rate
        self.perturb_fn = perturb_fn
        self.is_discrete = is_discrete
        self.init_sol_x = sol_x
        self.sol_x = tf.Variable(sol_x)
        self.sol_x_opt = sol_x_opt
        self.coef_pessimism = coef_pessimism
        self.coef_stddev = coef_stddev
        self.sol_x_samples = tf.shape(self.sol_x)[0]

    def get_sol_x(self):
        return self.sol_x.read_value()

    @tf.function(experimental_relax_shapes=True)
    def train_step(self, x, y):
        x = self.perturb_fn(x)

        with tf.GradientTape() as outer_tape:
            d = self.model.get_distribution(x, training=True)
            loss_nll = -d.log_prob(y)
            rank_correlation = spearman(y[:, 0], d.mean()[:, 0])

            with tf.GradientTape() as inner_tape:
                inner_tape.watch(self.sol_x)
                inp = tf.math.softmax(self.sol_x) if self.is_discrete else self.sol_x
                sol_d = self.model.get_distribution(inp, training=True)
                loss_sol_x = sol_d.mean() - self.coef_stddev * tf.math.log(sol_d.stddev())

            sol_x_grad = inner_tape.gradient(loss_sol_x, self.sol_x)
            loss_pessimism = tf.norm(tf.reshape(sol_x_grad, [self.sol_x_samples, -1]), axis=1)
            loss_total = (
                tf.reduce_mean(loss_nll) + self.coef_pessimism * tf.reduce_mean(loss_pessimism)
                )

        # take gradient steps on the model
        grads = outer_tape.gradient(loss_total, self.model.trainable_variables)
        grads = [tf.clip_by_norm(grad, 1.0) for grad in grads]
        self.model_opt.apply_gradients(zip(grads, self.model.trainable_variables))

        for var, ema_var in zip(self.model.trainable_variables, self.ema_model.trainable_variables):
            ema_var.assign(self.ema_rate * ema_var + (1 - self.ema_rate) * var)

        statistics = dict()
        statistics["loss/nll"] = loss_nll
        statistics["loss/pessimism"] = loss_pessimism
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
            inp = tf.math.softmax(self.sol_x) if self.is_discrete else self.sol_x
            d = self.ema_model.get_distribution(inp, training=False)
            loss = -(d.mean() - self.coef_stddev * tf.math.log(d.stddev()))

        sol_x_grad = tape.gradient(loss, self.sol_x)
        sol_x_grad_norm = tf.norm(tf.reshape(sol_x_grad, [self.sol_x_samples, -1]), axis=1)
        self.sol_x_opt.apply_gradients([[sol_x_grad, self.sol_x]])

        travelled = tf.linalg.norm(self.sol_x - self.init_sol_x) / tf.cast(
            tf.shape(self.sol_x)[0], dtype=tf.float32
        )

        statistics = dict()
        statistics["loss"] = tf.reduce_mean(loss)
        statistics["mean"] = tf.reduce_mean(d.mean())
        statistics["log_stddev"] = tf.reduce_mean(tf.math.log(d.stddev()))
        statistics["travelled"] = travelled
        statistics["sol_x_grad_norm"] = tf.reduce_mean(sol_x_grad_norm)

        return statistics
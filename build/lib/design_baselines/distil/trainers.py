from design_baselines.utils import spearman
from design_baselines.utils import perturb
import tensorflow as tf
import numpy as np
import random

class MainTrainer(tf.Module):
    def __init__(
        self,
        model,
        model_opt,
        perturb_fn,
        sol_x,
        sol_x_opt,
    ):

        super().__init__()
        self.model = model
        self.model_opt = model_opt
        self.perturb_fn = perturb_fn
        self.init_sol_x = sol_x
        self.sol_x = tf.Variable(sol_x)
        self.sol_x_opt = sol_x_opt

    def get_sol_x(self):
        return self.sol_x.read_value()

    @tf.function(experimental_relax_shapes=True)
    def train_step(self, x, y):
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
            d = self.model.get_distribution(self.sol_x, training=False)
            loss = -d.mean()

        sol_xution_grad = tape.gradient(loss, self.sol_x)
        self.sol_x_opt.apply_gradients([[sol_xution_grad, self.sol_x]])

        travelled = tf.linalg.norm(self.sol_x - self.init_sol_x) / tf.cast(
            tf.shape(self.sol_x)[0], dtype=tf.float32
        )

        statistics = dict()
        statistics["loss"] = tf.reduce_mean(loss)
        statistics["mean"] = tf.reduce_mean(d.mean())
        statistics["stddev"] = tf.reduce_mean(d.stddev())
        statistics["travelled"] = travelled

        return statistics

class PessimisticTrainer(tf.Module):
    def __init__(
        self,
        model,
        model_opt,
        perturb_fn,
        sol_x,
        coef_pessimism,
        ema_rate,
    ):

        super().__init__()
        self.model = model
        self.model_opt = model_opt
        self.perturb_fn = perturb_fn
        self.init_sol_x = sol_x
        self.sol_x = tf.Variable(sol_x)
        self.sol_y = tf.Variable(tf.zeros([sol_x.shape[0], 1]))
        self.coef_pessimism = coef_pessimism
        self.ema_rate = ema_rate

    def assign_sol_x(self, sol_x):
        self.sol_x.assign(sol_x)

    def get_sol_y(self):
        return self.sol_y.read_value()

    @tf.function(experimental_relax_shapes=True)
    def init_sol_y(self):
        sol_y = self.model.get_distribution(self.sol_x, training=True).mean()
        self.sol_y.assign(sol_y)

    @tf.function(experimental_relax_shapes=True)
    def update_sol_y(self, sol_y):
        next_sol_y = self.ema_rate*self.sol_y + (1-self.ema_rate) * sol_y
        self.sol_y.assign(next_sol_y)

    @tf.function(experimental_relax_shapes=True)
    def train_step(self, x, y):
        x = self.perturb_fn(x)
        sol_x = self.perturb_fn(self.sol_x)

        with tf.GradientTape(persistent=True) as tape:
            d = self.model.get_distribution(x, training=True)
            loss_nll = -tf.reduce_mean(d.log_prob(y))
            rank_correlation = spearman(y[:, 0], d.mean()[:, 0])

            sol_d = self.model.get_distribution(sol_x, training=True)
            loss_pessimism = - tf.reduce_mean(d.mean()) + tf.reduce_mean(sol_d.mean())

            loss_total = loss_nll + self.coef_pessimism * loss_pessimism

        # take gradient steps on the model
        grads = tape.gradient(loss_total, self.model.trainable_variables)
        self.model_opt.apply_gradients(zip(grads, self.model.trainable_variables))

        self.update_sol_y(sol_d.mean())

        statistics = dict()
        statistics["loss/nll"] = loss_nll
        statistics["loss/pessimism"] = loss_pessimism
        statistics["loss/total"] = loss_total
        statistics["mean"] = tf.reduce_mean(d.mean())
        statistics["sol_mean"] = tf.reduce_mean(sol_d.mean())
        statistics["stddev"] = tf.reduce_mean(d.stddev())
        statistics["sol_stddev"] = tf.reduce_mean(sol_d.stddev())
        statistics["sol_y"] = tf.reduce_mean(self.sol_y)
        statistics["rank_corr"] = rank_correlation

        return statistics
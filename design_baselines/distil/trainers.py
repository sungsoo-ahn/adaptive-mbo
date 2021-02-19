import random
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from design_baselines.utils import spearman

class Trainer(tf.Module):
    def __init__(
        self,
        pess_model,
        pess_model_opt,
        distil_model,
        distil_model_opt,
        init_sol_x,
        sol_x_opt,
        coef_pessimism,
        mc_evals,
    ):

        super().__init__()
        self.pess_model = pess_model
        self.pess_model_opt = pess_model_opt
        self.distil_model = distil_model
        self.distil_model_opt = distil_model_opt

        self.init_sol_x = init_sol_x
        self.sol_x = tf.Variable(init_sol_x)
        self.sol_y = tf.Variable(tf.zeros([tf.shape(init_sol_x)[0], 1]))
        self.sol_x_opt = sol_x_opt

        self.coef_pessimism = coef_pessimism
        self.mc_evals = mc_evals

    def get_sol_x(self):
        return self.sol_x.read_value()

    @tf.function(experimental_relax_shapes=True)
    def train_distil_step(self, x, y):
        sol_x = self.sol_x
        with tf.GradientTape() as tape:
            d = self.distil_model.get_distribution(x, training=True)
            loss_model = loss_nll = -tf.reduce_mean(d.log_prob(y))
            rank_corr = spearman(y[:, 0], d.mean()[:, 0])

        grads = tape.gradient(loss_model, self.distil_model.trainable_variables)
        self.distil_model_opt.apply_gradients(zip(grads, self.distil_model.trainable_variables))

        statistics = dict()
        statistics["loss/nll"] = loss_nll
        statistics["loss/model"] = loss_model
        statistics["d/mean"] = tf.reduce_mean(d.mean())
        statistics["d/stddev"] = tf.reduce_mean(d.stddev())
        statistics["d/rank_corr"] = rank_corr

        return statistics

    @tf.function(experimental_relax_shapes=True)
    def train_pess_step(self, x, y):
        sol_x = self.sol_x
        with tf.GradientTape(persistent=True) as tape:
            d = self.pess_model.get_distribution(x, training=True)
            loss_nll = -tf.reduce_mean(d.log_prob(y))
            rank_corr = spearman(y[:, 0], d.mean()[:, 0])

            sol_d = self.pess_model.get_distribution(sol_x, training=True)
            loss_pessimism = tf.reduce_mean(sol_d.mean())

            loss_model = loss_nll + self.coef_pessimism * loss_pessimism

        model_grads = tape.gradient(loss_model, self.pess_model.trainable_variables)
        self.pess_model_opt.apply_gradients(zip(model_grads, self.pess_model.trainable_variables))

        statistics = dict()
        statistics["loss/nll"] = loss_nll
        statistics["loss/pessimism"] = loss_pessimism
        statistics["loss/model"] = loss_model
        statistics["d/mean"] = tf.reduce_mean(d.mean())
        statistics["d/stddev"] = tf.reduce_mean(d.stddev())
        statistics["d/rank_corr"] = rank_corr
        statistics["sol_d/mean"] = tf.reduce_mean(sol_d.mean())
        statistics["sol_d/stddev"] = tf.reduce_mean(sol_d.stddev())

        return statistics

    @tf.function(experimental_relax_shapes=True)
    def inference_step(self):
        sol_d_mean = 0.0
        sol_d_stddev = 0.0
        for _ in range(self.mc_evals):
            d = self.pess_model.get_distribution(self.sol_x, training=True)
            sol_d_mean += d.mean() / self.mc_evals
            sol_d_stddev += d.stddev() / self.mc_evals

        self.sol_y.assign(sol_d_mean)

        statistics = dict()
        statistics["sol_d/mean"] = tf.reduce_mean(sol_d_mean)
        statistics["sol_d/stddev"] = tf.reduce_mean(sol_d_stddev)

        return statistics

    @tf.function(experimental_relax_shapes=True)
    def update_step(self):
        sol_x_grad = 0.0
        sol_d_mean = 0.0
        sol_d_stddev = 0.0
        for _ in range(self.mc_evals):
            with tf.GradientTape() as tape:
                tape.watch(self.sol_x)
                d = self.distil_model.get_distribution(self.sol_x, training=True)
                loss = - d.mean()

            sol_x_grad += tape.gradient(loss, self.sol_x) / self.mc_evals
            sol_d_mean += d.mean() / self.mc_evals
            sol_d_stddev += d.stddev() / self.mc_evals

        self.sol_x_opt.apply_gradients([[sol_x_grad, self.sol_x]])

        statistics = dict()
        statistics["sol_d/mean"] = tf.reduce_mean(sol_d_mean)
        statistics["sol_d/stddev"] = tf.reduce_mean(sol_d_stddev)

        return statistics
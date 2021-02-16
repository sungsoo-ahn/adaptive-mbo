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
        prior,
        sgld_lr,
        sgld_noise_penalty,
        reg_coef,
    ):

        super().__init__()
        self.model = model
        self.model_opt = model_opt
        self.prior = prior

        # parameters for controlling learning rate for negative samples
        self.sgld_lr = sgld_lr
        self.sgld_noise_penalty = sgld_noise_penalty

        #
        self.reg_coef = reg_coef

    @tf.function(experimental_relax_shapes=True)
    def run_markovchains(self, x, steps):
        for step in range(steps):
            with tf.GradientTape() as tape:
                tape.watch(x)
                d = self.model.get_distribution(x)
                energy = self.prior.kl_divergence(d)

            grads = tape.gradient(energy, x)
            noise = tf.random.normal(x.shape, 0, 1)
            x = (
                x
                + 0.5 * self.sgld_lr * grads
                + self.sgld_noise_penalty * np.sqrt(self.sgld_lr) * noise
            )

            if step == 0:
                energy_start = energy
            if step == steps-1:
                energy_end = energy

        statistics = dict()
        statistics["energy_start"] = energy_start
        statistics["energy_end"] = energy_end

        return x, statistics

    @tf.function(experimental_relax_shapes=True)
    def warmup_step(self, x, y):
        with tf.GradientTape(persistent=True) as tape:
            d = self.model.get_distribution(x, training=True)
            loss_nll = -tf.reduce_mean(d.log_prob(y))
            rank_correlation = spearman(y[:, 0], d.mean()[:, 0])
            loss_total = loss_nll

        # take gradient steps on the model
        grads = tape.gradient(loss_total, self.model.trainable_variables)
        grads = [tf.clip_by_norm(g, 1.0) for g in grads]
        self.model_opt.apply_gradients(zip(grads, self.model.trainable_variables))

        statistics = dict()
        statistics["loss/nll"] = loss_nll
        statistics["loss/total"] = loss_total
        statistics["mean"] = tf.reduce_mean(d.mean())
        statistics["stddev"] = tf.reduce_mean(d.stddev())
        statistics["rank_corr"] = rank_correlation

        return statistics

    @tf.function(experimental_relax_shapes=True)
    def train_step(self, x, y, x_neg):
        with tf.GradientTape(persistent=True) as tape:
            #
            d = self.model.get_distribution(x, training=True)
            loss_nll = -tf.reduce_mean(d.log_prob(y))
            rank_correlation = spearman(y[:, 0], d.mean()[:, 0])

            d_neg = self.model.get_distribution(x_neg)
            loss_kl = self.prior.kl_divergence(d_neg)

            loss_total = loss_nll + self.reg_coef * loss_kl

        # take gradient steps on the model
        grads = tape.gradient(loss_total, self.model.trainable_variables)
        grads = [tf.clip_by_norm(g, 1.0) for g in grads]
        self.model_opt.apply_gradients(zip(grads, self.model.trainable_variables))

        statistics = dict()
        statistics["loss/nll"] = loss_nll
        statistics["loss/kl"] = loss_kl
        statistics["loss/total"] = loss_total
        statistics["mean"] = tf.reduce_mean(d.mean())
        statistics["stddev"] = tf.reduce_mean(d.stddev())
        statistics["rank_corr"] = rank_correlation

        return statistics


class Solver(tf.Module):
    def __init__(
        self,
        model,
        prior,
        sol,
        sol_optim,
        stddev_coef,
    ):
        super().__init__()
        self.model = model
        self.prior = prior

        self.sol = sol
        self.init_sol = sol.read_value()
        self.sol_optim = sol_optim
        self.stddev_coef = stddev_coef

    @tf.function(experimental_relax_shapes=True)
    def solve_step(self):
        with tf.GradientTape() as tape:
            tape.watch(self.sol)
            d = self.model.get_distribution(self.sol, training=False)
            loss = -d.mean() + self.stddev_coef * tf.math.log(d.stddev())
            prior_kl = self.prior.kl_divergence(d)

        solution_grad = tape.gradient(loss, self.sol)
        self.sol_optim.apply_gradients([[solution_grad, self.sol]])

        travelled = tf.linalg.norm(self.sol - self.init_sol) / tf.cast(
            tf.shape(self.sol)[0], dtype=tf.float32
        )

        statistics = dict()
        statistics["loss"] = tf.reduce_mean(loss)
        statistics["prior_kl"] = tf.reduce_mean(prior_kl)
        statistics["mean"] = tf.reduce_mean(d.mean())
        statistics["stddev"] = tf.reduce_mean(d.stddev())
        statistics["travelled"] = travelled
        return statistics
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
        perturb_fn,
        is_discrete,
        sol_x,
        sol_x_opt,
        coef_sol,
    ):

        super().__init__()
        self.model = model
        self.model_opt = model_opt
        self.perturb_fn = perturb_fn
        self.is_discrete = is_discrete
        self.init_sol_x = sol_x
        self.sol_x = tf.Variable(sol_x)
        self.sol_y = sol_y
        self.sol_x_opt = sol_x_opt
        self.coef_sol = coef_sol

        self.num_evals = num_evals

    def get_sol_x(self):
        return self.sol_x.read_value()

    @tf.function(experimental_relax_shapes=True)
    def train_step(self, x, y):
        if self.is_discrete:
            x = tf.math.softmax(x)

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
        if self.is_discrete:
            x = tf.math.softmax(x)

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
        noises = tf.random.normal([self.num_evals] + self.input_shape)
        losses = tf.zeros([self.num_evals, self.solver_samples])
        for eval_idx in range(self.num_evals):
            if self.is_discrete:
                sol_x = tf.math.softmax(self.sol_x) + noises[eval_idx]
            else:
                sol_x = self.sol_x + noises[eval_idx]

            d = self.ema_model.get_distribution(sol_x, training=False)
            losses[eval_idx] = -(d.mean() - self.coef_stddev * tf.math.log(d.stddev()))

        tf.argsort(losses, axis=0)

        sol_x_grad = tape.gradient(loss, self.sol_x)
        sol_x_grad = tf.clip_by_norm(sol_x_grad, 1.0)
        self.sol_x_opt.apply_gradients([[sol_x_grad, self.sol_x]])

        travelled = tf.linalg.norm(self.sol_x - self.init_sol_x) / tf.cast(
            tf.shape(self.sol_x)[0], dtype=tf.float32
        )

        statistics = dict()
        statistics["loss"] = tf.reduce_mean(loss)
        statistics["mean"] = tf.reduce_mean(d.mean())
        statistics["log_stddev"] = tf.reduce_mean(tf.math.log(d.stddev()))
        statistics["travelled"] = travelled

        return statistics
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
        sol_x_list,
        sol_x_opt_list,
        num_evals,
        coef_stddev,
    ):

        super().__init__()
        self.model = model
        self.model_opt = model_opt
        self.perturb_fn = perturb_fn
        self.is_discrete = is_discrete
        self.sol_x_list = [tf.Variable(sol_x) for sol_x in sol_x_list]
        self.sol_x_opt_list = sol_x_opt_list
        self.coef_stddev = coef_stddev
        self.num_evals = num_evals

    def get_sol_x(self):
        return tf.concat(self.sol_x_list, axis=0)

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
        avg_loss = 0.0
        avg_mean = 0.0
        avg_log_stddev = 0.0
        avg_score_stddev = 0.0

        for sol_x, sol_x_opt in zip(self.sol_x_list, self.sol_x_opt_list):
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(sol_x)
                inp = tf.tile(sol_x, [self.num_evals, 1])
                if self.is_discrete:
                    inp = self.perturb_fn(tf.math.softmax(inp))
                else:
                    inp = self.perturb_fn(inp)

                d = self.model.get_distribution(inp, training=False)
                scores = -(d.mean() - self.coef_stddev * tf.math.log(d.stddev()))
                loss = tf.reduce_max(scores)
                #tf.math.top_k(tf.reshape(scores, [-1]), k=(self.num_evals // 2))[0][-1]

            sol_x_grad = tape.gradient(loss, sol_x)
            sol_x_grad = tf.clip_by_norm(sol_x_grad, 1.0)
            sol_x_opt.apply_gradients([[sol_x_grad, sol_x]])

            avg_loss += loss / len(self.sol_x_list)
            avg_mean += d.mean() / len(self.sol_x_list)
            avg_log_stddev += tf.math.log(d.stddev()) / len(self.sol_x_list)
            avg_score_stddev += tf.math.reduce_std(scores) / len(self.sol_x_list)

        statistics = dict()
        statistics["loss"] = tf.reduce_mean(loss)
        statistics["mean"] = tf.reduce_mean(d.mean())
        statistics["log_stddev"] = tf.reduce_mean(tf.math.log(d.stddev()))
        statistics["score_std"] = tf.reduce_mean(avg_score_stddev)

        return statistics
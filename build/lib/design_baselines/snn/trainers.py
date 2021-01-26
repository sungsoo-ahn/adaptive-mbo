from design_baselines.utils import spearman
from design_baselines.utils import soft_noise
from design_baselines.utils import cont_noise
from collections import defaultdict
from tensorflow_probability import distributions as tfpd
import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np
import random


class Smoothing(tf.Module):
    def __init__(
        self,
        models,
        model_optim,
        model_lr,
        ema_models,
        ema_rate,
        sol_x,
        sol_x_optim,
        sol_x_lr,
        mc_evals,
        smoothing_coef,
        is_discrete,
        continuous_noise_std,
        discrete_smoothing,
    ):
        super().__init__()
        self.models = models
        self.model_optims = [model_optim(learning_rate=model_lr) for _ in self.models]
        self.ema_models = ema_models
        self.ema_rate = ema_rate
        self.init_sol_x = sol_x.read_value()
        self.sol_x = sol_x
        self.sol_x_optim_ = sol_x_optim
        if self.sol_x_optim_ == "sgd":
            self.sol_x_optim = tf.keras.optimizers.SGD(learning_rate=sol_x_lr)
        elif self.sol_x_optim_ == "adam":
            self.sol_x_optim = tf.keras.optimizers.Adam(learning_rate=sol_x_lr)

        self.mc_evals = mc_evals

        self.smoothing_coef = smoothing_coef

        self.is_discrete = is_discrete

        self.noise_std = continuous_noise_std
        self.keep = discrete_smoothing

    def update_ema_models(self):
        for model, ema_model in zip(self.models, self.ema_models):
            for var, ema_var in zip(model.trainable_variables, ema_model.trainable_variables):
                ema_var.assign(self.ema_rate * ema_var + (1 - self.ema_rate) * var)

    @tf.function(experimental_relax_shapes=True)
    def train_step(self, x0, y0, b0):
        # corrupt the inputs with noise
        if self.is_discrete:
            x0 = soft_noise(x0, self.keep)
            x1 = tf.math.softmax(self.sol_x)
            x1 = tf.concat([x1, x1], axis=0)

        else:
            x0 = cont_noise(x0, self.noise_std)
            x1 = cont_noise(self.sol_x, self.noise_std)
            x1 = tf.concat([x1, x1], axis=0)

        avg_total_loss = 0.0
        avg_nll_loss = 0.0
        avg_smoothing_loss = 0.0
        avg_rank_correlation = 0.0
        avg_grad_norm = 0.0
        avg_pred_mean0 = 0.0
        avg_pred_std0 = 0.0
        avg_pred_mean1 = 0.0
        avg_pred_std1 = 0.0

        for model_idx, (model, model_optim, ema_model) in enumerate(
            zip(self.models, self.model_optims, self.ema_models)
        ):
            with tf.GradientTape() as tape:
                d0 = model.get_distribution(x0, training=True)
                nll_loss = -d0.log_prob(y0)
                if len(self.models) > 1:
                    nll_loss = tf.math.divide_no_nan(
                        tf.reduce_sum(b0[:, model_idx] * nll_loss), tf.reduce_sum(b0[:, model_idx])
                    )

                rank_correlation = spearman(y0[:, 0], d0.mean()[:, 0])

                # model loss that combines maximum likelihood
                params1 = model.get_params(x1, training=True)
                loc10, loc11 = tf.split(params1["loc"], 2, axis=0)
                scale10, scale11 = tf.split(params1["scale"], 2, axis=0)
                smoothing_loss = 0.5 * (
                    tf.square(loc10 - loc11) + tf.square(scale10 ** 2 - scale11 ** 2)
                )

                total_loss = tf.reduce_mean(nll_loss) + self.smoothing_coef * tf.reduce_mean(
                    smoothing_loss
                )

            grads = tape.gradient(total_loss, model.trainable_variables)
            grad_norm = tf.reduce_mean([tf.linalg.norm(grad) for grad in grads])
            grads = [tf.clip_by_norm(grad, 1.0) for grad in grads]
            model_optim.apply_gradients(zip(grads, model.trainable_variables))

            avg_total_loss += total_loss / len(self.models)
            avg_nll_loss += nll_loss / len(self.models)
            avg_smoothing_loss += smoothing_loss / len(self.models)
            avg_rank_correlation += rank_correlation / len(self.models)
            avg_grad_norm += grad_norm / len(self.models)
            avg_pred_mean0 += tf.reduce_mean(d0.mean()) / len(self.models)
            avg_pred_std0 += tf.reduce_mean(d0.stddev()) / len(self.models)
            avg_pred_mean1 += 0.5 * tf.reduce_mean(loc10 + loc11) / len(self.models)
            avg_pred_std1 += 0.5 * tf.reduce_mean(scale10 + scale11) / len(self.models)

        self.update_ema_models()

        statistics = dict()
        statistics["total_loss"] = total_loss
        statistics["nll_loss"] = avg_nll_loss
        statistics["rank_corr"] = avg_rank_correlation
        statistics["smoothing_loss"] = avg_smoothing_loss
        statistics["grad_norm"] = avg_grad_norm
        statistics["mean0"] = avg_pred_mean0
        statistics["std0"] = avg_pred_std0
        statistics["mean1"] = avg_pred_mean1
        statistics["std1"] = avg_pred_std1

        return statistics

    @tf.function(experimental_relax_shapes=True)
    def validate_step(self, x, y):
        nll_loss = 0.0
        rank_correlation = 0.0
        for ema_model in self.ema_models:
            d = ema_model.get_distribution(x, training=False)
            nll_loss += -d.log_prob(y) / len(self.ema_models)
            rank_correlation += spearman(y[:, 0], d.mean()[:, 0]) / len(self.ema_models)

        statistics = dict()
        statistics["nll_loss"] = nll_loss
        statistics["rank_corr"] = rank_correlation
        statistics["mean"] = tf.reduce_mean(d.mean())
        statistics["std"] = tf.reduce_mean(d.stddev())

        return statistics

    @tf.function(experimental_relax_shapes=True)
    def update_solution(self):
        sol_x_grad = 0.0
        sol_y_pred = 0.0
        for ema_model in self.ema_models:
            for mc_eval in range(self.mc_evals):
                with tf.GradientTape() as tape:
                    tape.watch(self.sol_x)
                    inp = tf.math.softmax(self.sol_x) if self.is_discrete else self.sol_x
                    d = ema_model.get_distribution(
                        inp, training=(True if self.mc_evals > 1 else False)
                        )
                    sol_x_loss = -d.mean()

                sol_x_grad += (
                    tape.gradient(sol_x_loss, self.sol_x) / self.mc_evals / len(self.ema_models)
                )
                sol_y_pred += d.mean() / self.mc_evals / len(self.ema_models)

        normalized_sol_x_grad, grad_norm = tf.linalg.normalize(sol_x_grad)
        sol_x_grad = tf.clip_by_norm(sol_x_grad, 1.0)
        self.sol_x_optim.apply_gradients([[sol_x_grad, self.sol_x]])

        travelled = tf.linalg.norm(self.sol_x - self.init_sol_x) / tf.cast(
            tf.shape(self.sol_x)[0], dtype=tf.float32
        )

        statistics = dict()
        statistics["sol_y_pred"] = tf.reduce_mean(sol_y_pred)
        statistics["grad_norm"] = grad_norm
        statistics["travelled"] = travelled

        return statistics


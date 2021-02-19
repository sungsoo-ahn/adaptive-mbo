from design_baselines.utils import spearman
import tensorflow as tf
import numpy as np
import random

def optimize_linear(grad, eps, norm=np.inf):
    axis = list(range(1, len(grad.get_shape())))
    avoid_zero_div = 1e-12
    if norm == np.inf:
        optimal_perturbation = tf.sign(grad)
        optimal_perturbation = tf.stop_gradient(optimal_perturbation)
    elif norm == 1:
        abs_grad = tf.abs(grad)
        sign = tf.sign(grad)
        max_abs_grad = tf.reduce_max(abs_grad, axis, keepdims=True)
        tied_for_max = tf.dtypes.cast(tf.equal(abs_grad, max_abs_grad), dtype=tf.float32)
        num_ties = tf.reduce_sum(tied_for_max, axis, keepdims=True)
        optimal_perturbation = sign * tied_for_max / num_ties
    elif norm == 2:
        square = tf.maximum(avoid_zero_div, tf.reduce_sum(tf.square(grad), axis, keepdims=True))
        optimal_perturbation = grad / tf.sqrt(square)
    else:
        raise NotImplementedError("Only L-inf, L1 and L2 norms are currently implemented.")

    scaled_perturbation = tf.multiply(eps, optimal_perturbation)
    return scaled_perturbation


def clip_eta(eta, norm, eps):
    if norm not in [np.inf, 1, 2]:
        raise ValueError("norm must be np.inf, 1, or 2.")
    axis = list(range(1, len(eta.get_shape())))
    avoid_zero_div = 1e-12
    if norm == np.inf:
        eta = tf.clip_by_value(eta, -eps, eps)
    else:
        if norm == 1:
            raise NotImplementedError("")
        elif norm == 2:
            norm = tf.sqrt(
                tf.maximum(avoid_zero_div, tf.reduce_sum(tf.square(eta), axis, keepdims=True))
            )
        factor = tf.minimum(1.0, tf.math.divide(eps, norm))
        eta = eta * factor
    return eta


def random_lp_vector(shape, ord, eps, dtype=tf.float32, seed=None):
    if ord not in [np.inf, 1, 2]:
        raise ValueError("ord must be np.inf, 1, or 2.")

    if ord == np.inf:
        r = tf.random.uniform(shape, -eps, eps, dtype=dtype, seed=seed)
    else:
        dim = tf.reduce_prod(shape[1:])

        if ord == 1:
            x = random_laplace((shape[0], dim), loc=1.0, scale=1.0, dtype=dtype, seed=seed)
            norm = tf.reduce_sum(tf.abs(x), axis=-1, keepdims=True)
        elif ord == 2:
            x = tf.random.normal((shape[0], dim), dtype=dtype, seed=seed)
            norm = tf.sqrt(tf.reduce_sum(tf.square(x), axis=-1, keepdims=True))
        else:
            raise ValueError("ord must be np.inf, 1, or 2.")

        w = tf.pow(
            tf.random.uniform((shape[0], 1), dtype=dtype, seed=seed), 1.0 / tf.cast(dim, dtype)
        )
        r = eps * tf.reshape(w * x / norm, shape)

    return r


class Trainer(tf.Module):
    def __init__(
        self,
        model,
        model_opt,
        ema_model,
        adv_eps,
        adv_steps,
        adv_norm,
        perturb_fn,
        is_discrete,
        sol_x,
        coef_pess,
        ema_rate,
    ):

        super().__init__()

        self.model = model
        self.model_opt = model_opt
        self.ema_model = ema_model
        self.perturb_fn = perturb_fn
        self.is_discrete = is_discrete
        self.init_sol_x = sol_x
        self.sol_x = tf.Variable(sol_x)
        self.coef_pess = coef_pess

        self.adv_eps = adv_eps
        self.adv_steps = adv_steps
        self.adv_norm = adv_norm
        self.adv_eps_iter = 2.5 * adv_eps / adv_steps

        self.ema_rate = ema_rate

    def get_sol_x(self):
        return self.sol_x.read_value().numpy()

    @tf.function(experimental_relax_shapes=True)
    def adv_perturb(self, model, x, init_eta):
        adv_x = x + init_eta
        for _ in range(self.adv_steps):
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(adv_x)
                d = model.get_distribution(adv_x, training=True)
                adv_loss = d.mean()

            adv_grad = tape.gradient(adv_loss, adv_x)
            optimal_perturbation = optimize_linear(adv_grad, eps=self.adv_eps_iter, norm=self.adv_norm)
            adv_x = adv_x + optimal_perturbation

            eta = adv_x - x
            eta = clip_eta(eta, self.adv_norm, self.adv_eps)
            adv_x = x + eta

        return adv_x

    @tf.function(experimental_relax_shapes=True)
    def train_step(self, x, y):
        x = self.perturb_fn(x)
        sol_x = self.sol_x
        init_eta = random_lp_vector(
            tf.shape(self.sol_x), ord=self.adv_norm, eps=self.adv_eps, dtype=self.sol_x.dtype
            )
        adv_sol_x = self.adv_perturb(self.model, self.sol_x, init_eta=init_eta)

        if self.is_discrete:
            x = tf.math.softmax(inp)
            sol_x = tf.math.softmax(sol_x)
            adv_sol_x = tf.math.softmax(adv_sol_x)

        with tf.GradientTape(persistent=True) as tape:
            d = self.model.get_distribution(x, training=True)
            loss_nll = -d.log_prob(y)
            rank_correlation = spearman(y[:, 0], d.mean()[:, 0])

            inp = tf.concat([sol_x, adv_sol_x], axis=0)
            params = self.model.get_params(inp, training=True)
            sol_d_mean, adv_sol_d_mean = tf.split(params["loc"], 2, axis=0)

            loss_pess = adv_sol_d_mean - sol_d_mean
            loss_total = tf.reduce_mean(loss_nll) + self.coef_pess * tf.reduce_mean(loss_pess)

        # take gradient steps on the model
        grads = tape.gradient(loss_total, self.model.trainable_variables)
        grads = [tf.clip_by_norm(grad, 1.0) for grad in grads]
        self.model_opt.apply_gradients(zip(grads, self.model.trainable_variables))

        for var, ema_var in zip(self.model.trainable_variables, self.ema_model.trainable_variables):
            ema_var.assign(self.ema_rate * ema_var + (1 - self.ema_rate) * var)

        statistics = dict()
        statistics["loss/nll"] = loss_nll
        statistics["loss/pess"] = loss_pess
        statistics["loss/total"] = loss_total
        statistics["d/mean"] = tf.reduce_mean(d.mean())
        statistics["d/rank_corr"] = rank_correlation
        statistics["sol_d/mean"] = tf.reduce_mean(sol_d_mean)
        statistics["adv_sol_d/mean"] = tf.reduce_mean(adv_sol_d_mean)

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
        adv_sol_x = self.adv_perturb(self.ema_model, self.sol_x, init_eta=0.0)
        self.sol_x.assign(adv_sol_x)

        travelled = tf.linalg.norm(self.sol_x - self.init_sol_x) / tf.cast(
            tf.shape(self.sol_x)[0], dtype=tf.float32
        )

        statistics = dict()
        statistics["travelled"] = travelled

        return statistics

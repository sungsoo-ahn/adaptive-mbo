from design_baselines.utils import soft_noise
from design_baselines.utils import cont_noise
from collections import defaultdict
import tensorflow as tf
import numpy as np


class DiffusionRecoveryLikelihood(tf.Module):
    def __init__(
        self,
        forward_model,
        forward_model_opt,
        forward_model_lr,
        target_scale,
        alpha,
        T,
        K,
        b,
        max_sigma,
        is_discrete,
        continuous_noise_std,
        discrete_smoothing,
    ):

        super().__init__()
        self.forward_model = forward_model
        self.forward_model_opt = forward_model_opt(learning_rate=forward_model_lr)

        self.target_scale = target_scale
        #
        self.alpha = alpha

        #
        self.T = T
        self.K = K
        self.b = b
        self.sigma_scale = max_sigma / self.T

        # extra parameters for controlling data noise
        self.is_discrete = is_discrete
        self.continuous_noise_std = continuous_noise_std
        self.discrete_smoothing = discrete_smoothing

    @tf.function(experimental_relax_shapes=True)
    def train_step(self, x, energy_target):
        statistics = dict()

        # corrupt the inputs with noise
        x = (
            soft_noise(x, self.discrete_smoothing)
            if self.is_discrete
            else cont_noise(x, self.continuous_noise_std)
        )

        #
        xt = x
        max_t = np.random.randint(self.T)
        for t in range(max_t+1):
            sigmatp = self.sigma_scale * (t + 1)
            epstp = tf.random.normal(xt.shape)
            yt = np.sqrt(1-sigmatp**2) * xt
            xtp = yt + sigmatp * epstp
            xt = xtp

        ytm = xtp
        sigmat = self.sigma_scale * max_t
        for tau in range(self.K):
            with tf.GradientTape() as tape:
                tape.watch(ytm)
                energy = self.forward_model(tf.math.softmax(ytm) if self.is_discrete else ytm)

            grads = tape.gradient(energy, ytm)
            epsilontau = tf.random.normal(ytm.shape)
            term0 = 0.5 * (self.b ** 2) * (sigmat ** 2) * grads
            term1 = 0.5 * (self.b ** 2) * (xtp - ytm)
            term2 = self.b * sigmat * epsilontau
            ytm = ytm + term0 + term1 + term2

        with tf.GradientTape(persistent=True) as tape:
            energy = self.forward_model(tf.math.softmax(x) if self.is_discrete else ytm, training=True)
            loss0 = tf.keras.losses.MSE(self.target_scale*energy_target, energy)
            statistics[f"train/loss0"] = loss0

            pos_energy = self.forward_model(
                tf.math.softmax(yt) if self.is_discrete else yt, training=True
                )
            loss1 = - pos_energy
            statistics[f"train/loss1"] = loss1

            neg_energy = self.forward_model(
                tf.math.softmax(ytm) if self.is_discrete else ytm, training=True
                )
            loss2 = neg_energy
            statistics[f"train/loss2"] = loss2

            #total_loss = loss1 + loss2
            total_loss = loss0 + self.alpha * (loss1 + loss2)
            statistics[f"train/total_loss"] = total_loss
            #print(loss)

        # take gradient steps on the model
        grads = tape.gradient(total_loss, self.forward_model.trainable_variables)
        self.forward_model_opt.apply_gradients(zip(grads, self.forward_model.trainable_variables))

        return statistics

    def run_markovchain(self, xT):
        xtp = xT
        for t in range(self.T-1, -1, -1):
            sigmat = self.sigma_scale * t
            yt = xtp
            for tau in range(self.K):
                with tf.GradientTape() as tape:
                    tape.watch(yt)
                    energy = self.forward_model(tf.math.softmax(yt) if self.is_discrete else yt)

                grads = tape.gradient(energy, yt)
                epsilontau = tf.random.normal(yt.shape)
                term0 = 0.5 * (self.b ** 2) * (sigmat ** 2) * grads
                term1 = 0.5 * (self.b ** 2) * (xtp - yt)
                term2 = self.b * sigmat * epsilontau
                yt = yt + term0 + term1 + term2

            sigmatp = self.sigma_scale * (t + 1)
            xt = yt / np.sqrt(1 - sigmatp**2)
            xtp = xt

        return xt


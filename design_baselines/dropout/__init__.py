from design_baselines.data import StaticGraphTask
from design_baselines.logger import Logger
from design_baselines.utils import soft_noise, cont_noise
from design_baselines.dropout.trainers import Smoothing
from design_baselines.dropout.nets import ForwardModel
from collections import defaultdict
import tensorflow_probability as tfp
import tensorflow as tf
import scipy.spatial
import numpy as np
import os
import glob

from ray import tune
from tensorboard.plugins.hparams import api as hp


def dropout(config):
    """Train a Score Function to solve a Model-Based Optimization
    using gradient ascent on the input design

    Args:

    config: dict
        a dictionary of hyper parameters such as the learning rate
    """

    # create the training task and logger
    logger = Logger(config["logging_dir"])
    task = StaticGraphTask(config["task"], **config["task_kwargs"])

    # save the initial dataset statistics for safe keeping
    x = task.x
    y = task.y

    if config["normalize_ys"]:
        # compute normalization statistics for the score
        mu_y = np.mean(y, axis=0, keepdims=True)
        mu_y = mu_y.astype(np.float32)
        y = y - mu_y
        st_y = np.std(y, axis=0, keepdims=True)
        st_y = np.where(np.equal(st_y, 0), 1, st_y)
        st_y = st_y.astype(np.float32)
        y = y / st_y

    else:

        # compute normalization statistics for the score
        mu_y = np.zeros_like(y[:1])
        st_y = np.ones_like(y[:1])

    if config["normalize_xs"] and not config["is_discrete"]:

        # compute normalization statistics for the data vectors
        mu_x = np.mean(x, axis=0, keepdims=True)
        mu_x = mu_x.astype(np.float32)
        x = x - mu_x
        st_x = np.std(x, axis=0, keepdims=True)
        st_x = np.where(np.equal(st_x, 0), 1, st_x)
        st_x = st_x.astype(np.float32)
        x = x / st_x

    else:

        # compute normalization statistics for the score
        mu_x = np.zeros_like(x[:1])
        st_x = np.ones_like(x[:1])

    total_x = x
    total_y = y

    # create a bootstrapped data set
    train_data, validate_data = task.build(
        x=x,
        y=y,
        batch_size=config["batch_size"],
        val_size=config["val_size"],
        bootstraps=config["num_models"],
    )

    models = [
        ForwardModel(
            input_shape=task.input_shape,
            hidden=config["hidden_size"],
            dropout_type=config["dropout_type"],
            dropout_rate=config["dropout_rate"],
        )
        for _ in range(config["num_models"])
    ]

    ema_models = [
        ForwardModel(
            input_shape=task.input_shape,
            hidden=config["hidden_size"],
            dropout_type=config["dropout_type"],
            dropout_rate=config["dropout_rate"],
        )
        for _ in range(config["num_models"])
    ]

    for ema_model, model in zip(ema_models, models):
        ema_model.set_weights(model.get_weights())

    indices = tf.math.top_k(y[:, 0], k=config["solver_samples"])[1]
    sol_x = initial_x = tf.gather(x, indices, axis=0)
    sol_x = (
        tf.math.log(soft_noise(sol_x, config["discrete_smoothing"]))
        if config["is_discrete"]
        else sol_x
    )
    sol_x = tf.Variable(sol_x)

    trainer = Smoothing(
        models=models,
        model_optim=tf.keras.optimizers.Adam,
        model_lr=config["model_lr"],
        ema_models=ema_models,
        ema_rate=config["ema_rate"],
        sol_x=sol_x,
        sol_x_optim=config["sol_x_optim"],
        sol_x_lr=config["sol_x_lr"],
        mc_evals=config["mc_evals"],
        smoothing_coef=config["smoothing_coef"],
        is_discrete=config["is_discrete"],
        continuous_noise_std=config.get("continuous_noise_std", 0.0),
        discrete_smoothing=config.get("discrete_smoothing", 0.6),
    )

    update = 0
    step = 0
    epoch = 0
    while update < config["updates"]:
        epoch += 1
        train_statistics = defaultdict(list)
        for x, y, b in train_data:
            step += 1
            for name, tsr in trainer.train_step(x, y, b).items():
                train_statistics[name].append(tsr)

            if step > config["warmup"] and step % config["update_freq"] == 0:
                update += 1
                update_statistics = trainer.update_solution()
                for name, tsr in update_statistics.items():
                    logger.record(f"update/{name}", tsr, update)

                inp = tf.math.softmax(trainer.sol_x) if config["is_discrete"] else trainer.sol_x
                dists = scipy.spatial.distance.cdist(
                    np.reshape(inp, [inp.shape[0], -1]), np.reshape(total_x, [total_x.shape[0], -1])
                )
                dists /= np.prod(total_x.shape[1:])
                min_dists = np.min(dists, axis=1)
                mean_dists = np.mean(dists, axis=1)
                logger.record("update/min_dist", np.mean(min_dists), update)
                logger.record("update/mean_dists", np.mean(mean_dists), update)

                if update % config["score_freq"] == 0:
                    inp = tf.math.softmax(trainer.sol_x) if config["is_discrete"] else trainer.sol_x
                    score = task.score(inp * st_x + mu_x)
                    y = (score - mu_y) / st_y
                    logger.record(f"update/score", score, update, percentile=True)
                    logger.record(
                        f"update/sol_score_pred",
                        update_statistics["sol_y_pred"] * st_y + mu_y,
                        update,
                        percentile=True,
                    )

                    statistics = trainer.validate_step(inp, y)
                    for name, tsr in statistics.items():
                        logger.record(f"update/{name}", tf.reduce_mean(tsr), update)


                if update == config["updates"]:
                    break

                logger.record("update/epoch", epoch, update)
                logger.record("update/step", step, update)

        for name, tsrs in train_statistics.items():
            logger.record(f"train/{name}", tf.reduce_mean(tf.concat(tsrs, axis=0)), epoch)

        validate_statistics = defaultdict(list)
        for x, y in validate_data:
            for name, tsr in trainer.validate_step(x, y).items():
                validate_statistics[name].append(tsr)

        for name, tsrs in validate_statistics.items():
            logger.record(f"validate/{name}", tf.reduce_mean(tf.concat(tsrs, axis=0)), epoch)

        logger.record("train/step", step, epoch)
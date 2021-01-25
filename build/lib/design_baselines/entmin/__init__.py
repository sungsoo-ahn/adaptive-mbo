from design_baselines.data import StaticGraphTask
from design_baselines.logger import Logger
from design_baselines.utils import soft_noise, cont_noise
from design_baselines.entmin.trainers import Buffer, EntMin
from design_baselines.entmin.nets import ForwardModel
from collections import defaultdict
import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np
import os
import glob


def entmin(config):
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

    # create a bootstrapped data set
    train_data, validate_data = task.build(
        x=x, y=y, batch_size=config["batch_size"], val_size=config["val_size"], bootstraps=0,
    )

    #buffer = Buffer()

    indices = tf.math.top_k(y[:, 0], k=config["solver_samples"])[1]
    sol_x = initial_x = tf.gather(x, indices, axis=0)
    sol_x = (
        tf.math.log(soft_noise(sol_x, config["discrete_smoothing"]))
        if config["is_discrete"]
        else cont_noise(sol_x, config["continuous_noise_std"])
    )
    sol_x = tf.Variable(sol_x)
    x_optim = tf.keras.optimizers.Adam(config["x_lr"])

    model = ForwardModel(input_shape=task.input_shape, hidden=config["hidden_size"])
    trainer = EntMin(
        model,
        forward_model_optim=tf.keras.optimizers.Adam,
        forward_model_lr=config["forward_model_lr"],
        alpha=config["alpha"],
        is_discrete=config["is_discrete"],
        continuous_noise_std=config.get("continuous_noise_std", 0.0),
        discrete_smoothing=config.get("discrete_smoothing", 0.6),
    )

    step = 0
    for epoch in range(config["epochs"]):
        statistics = defaultdict(list)
        warmup = epoch < config["warmup"]
        for x, y in train_data:
            step += 1
            if warmup:
                for name, tsr in trainer.warmup_train_step(x, y).items():
                    statistics[name].append(tsr)
            else:
                x_neg = tf.math.softmax(sol_x) if config["is_discrete"] else sol_x
                for name, tsr in trainer.train_step(x, y, x_neg).items():
                    statistics[name].append(tsr)

        for name, tsr in statistics.items():
            logger.record(f"epoch/{name}", tsr, epoch)

        if not warmup and (epoch + 1) % config["update_freq"] == 0:
            with tf.GradientTape() as tape:
                tape.watch(sol_x)
                inp = tf.math.softmax(sol_x) if config["is_discrete"] else sol_x
                d = model.get_distribution(inp)
                x_loss = -d.mean()

            score_pred = d.mean() * st_y + mu_y
            score = task.score(inp * st_x + mu_x)
            travelled = tf.linalg.norm(inp - initial_x)

            logger.record(f"update/score_pred", score_pred, epoch, percentile=True)
            logger.record(f"update/score", score, epoch, percentile=True)
            logger.record(f"update/d_mean", d.mean(), epoch)
            logger.record(f"update/d_stddev", d.stddev(), epoch)
            logger.record(f"update/travelled", travelled, epoch)

            x_grad = tape.gradient(x_loss, sol_x)
            x_optim.apply_gradients([[x_grad, sol_x]])
            #sol_x = sol_x + config["x_lr"] * x_grad






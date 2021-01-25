from design_baselines.data import StaticGraphTask
from design_baselines.logger import Logger
from design_baselines.utils import spearman
from design_baselines.utils import soft_noise, cont_noise
from design_baselines.utils import render_video
from design_baselines.meta.trainers import MaximumLikelihood
from design_baselines.meta.nets import ForwardModel
from collections import defaultdict
import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np
import os
import glob


def meta(config):
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

    config["outer_step_size"] *= np.sqrt(np.prod(x.shape[1:]))

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

    # scale the learning rate based on the number of channels in x
    config["outer_step_size"] *= np.sqrt(np.prod(x.shape[1:]))

    # make several keras neural networks with different architectures
    model_config = {
        "input_shape": task.input_shape,
        "activations": config["activations"],
        "hidden": config["hidden_size"],
        "initial_max_std": config["initial_max_std"],
        "initial_min_std": config["initial_min_std"],
    }

    # create a bootstrapped data set
    train_data, validate_data = task.build(
        x=x,
        y=y,
        batch_size=config["batch_size"],
        val_size=config["val_size"],
        bootstraps=config["num_models"],
    )

    model = ForwardModel(**model_config)

    # create a trainer for a forward model with a conservative objective
    trainer = MaximumLikelihood(
        model,
        forward_model_optim=tf.keras.optimizers.Adam,
        forward_model_lr=config["forward_model_lr"],
        bootstrap_id=0,
        is_discrete=config["is_discrete"],
        continuous_noise_std=config.get("continuous_noise_std", 0.0),
        discrete_smoothing=config.get("discrete_smoothing", 0.6),
    )

    # train the model for an additional number of epochs
    trainer.launch(
        train_data,
        validate_data,
        logger,
        config["epochs"],
        header=f"pretrain_model/",
    )

    # select the top k initial designs from the dataset
    indices = tf.math.top_k(y[:, 0], k=config["solver_samples"])[1]
    initial_x = tf.gather(x, indices, axis=0)
    x = (
        tf.math.log(soft_noise(initial_x, config.get("discrete_smoothing", 0.6)))
        if config["is_discrete"]
        else initial_x
    )
    x = tf.Variable(x)

    # evaluate the starting point
    inp = tf.math.softmax(x) if config["is_discrete"] else x
    score = task.score(inp * st_x + mu_x)

    # record the prediction and score to the logger
    logger.record("score", score, 0, percentile=True)

    if config["optimizer"] == "adam":
        x_optim = tf.keras.optimizers.Adam(learning_rate=config["outer_step_size"])
    elif config["optimizer"] == "sgd":
        x_optim = tf.keras.optimizers.SGD(learning_rate=config["outer_step_size"])

    min_clip_weights = [weight - config["weight_clip_val"] for weight in model.inner_weights]
    max_clip_weights = [weight + config["weight_clip_val"] for weight in model.inner_weights]

    """
    fims = [config["min_fim"] for _ in model.inner_weights]
    for sample_idx, (xi, yi) in enumerate(zip(x, y)):
        xi = np.expand_dims(xi, 0)
        yi = np.expand_dims(yi, 0)
        xi0 = (
            soft_noise(xi, trainer.keep)
            if config["is_discrete"]
            else cont_noise(xi, trainer.noise_std)
        )
        with tf.GradientTape() as tape:
            nll = -tf.reduce_mean(model.get_distribution(xi0).log_prob(yi))

        grads = tape.gradient(nll, model.inner_weights)
        square_grads = [tf.square(grad) for grad in grads]
        fims = [
            fim + square_grad / x.shape[0] for fim, square_grad in zip(fims, square_grads)
        ]

        for fim_idx, fim in enumerate(fims):
            logger.record(f"fim/grad2_layer{fim_idx}", tf.reduce_mean(fim), sample_idx)
    """

    for outer_step in range(config["outer_steps"]):
        with tf.GradientTape() as outer_tape:
            outer_tape.watch(x)
            inp = tf.math.softmax(x) if config["is_discrete"] else x

            copied_model = ForwardModel.copy_from(model_config, model.inner_weights)
            for inner_step in range(config["inner_steps"]):
                with tf.GradientTape(watch_accessed_variables=False) as inner_tape:
                    inner_tape.watch(copied_model.inner_weights)
                    model_loss = copied_model.get_distribution(inp).mean()

                model_grads = inner_tape.gradient(model_loss, copied_model.inner_weights)
                copied_model = ForwardModel.copy_from(
                    model_config,
                    copied_model.inner_weights,
                    grads=model_grads,
                    step_size=config["inner_step_size"],
                    #clip_weights=(min_clip_weights, max_clip_weights),
                )
                logger.record(
                    f"inner_train_model/{outer_step:03d}th_loss",
                    model_loss,
                    inner_step,
                )

            model_loss = copied_model.get_distribution(inp).mean()
            if config["inner_steps"] > 0:
                logger.record(
                    f"inner_train_model/{outer_step:03d}th_loss",
                    model_loss,
                    inner_step + 1,
                )

            x_loss = -model_loss

        x_grad = outer_tape.gradient(x_loss, x)
        x_optim.apply_gradients([[x_grad, x]])

        inp = tf.math.softmax(x) if config["is_discrete"] else x
        logger.record("train/travelled", tf.linalg.norm(inp - initial_x), outer_step)

        if (outer_step + 1) % config["log_freq"] == 0:
            y_pred = model_loss
            logger.record("pred", y_pred * st_y + mu_y, outer_step, percentile=True)

            inp = tf.math.softmax(x) if config["is_discrete"] else x
            score = task.score(inp * st_x + mu_x)
            logger.record("score", score, outer_step, percentile=True)

            org_y_pred = model.get_distribution(inp).mean()
            logger.record("org_pred", org_y_pred * st_y + mu_y, outer_step, percentile=True)


"""
def validate_model(model_):
    statistics = defaultdict(list)
    for x_, y_ in validate_data:
        x_ = (
            soft_noise(x_, config.get("discrete_smoothing", 0.6))
            if config["is_discrete"]
            else cont_noise(x_, config.get("continuous_noise_std", 0.0))
        )

        d = model_.get_distribution(x_, training=False)
        nll = -d.log_prob(y_)

        rank_correlation = spearman(y_[:, 0], d.mean()[:, 0])

        statistics[f"validate/nll"].append(nll)
        statistics[f"validate/rank_corr"].append(rank_correlation)

    return statistics

statistics = validate_model(copied_model)
for name in statistics.keys():
    logger.record(name, tf.concat(statistics[name], axis=0), outer_step)
"""

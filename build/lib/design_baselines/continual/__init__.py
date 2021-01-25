from design_baselines.data import StaticGraphTask
from design_baselines.logger import Logger
from design_baselines.utils import spearman
from design_baselines.utils import soft_noise
from design_baselines.utils import render_video
from design_baselines.continual.trainers import MaximumLikelihood
from design_baselines.continual.nets import ForwardModel
from collections import defaultdict
import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np
import os
import glob


def continual(config):
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
    model = ForwardModel(**model_config)
    copied_model = ForwardModel(**model_config)
    # create a bootstrapped data set
    train_data, validate_data = task.build(
        x=x, y=y, batch_size=config["batch_size"], val_size=config["val_size"], bootstraps=1
    )

    # create a trainer for a forward model with a conservative objective
    trainer = MaximumLikelihood(
        model,
        forward_model_optim=tf.keras.optimizers.Adam,
        forward_model_lr=config["forward_model_lr"],
        is_discrete=config["is_discrete"],
        continuous_noise_std=config.get("continuous_noise_std", 0.0),
        discrete_smoothing=config.get("discrete_smoothing", 0.6),
    )

    # train the model for an additional number of epochs
    trainer.launch(train_data, validate_data, logger, config["epochs"], header=f"oracle/")

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

    # select the top k initial designs from the dataset
    indices = tf.math.top_k(y[:, 0], k=config["solver_samples"])[1]
    initial_x = tf.gather(x, indices, axis=0)
    x = (
        tf.math.log(soft_noise(initial_x, config.get("discrete_smoothing", 0.6)))
        if config["is_discrete"]
        else initial_x
    )

    x = tf.Variable(x, trainable=True)

    # evaluate the starting point
    solution = tf.math.softmax(x) if config["is_discrete"] else x
    score = task.score(solution * st_x + mu_x)

    # record the prediction and score to the logger
    logger.record("score", score, 0, percentile=True)

    x_optim = tf.keras.optimizers.Adam(learning_rate=config["outer_step_size"])
    for outer_step in range(config["outer_steps"]):
        #
        copied_model.set_weights(model.get_weights())
        model_optim = tf.keras.optimizers.Adam(learning_rate=config["inner_step_size"])
        #
        inp = tf.math.softmax(x) if config["is_discrete"] else x
        org_y_pred = model.get_distribution(inp).mean()
        logger.record("org_pred", org_y_pred * st_y + mu_y, outer_step, percentile=True)

        for inner_step in range(config["inner_steps"]):

            with tf.GradientTape() as inner_tape:
                inp = tf.math.softmax(x) if config["is_discrete"] else x
                loss0 = copied_model.get_distribution(inp).mean()
                loss1 = copied_model.get_ewc_loss(
                    fims, model.inner_weights, delta=config["huber_delta"]
                    )
                loss = loss0 + config["ewc_coef"] * loss1

            model_grads = inner_tape.gradient(loss, copied_model.trainable_variables)
            model_grads = [tf.clip_by_norm(grads, 0.1) for grads in model_grads]
            model_optim.apply_gradients(zip(grads, copied_model.trainable_variables))

            logger.record(f"inner_train/{outer_step}_loss", tf.reduce_mean(loss), inner_step)
            logger.record(f"inner_train/{outer_step}_loss0", tf.reduce_mean(loss0), inner_step)
            logger.record(f"inner_train/{outer_step}_loss1", tf.reduce_mean(loss1), inner_step)

        with tf.GradientTape() as outer_tape:

            outer_tape.watch(x)
            inp = tf.math.softmax(x) if config["is_discrete"] else x
            y_pred = copied_model.get_distribution(inp).mean()
            loss = -y_pred

        x_grads = outer_tape.gradient(loss, [x])
        x_optim.apply_gradients(zip(x_grads, [x]))

        logger.record("pred", y_pred * st_y + mu_y, outer_step, percentile=True)

        #
        score = task.score(x * st_x + mu_x)
        logger.record("score", score, outer_step, percentile=True)